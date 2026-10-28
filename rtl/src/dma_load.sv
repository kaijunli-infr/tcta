// LOAD engine (LOAD.2D / LOAD.3D / LOAD.IDX, ISA §5.1-§5.2): one row at a time -- fetch the row's bytes
// (DMA_BYTES per beat, any alignment, fixed-latency memory model), dequantise BANKS elements per cycle
// (dequant_elem), round to BF16 for INP/WGT destinations, write one row slice per cycle into the scratchpad.
// LOAD.IDX reads its indices from a VEC row tile (16 per scratchpad read), skips -1 rows with a zero fill and
// raises IDX_RANGE / IDX_SHORT like the simulator. Rows are processed sequentially in v0 (no row overlap).
module dma_load
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
(
  input  logic              clk,
  input  logic              rst_n,
  // instruction
  input  instr_t            ins,
  input  logic              issue,
  output logic              busy,
  output logic              done,
  // descriptor table (1-cycle read)
  output logic [7:0]        desc_idx,
  input  logic [DESC_W-1:0] desc_data,
  // memory read port
  output logic              mem_valid,
  output logic [ADDR_W-1:0] mem_addr,
  input  logic              mem_rvalid,
  input  logic [MEM_W-1:0]  mem_rdata,
  input  logic              mem_rerr,
  // scratchpad client (writes the destination, reads the index tile)
  output sp_req_t           sp_req,
  input  logic              sp_grant,
  input  logic              sp_rvalid,
  input  slice_t            sp_rdata,
  // tile metadata
  input  tmeta_t            meta [N_SLOTS],
  output logic              meta_we,
  output logic [7:0]        meta_idx,
  output tmeta_t            meta_wdata,
  input  logic              meta_ack,
  // fault (sticky until the next issue)
  output logic              fault,
  output fault_t            fault_code
);
  typedef enum logic [3:0] {S_IDLE, S_DESC, S_SETUP, S_IDX_RD, S_IDX_WAIT, S_ROW, S_FETCH, S_WAIT, S_DEQ, S_ZFILL,
                            S_META, S_DONE, S_FAULT} state_t;
  state_t st;

  instr_t      cur;
  desc_t       d;
  logic        is_idx;
  logic [15:0] dst;
  logic [11:0] idx_slot;
  int          rows, cols, row0, col0, col_base, n_el_row, nb, n_beats, n_cg;
  int          i, cg, beat_issue, beat_got, n_resp;
  logic        need_scales;        // *.b32: a second fetch of this row's UE8M0 codes
  logic        is_r16;
  logic [63:0] row_addr, scale_addr;
  logic [MEM_W-1:0] rowbuf [ROWBUF_BEATS];
  logic [MEM_W-1:0] scalebuf [2];
  logic [31:0] idx_slice [BANKS];
  logic        idx_have;
  int          idx_cg;
  logic signed [31:0] ri;
  logic        to_bf16;

  // ---- descriptor and geometry helpers ----
  always_comb desc_idx = cur.b[7:0];

  // ---- element extraction window: 5 consecutive beats from the row buffer ----
  int          o0, so0;                        // first byte of this slice's payload / r16 scale byte
  logic [5*MEM_W-1:0] win;
  logic [2*MEM_W-1:0] swin;
  logic [31:0] raw   [BANKS];
  logic [7:0]  sc8   [BANKS];
  logic [31:0] deq   [BANKS];
  logic [15:0] deq_bf [BANKS];
  slice_t      wslice;
  always_comb begin
    int p0, pe, b0, be, w0;
    p0 = col_base + cg * BANKS;
    if (d.dtype == DT_F4_B32 || d.dtype == DT_F4_R16) o0 = p0 >> 1; else o0 = p0 * el_bytes(d.dtype);
    so0 = (n_el_row >> 1) + (p0 >> 4);
    w0 = o0 >> 4;
    for (int k = 0; k < 5; k++) begin
      win[k*MEM_W +: MEM_W] = (w0 + k < ROWBUF_BEATS) ? rowbuf[w0 + k] : '0;
    end
    for (int k = 0; k < 2; k++) begin
      swin[k*MEM_W +: MEM_W] = ((so0 >> 4) + k < ROWBUF_BEATS) ? rowbuf[(so0 >> 4) + k] : '0;
    end
    for (int e = 0; e < BANKS; e++) begin
      pe = p0 + e;
      if (d.dtype == DT_F4_B32 || d.dtype == DT_F4_R16) begin
        be = (pe >> 1) - (o0 & ~32'hf);
        raw[e] = 32'(pe[0] ? win[be*8 + 4 +: 4] : win[be*8 +: 4]);
      end else begin
        be = pe * el_bytes(d.dtype) - (o0 & ~32'hf);
        raw[e] = win[be*8 +: 32];
      end
      case (d.dtype)
        DT_F8_B32, DT_F4_B32: begin
          b0 = (cg * BANKS + e) >> 5;                  // scale codes start at col0 (col0 % 32 == 0)
          sc8[e] = scalebuf[b0 >> 4][(b0 & 15)*8 +: 8];
        end
        DT_F4_R16: begin
          b0 = ((n_el_row >> 1) + (pe >> 4)) - (so0 & ~32'hf);
          sc8[e] = swin[b0*8 +: 8];
        end
        default: sc8[e] = 8'd0;
      endcase
    end
  end
  for (genvar e = 0; e < BANKS; e++) begin : g_deq
    dequant_elem u_dq (.dtype(d.dtype), .raw(raw[e]), .scale8(sc8[e]), .scale_t(d.scale_t), .y(deq[e]));
    f32_to_bf16  u_bf (.x(deq[e]), .y(deq_bf[e]));
    always_comb wslice[e] = to_bf16 ? {deq_bf[e], 16'd0} : deq[e];
  end

  // ---- index conversion ----
  logic signed [31:0] idx_int [BANKS];
  for (genvar e = 0; e < BANKS; e++) begin : g_idx
    f32_to_int_trunc u_i (.x(idx_slice[e]), .y(idx_int[e]));
  end

  // ---- outputs ----
  always_comb begin
    busy = (st != S_IDLE);
    done = (st == S_DONE);
    mem_valid = 1'b0; mem_addr = '0;
    sp_req = '0; meta_we = 1'b0; meta_idx = '0; meta_wdata = '0;
    case (st)
      S_IDX_RD: sp_req = slice_req(SP_VEC, idx_slot, 0, idx_cg, 1'b0, '0, '0);
      S_FETCH: begin
        mem_valid = 1'b1;
        if (beat_issue < n_beats) mem_addr = ADDR_W'(row_addr + 64'(beat_issue) * DMA_BYTES);
        else mem_addr = ADDR_W'(scale_addr + 64'(beat_issue - n_beats) * DMA_BYTES);
      end
      S_DEQ:   sp_req = slice_req(tref_sp(dst), tref_slot(dst), i, cg, 1'b1, slice_ben(cols, cg), wslice);
      S_ZFILL: sp_req = slice_req(tref_sp(dst), tref_slot(dst), i, cg, 1'b1, slice_ben(cols, cg), '0);
      S_META: begin
        meta_we = 1'b1; meta_idx = 8'(slot_index(tref_sp(dst), tref_slot(dst)));
        meta_wdata = '{rows: 10'(rows), cols: 10'(cols), pready: 1'b0};
      end
      default: ;
    endcase
  end

  task automatic set_fault(input fault_t code);
    fault <= 1'b1; fault_code <= code; st <= S_FAULT;
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; fault <= 1'b0; fault_code <= F_NONE; cur <= '0; d <= '0; is_idx <= 0; dst <= '0; idx_slot <= '0;
      rows <= 0; cols <= 0; row0 <= 0; col0 <= 0; col_base <= 0; n_el_row <= 0; nb <= 0; n_beats <= 0; n_cg <= 0;
      i <= 0; cg <= 0; beat_issue <= 0; beat_got <= 0; n_resp <= 0; need_scales <= 0; is_r16 <= 0;
      row_addr <= '0; scale_addr <= '0; idx_have <= 0; idx_cg <= 0; ri <= 0; to_bf16 <= 0;
      for (int k = 0; k < BANKS; k++) idx_slice[k] <= '0;
      scalebuf[0] <= '0; scalebuf[1] <= '0;
    end else begin
      case (st)
        S_IDLE: if (issue) begin
          cur <= ins; fault <= 1'b0; fault_code <= F_NONE; st <= S_DESC;
        end
        S_DESC: st <= S_SETUP;                       // desc_data valid now (registered read of desc_idx = cur.b[7:0])
        S_SETUP: begin
          d <= unpack_desc(desc_data);
          begin
            desc_t dd; int rb; logic [15:0] dstr;
            dd = unpack_desc(desc_data);
            dstr = cur.a;
            dst <= dstr;
            to_bf16 <= (tref_sp(dstr) == SP_INP || tref_sp(dstr) == SP_WGT);
            is_r16 <= (dd.dtype == DT_F4_R16);
            need_scales <= (dd.dtype == DT_F8_B32 || dd.dtype == DT_F4_B32);
            i <= 0; cg <= 0; idx_have <= 1'b0; idx_cg <= -1;
            if (cur.op == OP_LOAD_IDX) begin
              is_idx <= 1'b1;
              rows <= int'(cur.b[15:8]); cols <= int'(cur.imm[15:0]); col0 <= int'(cur.d); row0 <= 0;
              col_base <= int'(cur.d); n_el_row <= int'(dd.shape1); idx_slot <= tref_slot(cur.c);
              rb = int'(cur.imm[31:16]); nb <= rb; n_beats <= (rb + DMA_BYTES - 1) / DMA_BYTES;
              n_cg <= n_slices(int'(cur.imm[15:0]));
              if (tref_sp(dstr) == SP_STATE || tref_sp(dstr) > SP_STATE) set_fault(F_BAD_SHAPE);
              else if (rb % 64 != 0 || rb > ROWBUF_BYTES) set_fault(F_BAD_SHAPE);
              else if (tref_sp(cur.c) != SP_VEC || int'(tref_slot(cur.c)) < VEC_ROW_SLOT0) set_fault(F_BAD_SHAPE);
              else if (int'(meta[slot_index(SP_VEC, tref_slot(cur.c))].cols) < int'(cur.b[15:8])) set_fault(F_IDX_SHORT);
              else if (dd.dtype == DT_F8_B32 || dd.dtype == DT_F4_B32) set_fault(F_BAD_QUALIFIER);
              else if (int'(cur.imm[15:0]) > sp_cols(tref_sp(dstr), tref_slot(dstr))) set_fault(F_BAD_SHAPE);
              else st <= S_ROW;
            end else begin
              is_idx <= 1'b0;
              rows <= int'(cur.imm[31:16]); cols <= int'(cur.imm[15:0]); row0 <= int'(cur.c); col0 <= int'(cur.d);
              col_base <= 0; n_el_row <= int'(cur.imm[15:0]);
              rb = row_bytes_for(dd.dtype, int'(cur.imm[15:0])); nb <= rb; n_beats <= (rb + DMA_BYTES - 1) / DMA_BYTES;
              n_cg <= n_slices(int'(cur.imm[15:0]));
              if (int'(cur.imm[31:16]) > sp_rows(tref_sp(dstr), tref_slot(dstr)) ||
                  int'(cur.imm[15:0]) > sp_cols(tref_sp(dstr), tref_slot(dstr)) || rb > ROWBUF_BYTES) set_fault(F_BAD_SHAPE);
              else if (dd.dtype > DT_F4_R16) set_fault(F_BAD_QUALIFIER);
              else st <= S_ROW;
            end
          end
        end
        // ---- per row ----
        S_ROW: begin
          if (i >= rows) st <= S_META;
          else if (n_cg == 0) i <= i + 1;
          else if (is_idx && idx_cg != (i / BANKS)) st <= S_IDX_RD;
          else begin
            logic signed [31:0] r;
            r = is_idx ? idx_int[i % BANKS] : 32'sd0;
            ri <= r;
            if (is_idx && r == -1) begin cg <= 0; st <= S_ZFILL; end
            else if (is_idx && (r < 0 || r >= $signed(d.shape0))) set_fault(F_IDX_RANGE);
            else begin
              if (is_idx) row_addr <= d.base + 64'(cur.base_off) + 64'(r) * 64'(d.stride0);
              else begin
                int eb; eb = el_bytes(d.dtype);
                if (d.dtype == DT_F4_B32) row_addr <= d.base + 64'(cur.base_off) + 64'(row0 + i) * 64'(d.stride0) + 64'(col0 >> 1);
                else row_addr <= d.base + 64'(cur.base_off) + 64'(row0 + i) * 64'(d.stride0) + 64'(col0 * eb);
              end
              scale_addr <= d.scales_base + 64'(cur.scales_off) + 64'((row0 + i) / 32) * 64'(d.scales_stride) + 64'(col0 / 32);
              beat_issue <= 0; beat_got <= 0; n_resp <= n_beats + (need_scales ? 2 : 0);
              st <= S_FETCH;
            end
          end
        end
        S_IDX_RD: if (sp_grant) begin idx_cg <= i / BANKS; st <= S_IDX_WAIT; end
        S_IDX_WAIT: if (sp_rvalid) begin
          for (int k = 0; k < BANKS; k++) idx_slice[k] <= sp_rdata[k];
          idx_have <= 1'b1; st <= S_ROW;
        end
        S_FETCH: begin
          if (beat_issue + 1 >= n_resp) st <= S_WAIT;
          beat_issue <= beat_issue + 1;
          if (mem_rvalid) begin
            if (mem_rerr) set_fault(F_PAGE_FAULT);
            if (beat_got < n_beats) rowbuf[beat_got] <= mem_rdata; else scalebuf[beat_got - n_beats] <= mem_rdata;
            beat_got <= beat_got + 1;
          end
        end
        S_WAIT: begin
          if (mem_rvalid) begin
            if (mem_rerr) set_fault(F_PAGE_FAULT);
            if (beat_got < n_beats) rowbuf[beat_got] <= mem_rdata; else scalebuf[beat_got - n_beats] <= mem_rdata;
            beat_got <= beat_got + 1;
            if (beat_got + 1 >= n_resp) begin cg <= 0; st <= S_DEQ; end
          end
        end
        S_DEQ: if (sp_grant) begin
          if (cg + 1 >= n_cg) begin i <= i + 1; st <= S_ROW; end
          cg <= cg + 1;
        end
        S_ZFILL: if (sp_grant) begin
          if (cg + 1 >= n_cg) begin i <= i + 1; st <= S_ROW; end
          cg <= cg + 1;
        end
        S_META: if (meta_ack) st <= S_DONE;
        S_DONE: st <= S_IDLE;
        S_FAULT: ;                                   // sticky: the core halts
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
