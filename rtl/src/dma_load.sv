// LOAD engine (LOAD.2D / LOAD.3D / LOAD.IDX, ISA §5.1-§5.2): one row at a time -- fetch the row's bytes
// (DMA_BYTES per beat, any alignment, fixed-latency memory model), dequantise BANKS elements per cycle
// (dequant_elem), round to BF16 for INP/WGT destinations, write one row slice per cycle into the scratchpad.
// LOAD.IDX reads its indices from a VEC row tile (16 per scratchpad read), skips -1 rows with a zero fill and
// raises IDX_RANGE / IDX_SHORT like the simulator. Rows are processed sequentially in v0 (no row overlap).
//
// Row buffer (phase 8 K0.1): the engine fetches exactly the column window it writes (LOAD.IDX included: the
// bytes of columns col0 .. col0+cols-1, not the whole row), so slice cg of the row starts at beat 4cg (f32 / i32),
// 2cg (bf16), cg (f8) or cg/2 (f4: 8 bytes), and the buffer is RB_BANKS beat-wide LUT-RAM banks (beat b in bank
// b % RB_BANKS) written one beat per cycle as the responses arrive and read as a 4-beat window per slice (each
// bank at its own address, the outputs rotated into beat order). Block scales (*.b32 from the scale tensor,
// f4.r16 from the row's scale bytes after its nibbles) are a second 2-beat fetch into `scalebuf`. The v0 buffer
// (128 beat registers read as a 5-beat window at any beat offset) was 28 k LUTs.
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
  localparam int RB_BANKS = 4;                            // beats per f32 slice
  localparam int RB_DEPTH = ROWBUF_BEATS / RB_BANKS;
  localparam int RB_AW = $clog2(RB_DEPTH);
  localparam int RB_SH = $clog2(RB_BANKS);
  typedef enum logic [3:0] {S_IDLE, S_DESC, S_SETUP, S_IDX_RD, S_IDX_WAIT, S_ROW, S_FETCH, S_WAIT, S_DEQ, S_ZFILL,
                            S_META, S_DONE, S_FAULT} state_t;
  state_t st;

  instr_t      cur;
  desc_t       d;
  logic        is_idx;
  logic [15:0] dst;
  logic [11:0] idx_slot;
  int          rows, cols, row0, col0, n_el_row, nb, n_beats, n_cg;
  int          i, cg, beat_issue, beat_got, n_resp;
  logic        need_scales;        // *.b32 / f4.r16: a second fetch of this row's scale codes
  logic        is_r16, is_f4;
  logic [ADDR_W-1:0] row_addr, scale_addr;
  logic [MEM_W-1:0] scalebuf [2];
  logic [31:0] idx_slice [BANKS];
  int          idx_cg;
  logic signed [31:0] ri;
  logic        to_bf16;

  // ---- descriptor and geometry helpers ----
  always_comb desc_idx = cur.b[7:0];
  // bytes of the first `col` columns of a row (f4 formats: nibbles)
  function automatic int col_bytes(input logic [7:0] dtype, input int col);
    return (dtype == DT_F4_B32 || dtype == DT_F4_R16) ? (col >> 1) : col * el_bytes(dtype);
  endfunction

  // ---- row buffer: RB_BANKS LUT-RAM banks of RB_DEPTH beats; beat b lives in bank b % RB_BANKS ----
  logic [MEM_W-1:0] rb_rdata [RB_BANKS];
  logic [RB_AW-1:0] rb_raddr [RB_BANKS];
  int   w0;                                  // first beat of slice cg's payload
  logic rb_we;
  always_comb begin
    case (d.dtype)
      DT_F32, DT_I32:     w0 = cg * 4;
      DT_BF16:            w0 = cg * 2;
      DT_F8_T, DT_F8_B32: w0 = cg;
      default:            w0 = cg >> 1;
    endcase
    // bank k holds beat w0 + ((k - w0) mod RB_BANKS) of the window
    for (int k = 0; k < RB_BANKS; k++) rb_raddr[k] = RB_AW'((w0 + ((k - w0) & (RB_BANKS - 1))) >> RB_SH);
    rb_we = (st == S_FETCH || st == S_WAIT) && mem_rvalid && (beat_got < n_beats);
  end
  for (genvar k = 0; k < RB_BANKS; k++) begin : g_rb
    (* ram_style = "distributed" *) logic [MEM_W-1:0] mem [RB_DEPTH];
    always_ff @(posedge clk) if (rb_we && (beat_got & (RB_BANKS - 1)) == k) mem[RB_AW'(beat_got >> RB_SH)] <= mem_rdata;
    assign rb_rdata[k] = mem[rb_raddr[k]];
  end

  // ---- element extraction: the 4-beat window in beat order, then the dtype's element spacing ----
  logic [RB_BANKS*MEM_W-1:0] win;
  logic [31:0] raw   [BANKS];
  logic [7:0]  sc8   [BANKS];
  logic [31:0] deq   [BANKS];
  logic [15:0] deq_bf [BANKS];
  slice_t      wslice;
  int          nib0;                         // f4: nibble of element 0 of the slice within the window
  always_comb begin
    for (int j = 0; j < RB_BANKS; j++) win[j*MEM_W +: MEM_W] = rb_rdata[(w0 + j) & (RB_BANKS - 1)];
    nib0 = (cg & 1) * BANKS + (col0 & 1);
    for (int e = 0; e < BANKS; e++) begin
      int sc_i;
      case (d.dtype)
        DT_F32, DT_I32:     raw[e] = win[e*32 +: 32];
        DT_BF16:            raw[e] = 32'(win[e*16 +: 16]);
        DT_F8_T, DT_F8_B32: raw[e] = 32'(win[e*8 +: 8]);
        default:            raw[e] = 32'(win[(nib0 + e)*4 +: 4]);
      endcase
      case (d.dtype)
        DT_F8_B32, DT_F4_B32: sc_i = (cg * BANKS + e) >> 5;                 // scale codes start at col0 (col0 % 32 == 0)
        DT_F4_R16:            sc_i = ((col0 & 15) + cg * BANKS + e) >> 4;   // scale bytes fetched from column col0 & ~15
        default:              sc_i = 0;
      endcase
      sc8[e] = scalebuf[sc_i >> 4][(sc_i & 15)*8 +: 8];
    end
  end
  for (genvar e = 0; e < BANKS; e++) begin : g_deq
    dequant_elem u_dq (.dtype(d.dtype), .raw(raw[e]), .scale8(sc8[e]), .scale_t(d.scale_t), .y(deq[e]));
    f32_to_bf16  u_bf (.x(deq[e]), .y(deq_bf[e]));
    always_comb wslice[e] = to_bf16 ? {deq_bf[e], 16'd0} : deq[e];
  end

  // ---- index conversion ----
  // one index converter: only index i of the 16-entry index slice is needed per row
  logic signed [31:0] idx_cur;
  f32_to_int_trunc u_idx (.x(idx_slice[i % BANKS]), .y(idx_cur));

  // ---- row address: one 32 x 32 multiplier for the gathered-row (idx * stride0) and 2-D ((row0 + i) * stride0) cases ----
  logic signed [31:0] row_mul_a;
  logic signed [63:0] row_mul;
  logic [ADDR_W-1:0]  row_base;                // this row's first byte (column 0)
  always_comb begin
    row_mul_a = is_idx ? idx_cur : 32'(row0 + i);
    row_mul = row_mul_a * $signed({1'b0, d.stride0});
    row_base = ADDR_W'(d.base + 64'(cur.base_off) + 64'(row_mul));
  end

  // ---- outputs ----
  always_comb begin
    busy = (st != S_IDLE);
    done = (st == S_DONE);
    mem_valid = 1'b0; mem_addr = '0;
    sp_req = '0; meta_we = 1'b0; meta_idx = '0; meta_wdata = '0;
    case (st)
      S_IDX_RD: sp_req = slice_req(SP_VEC, idx_slot, 0, i / BANKS, 1'b0, '0, '0);   // the slice holding index i (v0 used the stale idx_cg)
      S_FETCH: begin
        mem_valid = 1'b1;
        if (beat_issue < n_beats) mem_addr = row_addr + ADDR_W'(beat_issue * DMA_BYTES);
        else mem_addr = scale_addr + ADDR_W'((beat_issue - n_beats) * DMA_BYTES);
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
      rows <= 0; cols <= 0; row0 <= 0; col0 <= 0; n_el_row <= 0; nb <= 0; n_beats <= 0; n_cg <= 0;
      i <= 0; cg <= 0; beat_issue <= 0; beat_got <= 0; n_resp <= 0; need_scales <= 0; is_r16 <= 0; is_f4 <= 0;
      row_addr <= '0; scale_addr <= '0; idx_cg <= 0; ri <= 0; to_bf16 <= 0;
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
            desc_t dd; int rb, ncols, dbytes; logic [15:0] dstr;
            dd = unpack_desc(desc_data);
            dstr = cur.a;
            dst <= dstr;
            to_bf16 <= (tref_sp(dstr) == SP_INP || tref_sp(dstr) == SP_WGT);
            is_r16 <= (dd.dtype == DT_F4_R16);
            is_f4 <= (dd.dtype == DT_F4_B32 || dd.dtype == DT_F4_R16);
            need_scales <= (dd.dtype == DT_F8_B32 || dd.dtype == DT_F4_B32 || dd.dtype == DT_F4_R16);
            i <= 0; cg <= 0; idx_cg <= -1;
            ncols = int'(cur.imm[15:0]); cols <= ncols; n_cg <= n_slices(ncols); col0 <= int'(cur.d);
            // f4: an odd col0 puts element 0 in the high nibble of the first byte, so the window is one byte longer
            dbytes = (dd.dtype == DT_F4_B32 || dd.dtype == DT_F4_R16) ? ((ncols + int'(cur.d[0]) + 1) >> 1) : ncols * el_bytes(dd.dtype);
            nb <= dbytes; n_beats <= (dbytes + DMA_BYTES - 1) / DMA_BYTES;
            if (cur.op == OP_LOAD_IDX) begin
              is_idx <= 1'b1;
              rows <= int'(cur.b[15:8]); row0 <= 0; n_el_row <= int'(dd.shape1); idx_slot <= tref_slot(cur.c);
              rb = int'(cur.imm[31:16]);
              if (tref_sp(dstr) == SP_STATE || tref_sp(dstr) > SP_STATE) set_fault(F_BAD_SHAPE);
              else if (rb % 64 != 0 || rb > ROWBUF_BYTES) set_fault(F_BAD_SHAPE);
              else if (tref_sp(cur.c) != SP_VEC || int'(tref_slot(cur.c)) < VEC_ROW_SLOT0) set_fault(F_BAD_SHAPE);
              else if (int'(meta[slot_index(SP_VEC, tref_slot(cur.c))].cols) < int'(cur.b[15:8])) set_fault(F_IDX_SHORT);
              else if (dd.dtype == DT_F8_B32 || dd.dtype == DT_F4_B32) set_fault(F_BAD_QUALIFIER);
              else if (ncols > sp_cols(tref_sp(dstr), tref_slot(dstr))) set_fault(F_BAD_SHAPE);
              else st <= S_ROW;
            end else begin
              is_idx <= 1'b0;
              rows <= int'(cur.imm[31:16]); row0 <= int'(cur.c); n_el_row <= ncols;
              rb = row_bytes_for(dd.dtype, ncols);
              if (int'(cur.imm[31:16]) > sp_rows(tref_sp(dstr), tref_slot(dstr)) ||
                  ncols > sp_cols(tref_sp(dstr), tref_slot(dstr)) || rb > ROWBUF_BYTES) set_fault(F_BAD_SHAPE);
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
            r = is_idx ? idx_cur : 32'sd0;
            ri <= r;
            if (is_idx && r == -1) begin cg <= 0; st <= S_ZFILL; end
            else if (is_idx && (r < 0 || r >= $signed(d.shape0))) set_fault(F_IDX_RANGE);
            else begin
              row_addr <= row_base + ADDR_W'(col_bytes(d.dtype, col0));
              if (is_r16) scale_addr <= row_base + ADDR_W'((n_el_row >> 1) + (col0 >> 4));
              else scale_addr <= ADDR_W'(d.scales_base + 64'(cur.scales_off) + 64'((row0 + i) / 32) * 64'(d.scales_stride) + 64'(col0 / 32));
              beat_issue <= 0; beat_got <= 0; n_resp <= n_beats + (need_scales ? 2 : 0);
              st <= S_FETCH;
            end
          end
        end
        S_IDX_RD: if (sp_grant) begin idx_cg <= i / BANKS; st <= S_IDX_WAIT; end
        S_IDX_WAIT: if (sp_rvalid) begin
          for (int k = 0; k < BANKS; k++) idx_slice[k] <= sp_rdata[k];
          st <= S_ROW;
        end
        S_FETCH: begin
          if (beat_issue + 1 >= n_resp) st <= S_WAIT;
          beat_issue <= beat_issue + 1;
          if (mem_rvalid) begin                        // data beats land in the row buffer banks (rb_we)
            if (mem_rerr) set_fault(F_PAGE_FAULT);
            if (beat_got >= n_beats) scalebuf[beat_got - n_beats] <= mem_rdata;
            beat_got <= beat_got + 1;
          end
        end
        S_WAIT: begin
          if (mem_rvalid) begin
            if (mem_rerr) set_fault(F_PAGE_FAULT);
            if (beat_got >= n_beats) scalebuf[beat_got - n_beats] <= mem_rdata;
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
