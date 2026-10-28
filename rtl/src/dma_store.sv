// STORE engine (STORE.2D / STORE.IDX, ISA §5.2): per row, read one row slice per cycle from the source
// tile (wide: four consecutive ACC slots as [M, 4*T_N]), convert (f32 as is, i32 = rint, bf16 = RNE top half),
// pack into a row buffer, then write the row as DMA_BYTES beats with byte enables. STORE.IDX scatters rows to
// idx[i] (-1 skips, IDX_RANGE / IDX_SHORT as the simulator). f4.r16 stores are BAD_QUALIFIER in v0.
module dma_store
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
(
  input  logic              clk,
  input  logic              rst_n,
  input  instr_t            ins,
  input  logic              issue,
  output logic              busy,
  output logic              done,
  output logic [7:0]        desc_idx,
  input  logic [DESC_W-1:0] desc_data,
  // memory write port (always accepted)
  output logic              mem_valid,
  output logic [ADDR_W-1:0] mem_addr,
  output logic [MEM_W-1:0]  mem_wdata,
  output logic [DMA_BYTES-1:0] mem_be,
  // scratchpad client (reads)
  output sp_req_t           sp_req,
  input  logic              sp_grant,
  input  logic              sp_rvalid,
  input  slice_t            sp_rdata,
  input  tmeta_t            meta [N_SLOTS],
  output logic              fault,
  output fault_t            fault_code
);
  typedef enum logic [3:0] {S_IDLE, S_DESC, S_SETUP, S_IDX_RD, S_IDX_WAIT, S_ROW, S_RD, S_RD_WAIT, S_WR, S_DONE, S_FAULT} state_t;
  state_t st;
  instr_t      cur;
  desc_t       d;
  logic        is_idx, wide;
  logic [15:0] src;
  logic [11:0] idx_slot;
  int          rows, cols, row0, col0, nb, n_beats, n_cg, eb;
  int          i, cg, beat, idx_cg;
  logic [63:0] row_addr;
  logic [7:0]  rowbuf [ROWBUF_BYTES];
  logic [31:0] idx_slice [BANKS];
  logic signed [31:0] idx_int [BANKS];
  for (genvar e = 0; e < BANKS; e++) begin : g_idx
    f32_to_int_trunc u_i (.x(idx_slice[e]), .y(idx_int[e]));
  end
  always_comb desc_idx = cur.b[7:0];

  // ---- conversion of the returned slice ----
  logic [31:0] cv_i32 [BANKS];
  logic [15:0] cv_bf  [BANKS];
  for (genvar e = 0; e < BANKS; e++) begin : g_cv
    f32_to_i32_rne u_i (.x(sp_rdata[e]), .y(cv_i32[e]));
    f32_to_bf16    u_b (.x(sp_rdata[e]), .y(cv_bf[e]));
  end

  // the read of slice cg of row i: wide tiles continue into the next slot every T_N columns
  function automatic sp_req_t src_req(input int row, input int c);
    logic [11:0] slot; int cgl;
    slot = tref_slot(src) + 12'(wide ? (c * BANKS) / T_N : 0);
    cgl = wide ? (c % (T_N / BANKS)) : c;
    return slice_req(tref_sp(src), slot, row, cgl, 1'b0, '0, '0);
  endfunction

  always_comb begin
    busy = (st != S_IDLE); done = (st == S_DONE);
    sp_req = '0; mem_valid = 1'b0; mem_addr = '0; mem_wdata = '0; mem_be = '0;
    case (st)
      S_IDX_RD: sp_req = slice_req(SP_VEC, idx_slot, 0, idx_cg, 1'b0, '0, '0);
      S_RD:     sp_req = src_req(i, cg);
      S_WR: begin
        mem_valid = 1'b1; mem_addr = ADDR_W'(row_addr + 64'(beat * DMA_BYTES));
        for (int k = 0; k < DMA_BYTES; k++) begin
          mem_wdata[k*8 +: 8] = rowbuf[beat * DMA_BYTES + k];
          mem_be[k] = (beat * DMA_BYTES + k) < nb;
        end
      end
      default: ;
    endcase
  end

  task automatic set_fault(input fault_t code);
    fault <= 1'b1; fault_code <= code; st <= S_FAULT;
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; fault <= 0; fault_code <= F_NONE; cur <= '0; d <= '0; is_idx <= 0; wide <= 0; src <= '0; idx_slot <= '0;
      rows <= 0; cols <= 0; row0 <= 0; col0 <= 0; nb <= 0; n_beats <= 0; n_cg <= 0; eb <= 0; i <= 0; cg <= 0; beat <= 0;
      idx_cg <= -1; row_addr <= '0;
      for (int k = 0; k < BANKS; k++) idx_slice[k] <= '0;
    end else begin
      case (st)
        S_IDLE: if (issue) begin cur <= ins; fault <= 0; fault_code <= F_NONE; st <= S_DESC; end
        S_DESC: st <= S_SETUP;
        S_SETUP: begin
          desc_t dd; int e2, c, ncols;
          dd = unpack_desc(desc_data); d <= dd;
          src <= cur.a; wide <= cur.flags[FL_WIDE] && (cur.op == OP_STORE_2D);
          e2 = el_bytes(dd.dtype); eb <= e2;
          i <= 0; cg <= 0; idx_cg <= -1;
          ncols = int'(cur.imm[15:0]); cols <= ncols; n_cg <= n_slices(ncols); nb <= ncols * e2;
          n_beats <= (ncols * e2 + DMA_BYTES - 1) / DMA_BYTES;
          if (cur.op == OP_STORE_IDX) begin
            is_idx <= 1; rows <= int'(cur.b[15:8]); row0 <= 0; col0 <= 0; idx_slot <= tref_slot(cur.c);
            if (tref_sp(cur.c) != SP_VEC || int'(tref_slot(cur.c)) < VEC_ROW_SLOT0) set_fault(F_BAD_SHAPE);
            else if (int'(meta[slot_index(SP_VEC, tref_slot(cur.c))].cols) < int'(cur.b[15:8])) set_fault(F_IDX_SHORT);
            else if (!(dd.dtype == DT_F32 || dd.dtype == DT_I32 || dd.dtype == DT_BF16)) set_fault(F_BAD_QUALIFIER);
            else st <= S_ROW;
          end else begin
            is_idx <= 0; rows <= int'(cur.imm[31:16]); row0 <= int'(cur.c); col0 <= int'(cur.d);
            c = cur.flags[FL_WIDE] ? 4 * sp_cols(tref_sp(cur.a), tref_slot(cur.a)) : sp_cols(tref_sp(cur.a), tref_slot(cur.a));
            if (int'(cur.imm[31:16]) > sp_rows(tref_sp(cur.a), tref_slot(cur.a)) || ncols > c || ncols * e2 > ROWBUF_BYTES)
              set_fault(F_BAD_SHAPE);
            else if (!(dd.dtype == DT_F32 || dd.dtype == DT_I32 || dd.dtype == DT_BF16)) set_fault(F_BAD_QUALIFIER);
            else st <= S_ROW;
          end
        end
        S_ROW: begin
          if (i >= rows) st <= S_DONE;
          else if (is_idx && idx_cg != (i / BANKS)) st <= S_IDX_RD;
          else begin
            logic signed [31:0] r;
            r = is_idx ? idx_int[i % BANKS] : 32'sd0;
            if (is_idx && r == -1) i <= i + 1;
            else if (is_idx && (r < 0 || r >= $signed(d.shape0))) set_fault(F_IDX_RANGE);
            else begin
              if (is_idx) row_addr <= d.base + 64'(cur.base_off) + 64'(r) * 64'(d.stride0);
              else row_addr <= d.base + 64'(cur.base_off) + 64'(row0 + i) * 64'(d.stride0) + 64'(col0 * eb);
              cg <= 0; st <= S_RD;
            end
          end
        end
        S_IDX_RD: if (sp_grant) begin idx_cg <= i / BANKS; st <= S_IDX_WAIT; end
        S_IDX_WAIT: if (sp_rvalid) begin
          for (int k = 0; k < BANKS; k++) idx_slice[k] <= sp_rdata[k];
          st <= S_ROW;
        end
        S_RD: if (sp_grant) st <= S_RD_WAIT;
        S_RD_WAIT: if (sp_rvalid) begin
          for (int e = 0; e < BANKS; e++) begin
            int o; o = (cg * BANKS + e) * eb;
            if (cg * BANKS + e < cols) begin
              case (d.dtype)
                DT_F32: for (int k = 0; k < 4; k++) rowbuf[o + k] <= sp_rdata[e][k*8 +: 8];
                DT_I32: for (int k = 0; k < 4; k++) rowbuf[o + k] <= cv_i32[e][k*8 +: 8];
                default: for (int k = 0; k < 2; k++) rowbuf[o + k] <= cv_bf[e][k*8 +: 8];
              endcase
            end
          end
          if (cg + 1 >= n_cg) begin beat <= 0; st <= S_WR; end else begin cg <= cg + 1; st <= S_RD; end
        end
        S_WR: begin
          if (beat + 1 >= n_beats) begin i <= i + 1; st <= S_ROW; end
          beat <= beat + 1;
        end
        S_DONE: st <= S_IDLE;
        S_FAULT: ;
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
