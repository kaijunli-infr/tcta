// STORE engine (STORE.2D / STORE.IDX, ISA §5.2): per row, read one row slice per cycle from the source
// tile (wide: four consecutive ACC slots as [M, 4*T_N]), convert (f32 as is, i32 = rint, bf16 = RNE top half),
// pack into a row buffer, then write the row as DMA_BYTES beats with byte enables. STORE.IDX scatters rows to
// idx[i] (-1 skips, IDX_RANGE / IDX_SHORT as the simulator). f4.r16 stores are BAD_QUALIFIER in v0.
//
// Row buffer (phase 8 K0.1): a converted slice is 64 bytes (f32 / i32) or 32 bytes (bf16) at a slice-aligned
// offset, i.e. whole DMA beats, so the buffer is RB_BANKS beat-wide LUT-RAM banks (beat b in bank b % RB_BANKS)
// written one slice (4 or 2 beats, one per bank) per cycle and read one beat per cycle through a RB_BANKS:1
// mux. The v0 byte-addressed buffer (2048 byte registers, each behind a 64-source write mux) was 760 k LUTs.
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
  localparam int RB_BANKS = 4;                            // beats per f32 slice
  localparam int RB_DEPTH = ROWBUF_BEATS / RB_BANKS;
  localparam int RB_AW = $clog2(RB_DEPTH);
  typedef enum logic [3:0] {S_IDLE, S_DESC, S_SETUP, S_IDX_RD, S_IDX_WAIT, S_ROW, S_RD, S_RD_WAIT, S_WR, S_DONE, S_FAULT} state_t;
  state_t st;
  instr_t      cur;
  desc_t       d;
  logic        is_idx, wide, is_bf;
  logic [15:0] src;
  logic [11:0] idx_slot;
  int          rows, cols, row0, col0, nb, n_beats, n_cg, eb;
  int          i, cg, beat, idx_cg;
  logic [ADDR_W-1:0] row_addr;
  logic [31:0] idx_slice [BANKS];
  // one index converter: only index i of the 16-entry index slice is needed per row
  logic signed [31:0] idx_cur;
  f32_to_int_trunc u_idx (.x(idx_slice[i % BANKS]), .y(idx_cur));
  always_comb desc_idx = cur.b[7:0];

  // ---- conversion of the returned slice ----
  logic [31:0] cv_i32 [BANKS];
  logic [15:0] cv_bf  [BANKS];
  for (genvar e = 0; e < BANKS; e++) begin : g_cv
    f32_to_i32_rne u_i (.x(sp_rdata[e]), .y(cv_i32[e]));
    f32_to_bf16    u_b (.x(sp_rdata[e]), .y(cv_bf[e]));
  end

  // ---- row buffer: RB_BANKS LUT-RAM banks of RB_DEPTH beats ----
  // slice cg of the row: f32 / i32 occupy beats 4cg .. 4cg+3 (bank k, entry cg holds elements 4k .. 4k+3), bf16
  // beats 2cg, 2cg+1 (banks 2(cg%2), +1, entry cg/2, eight elements per beat). Little-endian within the beat.
  logic [MEM_W-1:0] rb_wdata [RB_BANKS];
  logic [RB_BANKS-1:0] rb_we;
  logic [RB_AW-1:0] rb_waddr;
  logic [MEM_W-1:0] rb_rdata [RB_BANKS];
  always_comb begin
    rb_waddr = is_bf ? RB_AW'(cg >> 1) : RB_AW'(cg);
    for (int k = 0; k < RB_BANKS; k++) begin
      rb_we[k] = (st == S_RD_WAIT) && sp_rvalid && (!is_bf || ((k >> 1) == (cg & 1)));
      for (int j = 0; j < 4; j++) rb_wdata[k][j*32 +: 32] = (d.dtype == DT_I32) ? cv_i32[4*k + j] : sp_rdata[4*k + j];
      if (is_bf) for (int j = 0; j < 8; j++) rb_wdata[k][j*16 +: 16] = cv_bf[8*(k & 1) + j];
    end
  end
  for (genvar k = 0; k < RB_BANKS; k++) begin : g_rb
    (* ram_style = "distributed" *) logic [MEM_W-1:0] mem [RB_DEPTH];
    always_ff @(posedge clk) if (rb_we[k]) mem[rb_waddr] <= rb_wdata[k];
    assign rb_rdata[k] = mem[RB_AW'(beat >> $clog2(RB_BANKS))];
  end

  // the read of slice cg of row i: wide tiles continue into the next slot every T_N columns
  function automatic sp_req_t src_req(input int row, input int c);
    logic [11:0] slot; int cgl;
    slot = tref_slot(src) + 12'(wide ? (c * BANKS) / T_N : 0);
    cgl = wide ? (c % (T_N / BANKS)) : c;
    return slice_req(tref_sp(src), slot, row, cgl, 1'b0, '0, '0);
  endfunction

  int rem_bytes;
  always_comb begin
    busy = (st != S_IDLE); done = (st == S_DONE);
    sp_req = '0; mem_valid = 1'b0; mem_addr = '0; mem_wdata = '0; mem_be = '0;
    rem_bytes = nb - beat * DMA_BYTES;
    case (st)
      S_IDX_RD: sp_req = slice_req(SP_VEC, idx_slot, 0, i / BANKS, 1'b0, '0, '0);   // the slice holding index i (v0 used the stale idx_cg)
      S_RD:     sp_req = src_req(i, cg);
      S_WR: begin
        mem_valid = 1'b1; mem_addr = row_addr + ADDR_W'(beat * DMA_BYTES);
        mem_wdata = rb_rdata[beat % RB_BANKS];
        for (int k = 0; k < DMA_BYTES; k++) mem_be[k] = (k < rem_bytes);
      end
      default: ;
    endcase
  end

  task automatic set_fault(input fault_t code);
    fault <= 1'b1; fault_code <= code; st <= S_FAULT;
  endtask

  // row address: one 32 x 32 multiplier shared by the gathered-row (idx * stride0) and the 2-D ((row0 + i) * stride0) cases
  logic signed [31:0] row_mul_a;
  logic signed [63:0] row_mul;
  always_comb begin
    row_mul_a = is_idx ? idx_cur : 32'(row0 + i);
    row_mul = row_mul_a * $signed({1'b0, d.stride0});
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; fault <= 0; fault_code <= F_NONE; cur <= '0; d <= '0; is_idx <= 0; wide <= 0; is_bf <= 0; src <= '0; idx_slot <= '0;
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
          e2 = el_bytes(dd.dtype); eb <= e2; is_bf <= (dd.dtype == DT_BF16);
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
            r = is_idx ? idx_cur : 32'sd0;
            if (is_idx && r == -1) i <= i + 1;
            else if (is_idx && (r < 0 || r >= $signed(d.shape0))) set_fault(F_IDX_RANGE);
            else begin
              row_addr <= ADDR_W'(d.base + 64'(cur.base_off) + 64'(row_mul) + (is_idx ? 64'd0 : 64'(col0 * eb)));
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
        S_RD_WAIT: if (sp_rvalid) begin                   // the slice lands in the row buffer this cycle (rb_we)
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
