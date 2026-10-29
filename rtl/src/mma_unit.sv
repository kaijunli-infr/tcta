// MMA unit (MMA, MMA.T, MMA.MASKED; ISA §5.4, errata E1 / E5): an output-stationary MAC_ROWS x MAC_COLS systolic
// array over block-buffered operands. A product [M, K] x [K, N] is ceil(M/16) * ceil(N/16) output blocks, row-block
// major. Before a block streams, its operands sit in LUT-RAM block buffers (mma_opbuf): the A fill engine reads the
// 16 INP rows of a row block once per row block (K slices), the B fill engine reads the 16 WGT columns of every
// block (K slices; MMA.T: the 16 WGT rows), both one slice per cycle on their own scratchpad client, and record
// each row's / column's maximum exponent on the way. The buffers hold four blocks so the fills run ahead of the
// array: blocks start every P = max(K, 16) cycles when their operands are in and the result FIFO has room.
//
// Phase 8 K0.3 (ACC_F32 = 0, the KV260 configuration): each cell is FIX_B_LEVELS signed 24 x 18 multiply-accumulates
// into 48 bits (one DSP48E2 each) over significands pre-shifted once per row (A: FIX_SA = 15 binades below the row
// maximum, truncated beyond) and once per column (B: FIX_SB = 9 binades per level); the sum is exact for everything
// inside those bands and is rounded to binary32 once, in the writeback (fix_to_f32, mirrored by the simulator's
// acc_mode = "fixed48"). ACC_F32 = 1 keeps the phase-7 float32 cells (mma_pe, the ASIC configurations' format).
//
// Row i of the array streams a[i0+i][k] for k = 0..K-1 starting at cycle T_blk + i, column j streams b[k][j0+j] from
// T_blk + j; cell (i, j) then meets product k at T_blk + i + j + k + 2 and holds its finished sums for one cycle
// at T_blk + i + j + K + 4, when lane collector j picks them (a 16:1 mux over the rows of its column) into the
// result FIFO (16 LUT-RAM lanes, two blocks of rows). The converter stage turns a FIFO row into float32 eight
// lanes per cycle; the writeback then applies the mask (add / mul) and adds the old accumulator (accumulate) on
// eight adders / multipliers in two lane halves each (phase 8 L3; 16 adders before), and writes the row. `wide`
// and `row0` are address arithmetic there.
module mma_unit
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
#(
  parameter bit ACC_F32 = 1'b0
)(
  input  logic     clk,
  input  logic     rst_n,
  input  instr_t   ins,
  input  logic     issue,
  output logic     busy,
  output logic     done,
  // operand fetch clients (always granted) and the writeback client
  output sp_req_t  a_req,
  input  logic     a_grant,
  input  slice_t   a_rdata,
  output sp_req_t  b_req,
  input  logic     b_grant,
  input  slice_t   b_rdata,
  output sp_req_t  wb_req,
  input  logic     wb_grant,
  input  logic     wb_rvalid,
  input  slice_t   wb_rdata,
  // tile metadata
  input  tmeta_t   meta [N_SLOTS],
  output logic     meta_we,
  output logic [7:0] meta_idx,
  output tmeta_t   meta_wdata,
  input  logic     meta_ack,
  output logic     fault,
  output fault_t   fault_code
);
  localparam int R = MAC_ROWS, C = MAC_COLS;
  localparam int RF_DEPTH = 32;                 // result FIFO rows (two blocks)
  localparam int RF_SLOTS = RF_DEPTH / R;
  localparam int HALVES = 4;                    // block buffers per operand
  localparam int HALF_W = $clog2(HALVES);
  localparam int ENTRIES = FIX_K_MAX / BANKS;   // slices per buffered row / column
  localparam int ENT_W = $clog2(ENTRIES);
  localparam int KW = $clog2(FIX_K_MAX);
  localparam int CV_LANES = 8;                  // fixed -> f32 converters: a row in BANKS / CV_LANES cycles. Kept at 8 in
                                                // phase 8 L3: 4 lanes (-2.5 k LUTs) saturate at P = 64 (16 rows x 4 cycles) and
                                                // cost FA's 64x64x64 products +15.6 % (measured 1148 -> 1327)
  localparam int MUL_LANES = 8;                 // mask multipliers: a row in BANKS / MUL_LANES cycles
  localparam int RFW = 3 + FIX_ACC_W;           // a result FIFO lane entry: {flags, acc}

  // ---------------------------------------------------------------- instruction decode
  typedef enum logic [2:0] {S_IDLE, S_SETUP, S_RUN, S_META, S_DONE, S_FAULT} state_t;
  state_t st;
  instr_t cur;
  logic [15:0] acc_t, a_t, b_t, mask_t;
  logic transpose, masked, mask_mul, accumulate, wide;
  int M, K, N, r0, n_ib, n_jb, n_blocks, P, n_cg;

  // ---------------------------------------------------------------- operand buffers
  logic        ab_we, bb_we, bb_scatter;
  logic [HALF_W-1:0] ab_whalf, bb_whalf;
  logic [ENT_W-1:0]  ab_wentry, bb_wentry;
  logic [3:0]  bb_wlane, ab_wrow, bb_wrow;
  logic [15:0] ab_wdata [BANKS], bb_wdata [BANKS];
  logic [HALF_W-1:0] ab_rhalf [R], bb_rhalf [C];
  logic [KW-1:0]     ab_rk [R], bb_rk [C];
  logic [15:0] ab_rdata [R], bb_rdata [C];
  mma_opbuf #(.ROWS(R), .HALVES(HALVES), .ENTRIES(ENTRIES), .SCATTER(1'b0)) u_abuf (
    .clk(clk), .we(ab_we), .w_scatter(1'b0), .w_half(ab_whalf), .w_entry(ab_wentry), .w_lane(4'd0), .w_row(ab_wrow),
    .w_data(ab_wdata), .r_half(ab_rhalf), .r_k(ab_rk), .r_data(ab_rdata));
  mma_opbuf #(.ROWS(C), .HALVES(HALVES), .ENTRIES(ENTRIES)) u_bbuf (
    .clk(clk), .we(bb_we), .w_scatter(bb_scatter), .w_half(bb_whalf), .w_entry(bb_wentry), .w_lane(bb_wlane), .w_row(bb_wrow),
    .w_data(bb_wdata), .r_half(bb_rhalf), .r_k(bb_rk), .r_data(bb_rdata));
  always_comb for (int l = 0; l < BANKS; l++) begin ab_wdata[l] = a_rdata[l][31:16]; bb_wdata[l] = b_rdata[l][31:16]; end

  // the maximum effective exponent over the valid, finite lanes of a slice (1 if none)
  function automatic logic [7:0] lane_emax(input slice_t s, input logic [BANKS-1:0] lanes);
    logic [7:0] m; bf16_f_t f;
    m = 8'd1;
    for (int l = 0; l < BANKS; l++) begin
      f = bf16_fields(s[l][31:16]);
      if (lanes[l] && !f.nan && !f.inf && f.e > m) m = f.e;
    end
    return m;
  endfunction

  // ---------------------------------------------------------------- A fill engine (one row block = R rows x n_cg slices)
  logic af_active, af_start;
  int   af_ib, af_row, af_cg;
  logic [HALF_W-1:0] af_half;
  logic a_ready [HALVES], a_busy [HALVES];
  int   a_rel [HALVES];
  logic [7:0] emax_a [HALVES][R];
  logic afq_v, afq_first, afq_last; logic [HALF_W-1:0] afq_half; logic [3:0] afq_row; logic [ENT_W-1:0] afq_cg; logic [BANKS-1:0] afq_lanes;
  logic af_next_ok;                              // the fill of the next row block may follow without a bubble
  always_comb begin
    af_half = HALF_W'(af_ib % HALVES);
    af_start = (st == S_RUN) && !af_active && (af_ib < n_ib) && !a_ready[af_half] && !a_busy[af_half];
    af_next_ok = (af_ib + 1 < n_ib) && !a_ready[HALF_W'((af_ib + 1) % HALVES)] && !a_busy[HALF_W'((af_ib + 1) % HALVES)];
    a_req = '0;
    if (af_active || af_start) a_req = slice_req(tref_sp(a_t), tref_slot(a_t), af_ib * R + af_row, af_cg, 1'b0, '0, '0);
    ab_we = afq_v; ab_whalf = afq_half; ab_wentry = afq_cg; ab_wrow = afq_row;
  end

  // ---------------------------------------------------------------- B fill engine (one block = K slices)
  logic bf_active, bf_start;
  int   bf_blk, bf_k, bf_row, bf_cg;
  logic [HALF_W-1:0] bf_half;
  int   bf_j0;
  logic b_ready [HALVES], b_busy [HALVES];
  int   b_rel [HALVES];
  logic [7:0] emax_b [HALVES][C];
  logic bfq_v, bfq_first, bfq_last; logic [HALF_W-1:0] bfq_half; logic [3:0] bfq_row; logic [ENT_W-1:0] bfq_entry; logic [3:0] bfq_lane; logic [BANKS-1:0] bfq_lanes;
  logic bf_next_ok;
  always_comb begin
    bf_half = HALF_W'(bf_blk % HALVES);
    bf_j0 = (bf_blk % n_jb) * C;
    bf_start = (st == S_RUN) && !bf_active && (bf_blk < n_blocks) && !b_ready[bf_half] && !b_busy[bf_half];
    bf_next_ok = (bf_blk + 1 < n_blocks) && !b_ready[HALF_W'((bf_blk + 1) % HALVES)] && !b_busy[HALF_W'((bf_blk + 1) % HALVES)];
    b_req = '0;
    if (bf_active || bf_start) begin
      if (transpose) b_req = slice_req(tref_sp(b_t), tref_slot(b_t), bf_j0 + bf_row, bf_cg, 1'b0, '0, '0);
      else           b_req = slice_req(tref_sp(b_t), tref_slot(b_t), bf_k, bf_j0 / BANKS, 1'b0, '0, '0);
    end
    bb_we = bfq_v; bb_scatter = !transpose; bb_whalf = bfq_half; bb_wentry = bfq_entry; bb_wlane = bfq_lane; bb_wrow = bfq_row;
  end

  // ---------------------------------------------------------------- block scheduler
  int   blk, pt, committed;
  logic started_any, start_now;
  int   blk_ib, blk_jb;
  logic [HALF_W-1:0] blk_ahalf, blk_bhalf;
  logic blk_slot;
  always_comb begin
    blk_ib = blk / n_jb; blk_jb = blk % n_jb;
    blk_ahalf = HALF_W'(blk_ib % HALVES); blk_bhalf = HALF_W'(blk % HALVES); blk_slot = blk[0];
    start_now = (st == S_RUN) && (blk < n_blocks) && (!started_any || pt >= P) && (committed <= RF_DEPTH - R)
                && a_ready[blk_ahalf] && b_ready[blk_bhalf];
  end
  // per result-FIFO slot: the block's coordinates and the exponent maxima of its rows / columns
  int   sl_i0 [RF_SLOTS], sl_j0 [RF_SLOTS];
  logic [7:0] sl_ea [RF_SLOTS][R], sl_eb [RF_SLOTS][C];

  // ---------------------------------------------------------------- row / column controllers
  // start tokens travel down the rows (one per cycle) and right along the columns
  logic rs_v [R]; logic [HALF_W-1:0] rs_half [R]; logic rs_slot [R];
  logic cs_v [C]; logic [HALF_W-1:0] cs_half [C];
  logic r_act [R]; int r_k [R]; logic [HALF_W-1:0] r_half [R]; logic r_slot [R];
  logic c_act [C]; int c_k [C]; logic [HALF_W-1:0] c_half [C];
  logic r_act_now [R]; int r_k_now [R]; logic [HALF_W-1:0] r_half_now [R]; logic r_slot_now [R];
  logic c_act_now [C]; int c_k_now [C]; logic [HALF_W-1:0] c_half_now [C];
  always_comb begin
    for (int i = 0; i < R; i++) begin
      r_act_now[i]  = rs_v[i] || r_act[i];
      r_k_now[i]    = rs_v[i] ? 0 : r_k[i];
      r_half_now[i] = rs_v[i] ? rs_half[i] : r_half[i];
      r_slot_now[i] = rs_v[i] ? rs_slot[i] : r_slot[i];
      ab_rhalf[i] = r_half_now[i]; ab_rk[i] = KW'(r_k_now[i]);
    end
    for (int j = 0; j < C; j++) begin
      c_act_now[j]  = cs_v[j] || c_act[j];
      c_k_now[j]    = cs_v[j] ? 0 : c_k[j];
      c_half_now[j] = cs_v[j] ? cs_half[j] : c_half[j];
      bb_rhalf[j] = c_half_now[j]; bb_rk[j] = KW'(c_k_now[j]);
    end
  end

  // ---------------------------------------------------------------- operand preparation (one cycle after the read)
  logic [15:0] a_el_q [R], b_el_q [C];
  logic [7:0]  a_emax_q [R], b_emax_q [C];
  logic a_v_q [R], a_first_q [R], a_last_q [R], a_slot_q [R];
  fix_a_t a_prep [R]; fix_b_t b_prep [C];
  localparam int YW = 8 + FIX_B_LEVELS * FIX_SB;                 // the B significand with room for every level's shift
  if (!ACC_F32) begin : g_prep
    always_comb begin
      for (int i = 0; i < R; i++) begin
        bf16_f_t f; logic [8:0] da; logic [4:0] sh; logic [7:0] m; logic [FIX_SA+7:0] mag;
        f = bf16_fields(a_el_q[i]);
        m = (f.nan || f.inf) ? 8'd0 : f.m;
        da = 9'(a_emax_q[i]) - 9'(f.e);                            // >= 0: the maximum is over this row
        sh = (da > 9'd31) ? 5'd31 : da[4:0];
        mag = {m, {FIX_SA{1'b0}}} >> sh;                            // ma << (FIX_SA - da), truncated below the window
        a_prep[i].v = f.sgn ? FIX_AW'(-$signed({1'b0, mag})) : FIX_AW'($signed({1'b0, mag}));
        a_prep[i].nan = f.nan; a_prep[i].inf = f.inf; a_prep[i].zero = f.zero; a_prep[i].sgn = f.sgn;
      end
      for (int j = 0; j < C; j++) begin
        bf16_f_t f; logic [8:0] db; logic [4:0] sh; logic [7:0] m; logic [YW-1:0] y; logic [FIX_SB+7:0] lv;
        f = bf16_fields(b_el_q[j]);
        m = (f.nan || f.inf) ? 8'd0 : f.m;
        db = 9'(b_emax_q[j]) - 9'(f.e);
        sh = (db > 9'd31) ? 5'd31 : db[4:0];
        y = {m, {(FIX_B_LEVELS * FIX_SB){1'b0}}} >> sh;             // mb << (FIX_B_LEVELS * FIX_SB - db)
        for (int l = 0; l < FIX_B_LEVELS; l++) begin
          lv = y[(FIX_B_LEVELS - 1 - l) * FIX_SB +: FIX_SB + 8];    // level l: mb << ((l + 1) * FIX_SB - db)
          if (db > 9'((l + 1) * FIX_SB) && l != FIX_B_LEVELS - 1) lv = '0;   // a lower level takes it
          if (l > 0 && db <= 9'(l * FIX_SB)) lv = '0;               // a higher level took it
          b_prep[j].v[l] = f.sgn ? FIX_BW'(-$signed({1'b0, lv})) : FIX_BW'($signed({1'b0, lv}));
        end
        b_prep[j].nan = f.nan; b_prep[j].inf = f.inf; b_prep[j].zero = f.zero; b_prep[j].sgn = f.sgn;
      end
    end
  end else begin : g_raw
    always_comb begin
      for (int i = 0; i < R; i++) begin a_prep[i] = '0; a_prep[i].v = FIX_AW'(a_el_q[i]); end
      for (int j = 0; j < C; j++) begin b_prep[j] = '0; b_prep[j].v[0] = FIX_BW'(b_el_q[j]); end
    end
  end

  // array edge registers
  fix_a_t a_edge [R]; fix_b_t b_edge [C];
  logic v_edge [R], first_edge [R], last_edge [R], slot_edge [R];

  // ---------------------------------------------------------------- the array
  fix_a_t a_w [R][C+1]; fix_b_t b_w [R+1][C];
  logic v_w [R][C+1], first_w [R][C+1], last_w [R][C+1], slot_w [R][C+1];
  logic signed [FIX_BITS-1:0] cp [R][C][FIX_B_LEVELS];
  logic [2:0] cflags [R][C];
  logic res_v [R][C], res_slot [R][C];
  for (genvar i = 0; i < R; i++) begin : g_row
    assign a_w[i][0] = a_edge[i]; assign v_w[i][0] = v_edge[i]; assign first_w[i][0] = first_edge[i];
    assign last_w[i][0] = last_edge[i]; assign slot_w[i][0] = slot_edge[i];
    for (genvar j = 0; j < C; j++) begin : g_col
      if (i == 0) begin : g_top
        assign b_w[0][j] = b_edge[j];
      end
      if (!ACC_F32) begin : g_fix
        mma_cell u_cell (
          .clk(clk), .rst_n(rst_n),
          .a_in(a_w[i][j]), .b_in(b_w[i][j]), .v_in(v_w[i][j]), .first_in(first_w[i][j]), .last_in(last_w[i][j]), .slot_in(slot_w[i][j]),
          .a_out(a_w[i][j+1]), .b_out(b_w[i+1][j]), .v_out(v_w[i][j+1]), .first_out(first_w[i][j+1]), .last_out(last_w[i][j+1]), .slot_out(slot_w[i][j+1]),
          .p(cp[i][j]), .flags(cflags[i][j]), .result_valid(res_v[i][j]), .result_slot(res_slot[i][j]));
      end else begin : g_f32
        logic [15:0] a_o, b_o; logic [31:0] res;
        mma_pe u_pe (
          .clk(clk), .rst_n(rst_n),
          .a_in(a_w[i][j].v[15:0]), .b_in(b_w[i][j].v[0][15:0]), .v_in(v_w[i][j]), .first_in(first_w[i][j]), .last_in(last_w[i][j]), .slot_in(slot_w[i][j]),
          .a_out(a_o), .b_out(b_o), .v_out(v_w[i][j+1]), .first_out(first_w[i][j+1]), .last_out(last_w[i][j+1]), .slot_out(slot_w[i][j+1]),
          .result(res), .result_valid(res_v[i][j]), .result_slot(res_slot[i][j]));
        always_comb begin
          a_w[i][j+1] = '0; a_w[i][j+1].v = FIX_AW'(a_o);
          b_w[i+1][j] = '0; b_w[i+1][j].v[0] = FIX_BW'(b_o);
          for (int l = 0; l < FIX_B_LEVELS; l++) cp[i][j][l] = (l == 0) ? FIX_BITS'(res) : '0;
          cflags[i][j] = '0;
        end
      end
    end
  end

  // ---------------------------------------------------------------- lane collectors -> result FIFO lanes
  // cell (i, j) holds its sums for one cycle; in lane j at most one row is finishing per cycle, R cycles apart
  // per block, so a one-hot select over the rows writes {flags, acc} into lane RAM j at entry {slot, i}.
  logic sel_v [C]; logic [3:0] sel_i [C];
  logic signed [FIX_ACC_W-1:0] lane_acc [C];
  logic [RFW-1:0] rf_rd_lane [C];
  int   rf_rd, rf_cnt;
  logic rf_push, cv_pop;
  always_comb begin
    for (int j = 0; j < C; j++) begin
      sel_v[j] = 1'b0; sel_i[j] = 4'd0;
      for (int i = 0; i < R; i++) if (res_v[i][j]) begin sel_v[j] = 1'b1; sel_i[j] = 4'(i); end
      if (ACC_F32) lane_acc[j] = FIX_ACC_W'($signed(cp[sel_i[j]][j][0][31:0]));
      else begin
        lane_acc[j] = '0;
        for (int l = 0; l < FIX_B_LEVELS; l++)
          lane_acc[j] = lane_acc[j] + (FIX_ACC_W'(cp[sel_i[j]][j][l]) <<< ((FIX_B_LEVELS - 1 - l) * FIX_SB));
      end
    end
    rf_push = sel_v[C-1];
  end
  for (genvar j = 0; j < C; j++) begin : g_rf
    (* ram_style = "distributed" *) logic [RFW-1:0] mem [RF_DEPTH];
    always_ff @(posedge clk) if (sel_v[j]) mem[{res_slot[sel_i[j]][j], sel_i[j]}] <= {cflags[sel_i[j]][j], lane_acc[j]};
    assign rf_rd_lane[j] = mem[rf_rd[$clog2(RF_DEPTH)-1:0]];
  end

  // ---------------------------------------------------------------- converter stage (fixed -> f32, CV_LANES per cycle)
  logic cv_busy, cv_v, cv_start, cv_take;
  int   cv_half, cv_half_now;                  // the half being converted: 0 in the start cycle
  slice_t cv_row;
  int   cv_i0, cv_j0, cv_i;
  logic [31:0] cv_y [CV_LANES];
  logic rd_slot; logic [3:0] rd_i;
  always_comb begin rd_slot = rf_rd[$clog2(RF_DEPTH)-1]; rd_i = 4'(rf_rd % R); cv_half_now = cv_busy ? cv_half : 0; end
  for (genvar l = 0; l < CV_LANES; l++) begin : g_cv
    logic [RFW-1:0] e; logic signed [FIX_W_W-1:0] w; int lane;
    always_comb begin
      lane = cv_half_now * CV_LANES + l;
      e = rf_rd_lane[lane];
      w = FIX_W_W'(int'(sl_ea[rd_slot][rd_i]) + int'(sl_eb[rd_slot][lane]) - FIX_W_BIAS);
    end
    if (!ACC_F32) begin : g_fix
      fix_to_f32 u_cv (.acc(e[FIX_ACC_W-1:0]), .w(w), .flags(e[RFW-1 -: 3]), .y(cv_y[l]));
    end else begin : g_f32
      always_comb cv_y[l] = e[31:0];
    end
  end

  // ---------------------------------------------------------------- writeback
  typedef struct packed { logic [3:0] i; logic [9:0] i0; logic [9:0] j0; slice_t data; } rrow_t;
  typedef enum logic [3:0] {W_IDLE, W_MASK, W_MASK_WAIT, W_MCALC, W_MCALC2, W_OLD, W_OLD_WAIT, W_ADD, W_WRITE} wstate_t;
  // Phase 8 L3: ADD_LANES adders and MUL_LANES multipliers work on a row in two lane halves: the mask op in W_MCALC /
  // W_MCALC2, the old-accumulator add in W_ADD (lanes 0 .. 7, registered in s_outq) and W_WRITE (lanes 8 .. 15, straight
  // into the write data). One cycle more per accumulated or mask-added row than the 16-adder version.
  localparam int ADD_LANES = BANKS / 2;
  wstate_t wst;
  rrow_t   wrow;
  slice_t  s_masked, s_out, mask_d, old_d;
  logic [31:0] s_outq [ADD_LANES];
  int      popped;
  logic [31:0] add_a [ADD_LANES], add_b [ADD_LANES], add_y [ADD_LANES], mk_mul [MUL_LANES];
  int      wb_half;
  logic    wb_mask_op;
  for (genvar b = 0; b < ADD_LANES; b++) begin : g_wb
    // one adder bank: the mask add (W_MCALC, W_MCALC2) and the old-accumulator add (W_ADD, W_WRITE) never happen in the same cycle
    f32_add u_add (.a(add_a[b]), .b(add_b[b]), .sub(1'b0), .y(add_y[b]));
  end
  for (genvar b = 0; b < MUL_LANES; b++) begin : g_mul
    f32_mul u_mm (.a(wrow.data[wb_half * MUL_LANES + b]), .b(mask_d[wb_half * MUL_LANES + b]), .y(mk_mul[b]));
  end
  always_comb begin
    wb_half = (wst == W_MCALC2 || wst == W_WRITE) ? 1 : 0;
    wb_mask_op = (wst == W_MCALC) || (wst == W_MCALC2);
    for (int b = 0; b < ADD_LANES; b++) begin
      add_a[b] = wb_mask_op ? wrow.data[wb_half * ADD_LANES + b] : old_d[wb_half * ADD_LANES + b];
      add_b[b] = wb_mask_op ? mask_d[wb_half * ADD_LANES + b] : s_masked[wb_half * ADD_LANES + b];
    end
    for (int b = 0; b < BANKS; b++)
      s_out[b] = !accumulate ? s_masked[b] : (b < ADD_LANES) ? s_outq[b] : add_y[b - ADD_LANES];
  end
  function automatic logic [11:0] wb_slot(input int j0);
    return tref_slot(acc_t) + 12'(wide ? j0 / T_N : 0);
  endfunction
  function automatic int wb_cg(input int j0);
    return wide ? (j0 % T_N) / BANKS : j0 / BANKS;
  endfunction
  int w_row, w_j0;
  always_comb begin
    w_row = r0 + int'(wrow.i0) + int'(wrow.i); w_j0 = int'(wrow.j0);
    wb_req = '0;
    case (wst)
      W_MASK:  wb_req = slice_req(tref_sp(mask_t), tref_slot(mask_t), int'(wrow.i0) + int'(wrow.i), w_j0 / BANKS, 1'b0, '0, '0);
      W_OLD:   wb_req = slice_req(tref_sp(acc_t), wb_slot(w_j0), w_row, wb_cg(w_j0), 1'b0, '0, '0);
      W_WRITE: wb_req = slice_req(tref_sp(acc_t), wb_slot(w_j0), w_row, wb_cg(w_j0), 1'b1, slice_ben(N, w_j0 / BANKS), s_out);
      default: ;
    endcase
    cv_take  = (wst == W_IDLE) && cv_v && (st == S_RUN);
    cv_start = (st == S_RUN) && (rf_cnt > 0) && !cv_busy && (!cv_v || cv_take);
    cv_pop   = cv_busy && (cv_half == BANKS / CV_LANES - 1);       // the row leaves the FIFO with its last lanes
  end

  // ---------------------------------------------------------------- meta writes
  int mk, meta_prev, meta_c;
  always_comb begin
    meta_prev = int'(meta[slot_index(tref_sp(acc_t), tref_slot(acc_t))].rows);
    meta_c = N - mk * T_N; if (meta_c < 0) meta_c = 0; if (meta_c > T_N) meta_c = T_N;
    meta_we = (st == S_META); meta_idx = '0; meta_wdata = '0;
    if (st == S_META) begin
      if (r0 != 0 || !wide) begin
        meta_idx = 8'(slot_index(tref_sp(acc_t), tref_slot(acc_t)));
        meta_wdata.rows = 10'((r0 != 0) ? ((meta_prev > r0 + M) ? meta_prev : r0 + M) : M);
        meta_wdata.cols = 10'(N); meta_wdata.pready = 1'b0;
      end else begin
        meta_idx = 8'(slot_index(tref_sp(acc_t), tref_slot(acc_t) + 12'(mk)));
        meta_wdata.rows = 10'(M); meta_wdata.cols = 10'(meta_c); meta_wdata.pready = 1'b0;
      end
    end
  end

  always_comb begin
    busy = (st != S_IDLE); done = (st == S_DONE);
  end

  // ---------------------------------------------------------------- sequential
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; cur <= '0; fault <= 0; fault_code <= F_NONE;
      acc_t <= '0; a_t <= '0; b_t <= '0; mask_t <= '0; transpose <= 0; masked <= 0; mask_mul <= 0; accumulate <= 0; wide <= 0;
      M <= 0; K <= 0; N <= 0; r0 <= 0; n_ib <= 0; n_jb <= 0; n_blocks <= 0; P <= 16; n_cg <= 0;
      blk <= 0; pt <= 0; committed <= 0; started_any <= 0; mk <= 0; popped <= 0;
      af_active <= 0; af_ib <= 0; af_row <= 0; af_cg <= 0; afq_v <= 0; afq_first <= 0; afq_last <= 0; afq_half <= '0; afq_row <= '0; afq_cg <= '0; afq_lanes <= '0;
      bf_active <= 0; bf_blk <= 0; bf_k <= 0; bf_row <= 0; bf_cg <= 0; bfq_v <= 0; bfq_first <= 0; bfq_last <= 0; bfq_half <= '0; bfq_row <= '0; bfq_entry <= '0; bfq_lane <= '0; bfq_lanes <= '0;
      for (int h = 0; h < HALVES; h++) begin
        a_ready[h] <= 0; a_busy[h] <= 0; a_rel[h] <= 0; b_ready[h] <= 0; b_busy[h] <= 0; b_rel[h] <= 0;
        for (int i = 0; i < R; i++) emax_a[h][i] <= 8'd1;
        for (int j = 0; j < C; j++) emax_b[h][j] <= 8'd1;
      end
      for (int s = 0; s < RF_SLOTS; s++) begin
        sl_i0[s] <= 0; sl_j0[s] <= 0;
        for (int i = 0; i < R; i++) sl_ea[s][i] <= 8'd1;
        for (int j = 0; j < C; j++) sl_eb[s][j] <= 8'd1;
      end
      rf_rd <= 0; rf_cnt <= 0; wst <= W_IDLE; wrow <= '0; s_masked <= '0; mask_d <= '0; old_d <= '0;
      for (int b = 0; b < ADD_LANES; b++) s_outq[b] <= '0;
      cv_busy <= 0; cv_v <= 0; cv_half <= 0; cv_row <= '0; cv_i0 <= 0; cv_j0 <= 0; cv_i <= 0;
      for (int i = 0; i < R; i++) begin
        rs_v[i] <= 0; rs_half[i] <= '0; rs_slot[i] <= 0; r_act[i] <= 0; r_k[i] <= 0; r_half[i] <= '0; r_slot[i] <= 0;
        a_el_q[i] <= '0; a_emax_q[i] <= 8'd1; a_v_q[i] <= 0; a_first_q[i] <= 0; a_last_q[i] <= 0; a_slot_q[i] <= 0;
        a_edge[i] <= '0; v_edge[i] <= 0; first_edge[i] <= 0; last_edge[i] <= 0; slot_edge[i] <= 0;
      end
      for (int j = 0; j < C; j++) begin
        cs_v[j] <= 0; cs_half[j] <= '0; c_act[j] <= 0; c_k[j] <= 0; c_half[j] <= '0;
        b_el_q[j] <= '0; b_emax_q[j] <= 8'd1; b_edge[j] <= '0;
      end
    end else begin
      // ---- instruction FSM ----
      case (st)
        S_IDLE: if (issue) begin cur <= ins; fault <= 0; fault_code <= F_NONE; st <= S_SETUP; end
        S_SETUP: begin
          int m, k, n;
          acc_t <= cur.a; a_t <= cur.b; b_t <= cur.c;
          transpose <= (cur.op == OP_MMA_T);
          masked <= (cur.op == OP_MMA_MASKED);
          mask_t <= cur.d; mask_mul <= (cur.qual == QUAL_MASK_MODE_MUL);
          accumulate <= cur.flags[FL_ACCUMULATE];
          wide <= cur.flags[FL_WIDE] && (cur.op == OP_MMA);
          r0 <= (cur.op == OP_MMA) ? int'(cur.d) : (cur.op == OP_MMA_MASKED ? int'(cur.imm[15:0]) : 0);
          m = int'(meta[slot_index(tref_sp(cur.b), tref_slot(cur.b))].rows);
          k = int'(meta[slot_index(tref_sp(cur.b), tref_slot(cur.b))].cols);
          n = (cur.op == OP_MMA_T) ? int'(meta[slot_index(tref_sp(cur.c), tref_slot(cur.c))].rows)
                                   : int'(meta[slot_index(tref_sp(cur.c), tref_slot(cur.c))].cols);
          M <= m; K <= k; N <= n; n_cg <= n_slices(k);
          n_ib <= (m + R - 1) / R; n_jb <= (n + C - 1) / C; n_blocks <= ((m + R - 1) / R) * ((n + C - 1) / C);
          P <= (k < 16) ? 16 : k;
          blk <= 0; pt <= 0; committed <= 0; started_any <= 0; popped <= 0; mk <= 0;
          af_active <= 0; af_ib <= 0; af_row <= 0; af_cg <= 0; bf_active <= 0; bf_blk <= 0; bf_k <= 0; bf_row <= 0; bf_cg <= 0;
          for (int h = 0; h < HALVES; h++) begin a_ready[h] <= 0; a_busy[h] <= 0; a_rel[h] <= 0; b_ready[h] <= 0; b_busy[h] <= 0; b_rel[h] <= 0; end
          if (k == 0 || k > FIX_K_MAX || tref_sp(cur.a) > SP_STATE || tref_sp(cur.b) > SP_STATE || tref_sp(cur.c) > SP_STATE) begin
            fault <= 1; fault_code <= F_BAD_SHAPE; st <= S_FAULT;
          end else if (cur.op == OP_MMA_MASKED && cur.qual > QUAL_MASK_MODE_MUL) begin
            fault <= 1; fault_code <= F_BAD_QUALIFIER; st <= S_FAULT;
          end else st <= S_RUN;
        end
        S_RUN: begin
          if (popped >= n_blocks * R && wst == W_IDLE && rf_cnt == 0 && !cv_busy && !cv_v && blk >= n_blocks) st <= S_META;
        end
        S_META: if (meta_ack) begin
          if (r0 != 0 || !wide || mk == 3) st <= S_DONE; else mk <= mk + 1;
        end
        S_DONE: st <= S_IDLE;
        S_FAULT: ;
        default: st <= S_IDLE;
      endcase

      // ---- A fill ----
      if (af_start) af_active <= 1;
      afq_v <= af_active || af_start; afq_half <= af_half; afq_row <= 4'(af_row); afq_cg <= ENT_W'(af_cg);
      afq_first <= (af_cg == 0); afq_lanes <= slice_ben(K, af_cg); afq_last <= (af_row == R - 1) && (af_cg + 1 >= n_cg);
      if (af_active || af_start) begin
        if (af_cg + 1 >= n_cg) begin
          af_cg <= 0;
          if (af_row + 1 >= R) begin af_row <= 0; af_active <= af_next_ok; af_ib <= af_ib + 1; end else af_row <= af_row + 1;
        end else af_cg <= af_cg + 1;
      end
      if (afq_v) begin
        logic [7:0] lm;
        lm = lane_emax(a_rdata, afq_lanes);
        emax_a[afq_half][afq_row] <= (afq_first || emax_a[afq_half][afq_row] < lm) ? lm : emax_a[afq_half][afq_row];
        if (afq_last) a_ready[afq_half] <= 1;
      end
      // ---- B fill ----
      if (bf_start) bf_active <= 1;
      bfq_v <= bf_active || bf_start; bfq_half <= bf_half;
      if (transpose) begin
        bfq_row <= 4'(bf_row); bfq_entry <= ENT_W'(bf_cg); bfq_lane <= 4'd0; bfq_first <= (bf_cg == 0);
        bfq_lanes <= slice_ben(K, bf_cg); bfq_last <= (bf_row == C - 1) && (bf_cg + 1 >= n_cg);
      end else begin
        bfq_row <= 4'd0; bfq_entry <= ENT_W'(bf_k / BANKS); bfq_lane <= 4'(bf_k % BANKS); bfq_first <= (bf_k == 0);
        bfq_lanes <= '1; bfq_last <= (bf_k + 1 >= K);
      end
      if (bf_active || bf_start) begin
        if (transpose) begin
          if (bf_cg + 1 >= n_cg) begin
            bf_cg <= 0;
            if (bf_row + 1 >= C) begin bf_row <= 0; bf_active <= bf_next_ok; bf_blk <= bf_blk + 1; end else bf_row <= bf_row + 1;
          end else bf_cg <= bf_cg + 1;
        end else begin
          if (bf_k + 1 >= K) begin bf_k <= 0; bf_active <= bf_next_ok; bf_blk <= bf_blk + 1; end else bf_k <= bf_k + 1;
        end
      end
      if (bfq_v) begin
        if (transpose) begin
          logic [7:0] lm;
          lm = lane_emax(b_rdata, bfq_lanes);
          emax_b[bfq_half][bfq_row] <= (bfq_first || emax_b[bfq_half][bfq_row] < lm) ? lm : emax_b[bfq_half][bfq_row];
        end else begin
          for (int j = 0; j < C; j++) begin
            bf16_f_t f; logic [7:0] e;
            f = bf16_fields(b_rdata[j][31:16]);
            e = (f.nan || f.inf) ? 8'd1 : f.e;
            emax_b[bfq_half][j] <= (bfq_first || emax_b[bfq_half][j] < e) ? e : emax_b[bfq_half][j];
          end
        end
        if (bfq_last) b_ready[bfq_half] <= 1;
      end
      // ---- buffer release: a block's operands stay until its last row / column has streamed them ----
      for (int h = 0; h < HALVES; h++) begin
        if (a_rel[h] > 0) begin a_rel[h] <= a_rel[h] - 1; if (a_rel[h] == 1) a_busy[h] <= 0; end
        if (b_rel[h] > 0) begin b_rel[h] <= b_rel[h] - 1; if (b_rel[h] == 1) b_busy[h] <= 0; end
      end

      // ---- scheduler ----
      if (st == S_RUN) begin
        pt <= pt + 1;
        if (start_now) begin
          blk <= blk + 1; pt <= 1; started_any <= 1;
          sl_i0[blk_slot] <= blk_ib * R; sl_j0[blk_slot] <= blk_jb * C;
          for (int i = 0; i < R; i++) sl_ea[blk_slot][i] <= emax_a[blk_ahalf][i];
          for (int j = 0; j < C; j++) sl_eb[blk_slot][j] <= emax_b[blk_bhalf][j];
          b_ready[blk_bhalf] <= 0; b_busy[blk_bhalf] <= 1; b_rel[blk_bhalf] <= K + C + 3;
          a_busy[blk_ahalf] <= 1; a_rel[blk_ahalf] <= K + R + 3;
          if (blk_jb == n_jb - 1) a_ready[blk_ahalf] <= 0;
        end
      end
      committed <= committed + (start_now ? R : 0) - (cv_pop ? 1 : 0);

      // ---- start tokens ----
      rs_v[0] <= start_now; rs_half[0] <= blk_ahalf; rs_slot[0] <= blk_slot;
      cs_v[0] <= start_now; cs_half[0] <= blk_bhalf;
      for (int i = 1; i < R; i++) begin rs_v[i] <= rs_v[i-1]; rs_half[i] <= rs_half[i-1]; rs_slot[i] <= rs_slot[i-1]; end
      for (int j = 1; j < C; j++) begin cs_v[j] <= cs_v[j-1]; cs_half[j] <= cs_half[j-1]; end

      // ---- controllers and the operand pipeline ----
      for (int i = 0; i < R; i++) begin
        if (r_act_now[i]) begin
          r_k[i] <= r_k_now[i] + 1; r_act[i] <= (r_k_now[i] + 1 < K); r_half[i] <= r_half_now[i]; r_slot[i] <= r_slot_now[i];
        end
        a_el_q[i] <= ab_rdata[i]; a_emax_q[i] <= emax_a[r_half_now[i]][i];
        a_v_q[i] <= r_act_now[i]; a_first_q[i] <= (r_k_now[i] == 0); a_last_q[i] <= (r_k_now[i] == K - 1); a_slot_q[i] <= r_slot_now[i];
        a_edge[i] <= a_prep[i]; v_edge[i] <= a_v_q[i]; first_edge[i] <= a_first_q[i]; last_edge[i] <= a_last_q[i]; slot_edge[i] <= a_slot_q[i];
      end
      for (int j = 0; j < C; j++) begin
        if (c_act_now[j]) begin
          c_k[j] <= c_k_now[j] + 1; c_act[j] <= (c_k_now[j] + 1 < K); c_half[j] <= c_half_now[j];
        end
        b_el_q[j] <= bb_rdata[j]; b_emax_q[j] <= emax_b[c_half_now[j]][j];
        b_edge[j] <= b_prep[j];
      end

      // ---- result FIFO accounting ----
      rf_cnt <= rf_cnt + (rf_push ? 1 : 0) - (cv_pop ? 1 : 0);
      if (cv_pop) rf_rd <= (rf_rd + 1) % RF_DEPTH;

      // ---- converter stage ----
      if (cv_take) cv_v <= 0;
      if (cv_start || cv_busy)
        for (int l = 0; l < CV_LANES; l++) cv_row[cv_half_now * CV_LANES + l] <= cv_y[l];
      if (cv_start) begin                                           // the first lanes convert in the start cycle
        cv_busy <= 1; cv_half <= 1;
        cv_i0 <= sl_i0[rd_slot]; cv_j0 <= sl_j0[rd_slot]; cv_i <= int'(rd_i);
      end
      if (cv_busy) begin
        if (cv_half + 1 >= BANKS / CV_LANES) begin cv_busy <= 0; cv_v <= 1; end else cv_half <= cv_half + 1;
      end

      // ---- writeback FSM ----
      case (wst)
        W_IDLE: if (cv_take) begin
          wrow.i <= 4'(cv_i); wrow.i0 <= 10'(cv_i0); wrow.j0 <= 10'(cv_j0); wrow.data <= cv_row;
          popped <= popped + 1;
          s_masked <= cv_row;
          if (cv_i0 + cv_i >= M) wst <= W_IDLE;                    // padding row of the last row block
          else if (masked) wst <= W_MASK;
          else wst <= accumulate ? W_OLD : W_WRITE;
        end
        W_MASK: if (wb_grant) wst <= W_MASK_WAIT;
        W_MASK_WAIT: if (wb_rvalid) begin mask_d <= wb_rdata; wst <= W_MCALC; end
        W_MCALC: begin                                             // lanes 0 .. 7 of the mask add / mul
          for (int b = 0; b < ADD_LANES; b++) s_masked[b] <= mask_mul ? mk_mul[b] : add_y[b];
          wst <= W_MCALC2;
        end
        W_MCALC2: begin                                            // lanes 8 .. 15
          for (int b = 0; b < ADD_LANES; b++) s_masked[ADD_LANES + b] <= mask_mul ? mk_mul[b] : add_y[b];
          wst <= accumulate ? W_OLD : W_WRITE;
        end
        W_OLD: if (wb_grant) wst <= W_OLD_WAIT;
        W_OLD_WAIT: if (wb_rvalid) begin old_d <= wb_rdata; wst <= W_ADD; end
        W_ADD: begin                                               // lanes 0 .. 7 of old + result; 8 .. 15 add in W_WRITE
          for (int b = 0; b < ADD_LANES; b++) s_outq[b] <= add_y[b];
          wst <= W_WRITE;
        end
        W_WRITE: if (wb_grant) wst <= W_IDLE;
        default: wst <= W_IDLE;
      endcase
    end
  end
endmodule
