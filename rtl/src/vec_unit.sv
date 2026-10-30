// VEC unit (ISA §5.3): one in-order pipe of VEC_LANES lanes over row slices. A micro-sequencer issues one
// scratchpad read per cycle (the row vector operand once per row, then the slice operands in a fixed order);
// the response stage computes when the last operand of a slice returns and pushes the result into a small
// write FIFO drained by the write client. SFU ops hold a slice and pass it through SFU_LANES lanes over
// L / SFU_LANES cycles (one slice in flight). Row reductions keep L partials (lane l over columns l, l+L, ...)
// and add them as a binary tree (erratum E1). VTOPK is k selection passes over the row (stable: lowest index on
// ties), VCOUNT a counting pass that writes a front-end register.
//
// Phase 8 K0.2 (fit): one ALU per lane -- one f32 adder, one f32 multiplier and one BF16 rounder, operands
// muxed by the instruction kind; max is the adder's subtract path (sign of x - v) with f32_max's NaN / ±0 rules.
// The lanes are free during the SFU's step cycles, so the sigmoid chain's multiply (x * -log2 e) and add
// (1 + exp2) and the VGATE decay multiply run on lanes 0 .. SFU_LANES-1. The row-reduction tree is a four-cycle
// reducer on RED_W = L/2 adders (level n pairs adjacent partials, the E1 order); its row write is held until
// the write FIFO takes it, and a row's last slice is issued only when the reducer is free by the time the slice
// returns (never stalls for n_cg >= 4; K0.2's rule blocked while busy and cost 3 cycles per row at n_cg = 4). v0 had 95 adders, 36 multipliers and 47 comparators in this unit (122 k LUTs).
//
// Phase 8 K2 (erratum E2, VEC 2-D slots banked by row): the row-vector operand (a [M, 1] VEC column) is read as a
// column slice once per BANKS rows (rowvec_q, lane = row % BANKS) instead of once per row, and an elementwise kind
// whose tile operands are all 2-D VEC tiles runs in column-slice order (cs_mode: the sequencer's "row" is the row
// group, its "slice" the column; a [64, 1] op is 4 slices, not 64). Row reductions write their result as one element
// of a column slice. Mixed-shape kinds (an ACC tile with a 2-D VEC tile: VGATE, VSEL, VMASK tile) keep row order and
// pay the pad's element-serial access on the VEC side.
module vec_unit
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
(
  input  logic     clk,
  input  logic     rst_n,
  input  instr_t   ins,
  input  logic     issue,
  output logic     busy,
  output logic     done,
  // scratchpad clients
  output sp_req_t  rd_req,
  input  logic     rd_grant,
  input  logic     rd_rvalid,
  input  slice_t   rd_rdata,
  output sp_req_t  wr_req,
  input  logic     wr_grant,
  // tile metadata
  input  tmeta_t   meta [N_SLOTS],
  output logic     meta_we,
  output logic [7:0] meta_idx,
  output tmeta_t   meta_wdata,
  input  logic     meta_ack,
  // front-end registers (VTOPK base, VCOUNT result)
  input  logic [31:0] regs [N_REGS],
  output logic     reg_we,
  output logic [3:0] reg_idx,
  output logic [31:0] reg_wdata,
  output logic     fault,
  output fault_t   fault_code
);
  localparam int L = VEC_LANES;
  localparam int SFU_STEPS = L / SFU_LANES;
  localparam int RED_W = L / 2;
  if (2 * SFU_LANES > L) begin : g_check
    $error("vec_unit: the SFU chain needs 2 * SFU_LANES <= VEC_LANES lanes");
  end
  localparam int RED_LEVELS = $clog2(L);
  localparam logic [31:0] NEG_INF = 32'hff800000;
  localparam logic [31:0] POS_ZERO = 32'h00000000;
  localparam logic [31:0] NEG_LOG2E = 32'hbfb8aa3b;   // -1.4426950408889634 as binary32
  localparam logic [31:0] ONE = 32'h3f800000;
  localparam int WQ = 4;

  // ================================================================ declarations
  typedef enum logic [4:0] {
    K_SUB, K_MUL, K_ADD, K_MAX, K_NEG, K_SCALE, K_GATE_MUL, K_GATE_ADD, K_GATE_DECAY, K_CVT_BF16, K_CVT_RAW, K_COPY,
    K_MASK_CAUSAL, K_MASK_BAND, K_MASK_TILE, K_EXP2, K_EXP2_SUM, K_RCP, K_LOG2, K_SIGMOID, K_FILL, K_ROWMAX, K_ROWSUM,
    K_SEL, K_TOPK, K_COUNT
  } kind_t;
  typedef enum logic [3:0] {S_IDLE, S_SETUP, S_RUN, S_META, S_META2, S_REG, S_DONE, S_FAULT} state_t;
  typedef struct packed { logic vec; logic [1:0] op; logic last; logic [9:0] row; logic [5:0] cg; logic [2:0] slot; logic [6:0] pass; } tag_t;
  typedef struct packed { logic cs; sp_t sp; logic [11:0] slot; logic [9:0] row; logic [5:0] cg; logic [L-1:0] ben; slice_t data; } wreq_t;

  state_t st;
  instr_t cur;
  kind_t  kind;
  logic   is_sfu, is_red, use_vec, use_imm, has_val, has_base, inplace, row_vec, write_slice;
  logic   cs_mode, vec_cs, dst_2d;              // E2: column-slice order; the vector operand is a 2-D VEC column; the dst is one
  int     n_rg;                                 // row groups of M rows
  logic [15:0] t_src, t_dst, t_vec, t_aux, t_val;
  int     M, N, n_cg, r0, kk, n_pass, width_b, n_ops, n_slots;
  logic signed [15:0] q_off, k_off;
  logic [31:0] imm, bound_f;

  // sequencer
  int   s_slot, s_row, s_cg, s_op, s_pass, Mk, Nk, n_cgk, total_rd, got_rd;
  logic s_vec_pending, s_done, can_issue;
  tag_t tagq [2]; logic [1:0] tag_cnt; logic tag_wr, tag_rd; tag_t tag_in;
  logic [15:0] rd_tile; int rd_row, rd_cg;

  // write FIFO
  (* ram_style = "distributed" *) wreq_t wq [WQ];
  int wq_wr, wq_rd, wq_cnt; logic wq_push, wq_pop; wreq_t wq_in;
  logic  wo_push; wreq_t wo_in; logic [L-1:0] mask_ben;

  // response stage
  tag_t   rtag;
  logic   rsp_last;
  slice_t op0, op1, partial, partial_base, red_in, sfu_slice, acc_slice, exp_part_in, result, exp2_arg, x, v_bcast, g;
  slice_t rowvec_q, vrow;                      // the vector operand's column slice; the per-lane vector value of the returned slice
  logic [L-1:0] result_ben; logic result_push;
  int     sfu_step; logic sfu_busy, sfu_mode; tag_t sfu_tag; logic exp_sum_pending;
  logic   red_fire; logic [9:0] red_row;
  wreq_t  rsp_w, row_w, sfu_w, topk_w; logic rsp_w_push, row_w_push, sfu_w_push, topk_push;
  // row reducer
  logic   red_busy, red_done; int red_lvl; logic [31:0] red_reg [L]; logic [9:0] red_row_q, red_row_d; logic [31:0] red_result;
  logic   red_free_at_fire;                   // the reducer takes a fire two cycles after the slice's issue
  logic [31:0] red_a [RED_W], red_b [RED_W], red_sum [RED_W], red_max [RED_W];
  // VTOPK / VCOUNT
  logic [T_VR-1:0] selected;
  logic [31:0] best_v, best_as_f32; int best_i; logic best_valid;
  int     topk_pass; logic topk_val_phase, topk_fire;
  int     count_acc;
  logic signed [31:0] best_plus_base;

  // lane ALUs: one adder (sub selectable), one multiplier, one BF16 rounder per lane
  slice_t add_a, add_b, mul_a, mul_b;
  logic   add_sub;
  logic [31:0] l_sum [L], l_prod [L], l_prod_all [L], l_max [L];
  // the decay group (lanes SFU_LANES .. 2*SFU_LANES-1) has its own operand / product names: its product depends on
  // the SFU's exp2 output, which depends on lanes 0 .. SFU_LANES-1's product, so sharing one array would read as a
  // combinational loop
  logic [31:0] dec_a [SFU_LANES], dec_b [SFU_LANES], l_prod_dec [SFU_LANES];
  logic [15:0] l_bf [L];
  logic [31:0] sfu_in [SFU_LANES], sfu_out [SFU_LANES], sfu_y [SFU_LANES], sfu_u [SFU_LANES], sfu_acc_in [SFU_LANES];
  logic [1:0]  sfu_op;

  // ================================================================ helpers
  // sign-magnitude -> signed order; NaN below everything (argsort puts NaN last); +0 == -0
  function automatic logic signed [32:0] ord(input logic [31:0] f);
    if (f[30:23] == 8'hff && f[22:0] != 0) return -33'sd4294967296;
    if (f[30:0] == 0) return 33'sd0;
    return f[31] ? -$signed({2'b0, f[30:0]}) : $signed({2'b0, f[30:0]});
  endfunction
  function automatic logic f_nonzero(input logic [31:0] f); return f[30:0] != 0; endfunction
  function automatic logic f_isnan(input logic [31:0] f); return f[30:23] == 8'hff && f[22:0] != 0; endfunction
  // f32_max (np.maximum) from the adder's x - v: NaN propagates; x > v <=> x - v > 0 (exact for finite floats,
  // the difference never rounds to zero); equal: -0 vs +0 gives +0, otherwise x
  function automatic logic [31:0] max_from_diff(input logic [31:0] xv, input logic [31:0] vv, input logic [31:0] diff);
    logic eq;
    eq = (xv[30:0] == vv[30:0]) && ((xv[31] == vv[31]) || (xv[30:0] == 0));
    if (f_isnan(xv) || f_isnan(vv)) return 32'h7fc00000;
    if (eq) return (xv[31] & vv[31]) ? xv : {1'b0, xv[30:0]};
    return diff[31] ? vv : xv;
  endfunction

  // ================================================================ lane arithmetic instances
  for (genvar b = 0; b < L; b++) begin : g_lane
    f32_add u_add (.a(add_a[b]), .b(add_b[b]), .sub(add_sub), .y(l_sum[b]));
    if (b >= SFU_LANES && b < 2 * SFU_LANES) begin : g_dec
      f32_mul u_mul (.a(dec_a[b - SFU_LANES]), .b(dec_b[b - SFU_LANES]), .y(l_prod_dec[b - SFU_LANES]));
      assign l_prod_all[b] = l_prod_dec[b - SFU_LANES];
      assign l_prod[b] = '0;
    end else begin : g_mul
      f32_mul u_mul (.a(mul_a[b]), .b(mul_b[b]), .y(l_prod[b]));
      assign l_prod_all[b] = l_prod[b];
    end
    f32_to_bf16 u_bf (.x(x[b]), .y(l_bf[b]));
    always_comb l_max[b] = max_from_diff(add_a[b], add_b[b], l_sum[b]);
  end
  for (genvar b = 0; b < SFU_LANES; b++) begin : g_sfu
    sfu_lane u_sfu (.op(sfu_op), .x(sfu_in[b]), .t(l_prod[b]), .w(l_sum[b]), .u(sfu_u[b]), .y(sfu_y[b]));
  end
  for (genvar n = 0; n < RED_W; n++) begin : g_red
    f32_add u_ra (.a(red_a[n]), .b(red_b[n]), .sub(kind == K_ROWMAX), .y(red_sum[n]));
    always_comb red_max[n] = max_from_diff(red_a[n], red_b[n], red_sum[n]);
  end
  always_comb best_plus_base = best_i + (has_base ? $signed(regs[cur.d[3:0]]) : 32'sd0);
  i32_to_f32 u_bi (.x(best_plus_base), .y(best_as_f32));
  i32_to_f32 u_bound (.x($signed(cur.imm)), .y(bound_f));

  // ================================================================ issue side
  always_comb begin
    // pacing: the tag queue, room in the write FIFO (one slot reserved for a reduction result waiting to be pushed); SFU kinds
    // keep one slice in flight; VTOPK waits at a pass boundary until the previous pass has been written (its
    // column must be marked selected first); a reduction's last slice waits for the reducer
    if (is_sfu) can_issue = (tag_cnt == 0) && !sfu_busy && !exp_sum_pending && (wq_cnt + (red_done ? 1 : 0) <= WQ - 2);
    else can_issue = (tag_cnt < 2) && (wq_cnt + int'(tag_cnt) + (red_done ? 1 : 0) <= WQ - 2);
    if (kind == K_TOPK && (topk_fire || (s_cg == 0 && tag_cnt != 0))) can_issue = 1'b0;
    if (is_red) begin
      logic last_inflight;                        // a row's last slice is already in flight (its return would fire the reducer)
      last_inflight = (tag_cnt >= 2'd1 && !tagq[0].vec && int'(tagq[0].cg) + 1 >= n_cg) ||
                      (tag_cnt == 2'd2 && !tagq[1].vec && int'(tagq[1].cg) + 1 >= n_cg);
      // the last slice issued now returns next cycle and fires level 0 the cycle after: the reducer must be
      // free then (level 0 of another row this cycle or next cycle, or more than two levels left, block it),
      // and a held result (write FIFO full) with another row in the pipe blocks too, so a done never overwrites
      // an unpushed one. Rows of n_cg >= 4 slices therefore stream without a stall (K0.2 blocked while busy).
      red_free_at_fire = !red_fire && !last_inflight && (!red_busy || red_lvl >= RED_LEVELS - 2) && !(red_done && red_busy);
      if (red_fire && s_cg == 0) can_issue = 1'b0;
      if (!red_free_at_fire && s_cg + 1 >= n_cg && !s_vec_pending) can_issue = 1'b0;
    end
    // which tile the next read touches
    rd_tile = t_src; rd_row = s_row; rd_cg = s_cg;
    if (s_vec_pending) begin rd_tile = t_vec; rd_cg = 0; end
    else case (kind)
      K_GATE_MUL, K_GATE_ADD, K_GATE_DECAY: rd_tile = (s_op == 0) ? t_aux : t_src;
      K_MASK_TILE: rd_tile = t_aux;
      K_SEL: rd_tile = (s_op == 0) ? t_src : (s_op == 1) ? t_aux : t_vec;     // a, b, mask
      default: ;
    endcase
    rd_req = '0;
    if (st == S_RUN && !s_done && (n_ops > 0 || s_vec_pending) && can_issue) begin
      logic [11:0] sl;
      sl = tref_slot(rd_tile) + 12'((kind == K_SCALE && !s_vec_pending) ? s_slot : 0);
      if (s_vec_pending && vec_cs) rd_req = col_req(sl, 0, cs_mode ? s_row : s_row / BANKS, 1'b0, '0, '0);     // the vector's rows of this group
      else if (cs_mode)            rd_req = col_req(sl, s_cg, s_row, 1'b0, slice_ben(M, s_row), '0);           // column s_cg, row group s_row
      else rd_req = slice_req(tref_sp(rd_tile), sl, rd_row, rd_cg, 1'b0, s_vec_pending ? L'(1) : slice_ben(Nk, rd_cg), '0);
    end
    // tag of the read issued this cycle
    tag_in = '0;
    tag_in.vec = s_vec_pending; tag_in.op = 2'(s_op); tag_in.last = !s_vec_pending && (s_op + 1 >= n_ops);
    tag_in.row = 10'(s_row); tag_in.cg = 6'(s_cg); tag_in.slot = 3'(s_slot); tag_in.pass = 7'(s_pass);
    // write-only slices (VFILL, VMASK causal / band) go straight into the write FIFO
    for (int b = 0; b < L; b++) begin
      int qi, kj; qi = s_row + int'(q_off); kj = s_cg * L + b + int'(k_off);
      case (kind)
        K_MASK_CAUSAL: mask_ben[b] = (kj > qi);
        K_MASK_BAND:   mask_ben[b] = (kj > qi) || (kj <= qi - width_b);
        default:       mask_ben[b] = 1'b1;
      endcase
    end
    wo_push = (st == S_RUN) && !s_done && (n_ops == 0) && (wq_cnt < WQ) && !topk_push &&
              (kind == K_FILL || kind == K_MASK_CAUSAL || kind == K_MASK_BAND);
    wo_in = '0;
    wo_in.cs = cs_mode; wo_in.sp = tref_sp(t_dst); wo_in.slot = tref_slot(t_dst); wo_in.row = 10'(s_row); wo_in.cg = 6'(s_cg);
    wo_in.ben = (cs_mode ? slice_ben(M, s_row) : slice_ben(N, s_cg)) & mask_ben;
    for (int b = 0; b < L; b++) wo_in.data[b] = (kind == K_FILL) ? imm : NEG_INF;
  end

  always_comb begin
    tag_wr = rd_req.valid && rd_grant;
    tag_rd = rd_rvalid;
  end
  // SFU lane inputs (own block: the lane outputs feed the response block below)
  always_comb begin
    sfu_mode = is_sfu && sfu_busy;             // SFU step cycles: no read returns, lanes 0..SFU_LANES-1 serve the SFU
    for (int b = 0; b < SFU_LANES; b++) begin
      sfu_in[b] = sfu_slice[sfu_step * SFU_LANES + b];
      sfu_acc_in[b] = acc_slice[sfu_step * SFU_LANES + b];
    end
  end
  // ================================================================ response side (combinational)
  // operands first (these blocks read registers, the returned slice and the SFU's exp2 output, never a lane
  // output, so the ALU -> SFU -> ALU chain of the sigmoid and decay paths is acyclic block by block)
  always_comb begin
    rtag = tagq[0];
    rsp_last = rd_rvalid && !rtag.vec && rtag.last;
    x = rd_rdata; g = op0;
    for (int b = 0; b < L; b++) vrow[b] = cs_mode ? rowvec_q[b] : rowvec_q[vec_cs ? int'(rtag.row[3:0]) : 0];
    for (int b = 0; b < L; b++) v_bcast[b] = use_imm ? imm : vrow[b];
    // reduction input with invalid columns neutralised; partial base restarts at the identity after a row fires
    for (int b = 0; b < L; b++) begin
      logic valid_col; valid_col = (int'(rtag.cg) * L + b) < N;
      red_in[b] = (kind == K_ROWMAX) ? (valid_col ? rd_rdata[b] : NEG_INF) : (valid_col ? rd_rdata[b] : POS_ZERO);
      partial_base[b] = red_fire ? ((kind == K_ROWMAX) ? NEG_INF : POS_ZERO) : partial[b];
    end
  end
  // multiplier operands: x * v (imm / row vector), x * gate; SFU steps: t = x * -log2 e for the sigmoid chain
  always_comb begin
    for (int b = 0; b < L; b++) begin mul_a[b] = x[b]; mul_b[b] = (kind == K_GATE_MUL) ? g[b] : v_bcast[b]; end
    if (sfu_mode) for (int b = 0; b < SFU_LANES; b++) begin mul_a[b] = sfu_in[b]; mul_b[b] = NEG_LOG2E; end
  end
  // decay group: acc * exp2(gate) during the SFU steps, the lane's ordinary product otherwise
  always_comb begin
    for (int b = 0; b < SFU_LANES; b++) begin
      dec_a[b] = (sfu_mode && kind == K_GATE_DECAY) ? sfu_acc_in[b] : mul_a[SFU_LANES + b];
      dec_b[b] = (sfu_mode && kind == K_GATE_DECAY) ? sfu_u[b] : mul_b[SFU_LANES + b];
    end
  end
  // adder operands by kind; SFU steps: w = 1 + u for the sigmoid chain
  always_comb begin
    add_sub = 1'b0;
    for (int b = 0; b < L; b++) begin add_a[b] = x[b]; add_b[b] = v_bcast[b]; end
    case (kind)
      K_SUB, K_MAX: add_sub = 1'b1;
      K_GATE_ADD: for (int b = 0; b < L; b++) add_b[b] = g[b];
      K_ROWSUM, K_ROWMAX: begin
        add_sub = (kind == K_ROWMAX);
        for (int b = 0; b < L; b++) begin add_a[b] = partial_base[b]; add_b[b] = red_in[b]; end
      end
      K_EXP2, K_EXP2_SUM: begin
        if (exp_sum_pending) for (int b = 0; b < L; b++) begin add_a[b] = partial_base[b]; add_b[b] = exp_part_in[b]; end
        else begin add_sub = 1'b1; for (int b = 0; b < L; b++) add_b[b] = vrow[b]; end     // exp2 argument x - vec[r]
      end
      default: ;
    endcase
    if (sfu_mode) for (int b = 0; b < SFU_LANES; b++) begin add_a[b] = ONE; add_b[b] = sfu_u[b]; end
  end
  always_comb begin
    // one-shot slice result
    result = rd_rdata; result_ben = cs_mode ? slice_ben(M, int'(rtag.row)) : slice_ben(N, int'(rtag.cg)); result_push = 1'b0;
    case (kind)
      K_SUB, K_ADD, K_GATE_ADD: for (int b = 0; b < L; b++) result[b] = l_sum[b];
      K_MUL, K_SCALE, K_GATE_MUL: for (int b = 0; b < L; b++) result[b] = l_prod_all[b];
      K_MAX: for (int b = 0; b < L; b++) result[b] = l_max[b];
      K_NEG: for (int b = 0; b < L; b++) result[b] = {~rd_rdata[b][31], rd_rdata[b][30:0]};
      K_CVT_BF16: for (int b = 0; b < L; b++) result[b] = {l_bf[b], 16'd0};
      K_MASK_TILE: for (int b = 0; b < L; b++) begin result[b] = NEG_INF; result_ben[b] = result_ben[b] & f_nonzero(rd_rdata[b]); end
      K_SEL: for (int b = 0; b < L; b++) result[b] = f_nonzero(rd_rdata[b]) ? op0[b] : op1[b];
      default: ;
    endcase
    if (kind == K_SCALE) result_ben = slice_ben(int'(meta[slot_index(tref_sp(t_src), tref_slot(t_src) + 12'(int'(rtag.slot)))].cols), int'(rtag.cg));
    case (kind)
      K_SUB, K_MUL, K_ADD, K_MAX, K_NEG, K_SCALE, K_GATE_MUL, K_GATE_ADD, K_CVT_BF16, K_CVT_RAW, K_COPY, K_MASK_TILE, K_SEL:
        result_push = rsp_last;
      default: ;
    endcase
    // exp2 argument: x - vec[r] when a row vector is given
    for (int b = 0; b < L; b++) exp2_arg[b] = use_vec ? l_sum[b] : rd_rdata[b];
    // SFU lanes of the current step
    sfu_op = (kind == K_RCP) ? 2'd1 : (kind == K_LOG2) ? 2'd2 : (kind == K_SIGMOID) ? 2'd3 : 2'd0;
    for (int b = 0; b < SFU_LANES; b++) begin
      logic [31:0] xin; logic le_m127;
      xin = sfu_in[b];
      le_m127 = xin[31] && (xin[30:0] >= 31'h42fe0000) && !f_isnan(xin);     // x <= -127
      sfu_out[b] = (kind == K_GATE_DECAY) ? l_prod_dec[b] : (((kind == K_EXP2 || kind == K_EXP2_SUM) && le_m127) ? POS_ZERO : sfu_y[b]);
    end
    // ---- row reducer operands: level 0 pairs the partials, later levels the previous level's results ----
    for (int n = 0; n < RED_W; n++) begin
      red_a[n] = red_fire ? partial[2*n] : red_reg[2*n];
      red_b[n] = red_fire ? partial[2*n+1] : red_reg[2*n+1];
    end
    // ---- write FIFO inputs ----
    rsp_w = '0; rsp_w.cs = cs_mode; rsp_w.sp = tref_sp(t_dst); rsp_w.slot = tref_slot(t_dst) + 12'(kind == K_SCALE ? int'(rtag.slot) : 0);
    rsp_w.row = 10'(int'(rtag.row) + r0); rsp_w.cg = rtag.cg; rsp_w.ben = result_ben; rsp_w.data = result;
    rsp_w_push = result_push && write_slice;
    sfu_w = '0; sfu_w.cs = cs_mode;
    sfu_w.sp = (kind == K_EXP2_SUM) ? tref_sp(t_src) : tref_sp(t_dst);
    sfu_w.slot = (kind == K_EXP2_SUM) ? tref_slot(t_src) : tref_slot(t_dst);
    sfu_w.row = sfu_tag.row; sfu_w.cg = sfu_tag.cg; sfu_w.ben = cs_mode ? slice_ben(M, int'(sfu_tag.row)) : slice_ben(N, int'(sfu_tag.cg));
    for (int b = 0; b < L; b++) sfu_w.data[b] = sfu_slice[b];
    for (int b = 0; b < SFU_LANES; b++) sfu_w.data[sfu_step * SFU_LANES + b] = sfu_out[b];
    sfu_w_push = sfu_busy && (sfu_step == SFU_STEPS - 1) && write_slice;
    // the row result: element row % BANKS of column 0's row group (dst_2d), or lane 0 of row red_row_d
    row_w = '0; row_w.cs = dst_2d; row_w.sp = tref_sp(t_dst); row_w.slot = tref_slot(t_dst); row_w.cg = 0;
    row_w.row = dst_2d ? 10'(int'(red_row_d) / BANKS) : red_row_d;
    row_w.ben = dst_2d ? (L'(1) << (int'(red_row_d) % BANKS)) : L'(1);
    for (int b = 0; b < L; b++) row_w.data[b] = red_result;
    row_w_push = red_done && !sfu_w_push && (wq_cnt < WQ);        // held until the FIFO takes it
    topk_w = '0;
    topk_w.sp = topk_val_phase ? tref_sp(t_val) : tref_sp(t_dst); topk_w.slot = topk_val_phase ? tref_slot(t_val) : tref_slot(t_dst);
    topk_w.row = 0; topk_w.cg = 6'(topk_pass / L); topk_w.ben = L'(1) << (topk_pass % L);
    for (int b = 0; b < L; b++) topk_w.data[b] = topk_val_phase ? best_v : best_as_f32;
    // the sources are mutually exclusive by construction (one-shot results only for non-SFU, non-reduction kinds;
    // the SFU write and the reducer's write of VEXP2.SUM are ordered by row_w_push; topk pushes only between
    // passes; write-only kinds have no other source): one-hot AND-OR
    wq_push = rsp_w_push || sfu_w_push || row_w_push || topk_push || wo_push;
    wq_in = ({$bits(wreq_t){rsp_w_push}} & rsp_w) | ({$bits(wreq_t){sfu_w_push}} & sfu_w) | ({$bits(wreq_t){row_w_push}} & row_w) |
            ({$bits(wreq_t){topk_push}} & topk_w) | ({$bits(wreq_t){wo_push}} & wo_in);
    // write client
    wr_req = '0;
    if (wq_cnt > 0) begin
      if (wq[wq_rd].cs) wr_req = col_req(wq[wq_rd].slot, int'(wq[wq_rd].cg), int'(wq[wq_rd].row), 1'b1, wq[wq_rd].ben, wq[wq_rd].data);
      else wr_req = slice_req(wq[wq_rd].sp, wq[wq_rd].slot, int'(wq[wq_rd].row), int'(wq[wq_rd].cg), 1'b1, wq[wq_rd].ben, wq[wq_rd].data);
    end
    wq_pop = (wq_cnt > 0) && wr_grant;
  end

  // ================================================================ meta / reg / status outputs
  always_comb begin
    meta_we = 1'b0; meta_idx = '0; meta_wdata = '0; reg_we = 1'b0; reg_idx = '0; reg_wdata = '0;
    case (st)
      S_META: begin
        meta_we = 1'b1;
        meta_idx = 8'(slot_index(tref_sp(t_dst), tref_slot(t_dst)));
        meta_wdata = '{rows: 10'(M), cols: 10'(N), pready: 1'b0};
        case (kind)
          K_ROWMAX, K_ROWSUM: meta_wdata.cols = 10'd1;
          K_EXP2_SUM: begin meta_idx = 8'(slot_index(tref_sp(t_src), tref_slot(t_src))); end
          K_SCALE: begin
            meta_idx = 8'(slot_index(tref_sp(t_src), tref_slot(t_src) + 12'(s_slot)));
            meta_wdata = meta[slot_index(tref_sp(t_src), tref_slot(t_src) + 12'(s_slot))];
            meta_wdata.pready = 1'b0;
          end
          K_CVT_BF16: begin meta_wdata.rows = 10'(r0 != 0 ? r0 + M : M); meta_wdata.pready = inplace; end
          K_CVT_RAW:  meta_wdata.rows = 10'(r0 != 0 ? r0 + M : M);
          K_TOPK: begin meta_wdata.rows = 10'd1; meta_wdata.cols = 10'(kk); end
          K_COUNT: meta_we = 1'b0;
          default: ;
        endcase
      end
      S_META2: begin
        meta_we = 1'b1;
        if (kind == K_EXP2_SUM) begin
          meta_idx = 8'(slot_index(tref_sp(t_dst), tref_slot(t_dst)));
          meta_wdata = '{rows: 10'(M), cols: 10'd1, pready: 1'b0};
        end else begin
          meta_idx = 8'(slot_index(tref_sp(t_val), tref_slot(t_val)));
          meta_wdata = '{rows: 10'd1, cols: 10'(kk), pready: 1'b0};
        end
      end
      S_REG: begin reg_we = 1'b1; reg_idx = cur.a[3:0]; reg_wdata = 32'(count_acc); end
      default: ;
    endcase
    busy = (st != S_IDLE); done = (st == S_DONE);
  end

  // ================================================================ sequencer step (inlined into the always_ff)
  // rows / slices of the current slot and the metadata index of the next VSCALE_ACC slot (combinational)
  int rows_here, cgs_here, next_si;
  always_comb begin
    rows_here = cs_mode ? n_rg : (kind == K_SCALE) ? Mk : M;      // cs_mode: row groups outside, columns inside
    cgs_here  = cs_mode ? N : (kind == K_SCALE) ? n_cgk : n_cg;
    next_si   = slot_index(tref_sp(t_src), tref_slot(t_src) + 12'(s_slot + 1));
  end
`define ADVANCE_SLICE \
    if (s_cg + 1 < cgs_here) s_cg <= s_cg + 1; \
    else begin \
      s_cg <= 0; \
      if (kind == K_TOPK) begin \
        if (s_pass + 1 < n_pass) s_pass <= s_pass + 1; else s_done <= 1; \
      end else if (s_row + 1 < rows_here) begin \
        s_row <= s_row + 1; s_vec_pending <= row_vec && (cs_mode || !vec_cs || ((s_row + 1) % BANKS == 0)); \
      end else if (kind == K_SCALE && s_slot + 1 < n_slots) begin \
        s_slot <= s_slot + 1; s_row <= 0; \
        Mk <= int'(meta[next_si].rows); Nk <= int'(meta[next_si].cols); n_cgk <= n_slices(int'(meta[next_si].cols)); \
        s_vec_pending <= row_vec; \
        if (int'(meta[next_si].rows) == 0 || int'(meta[next_si].cols) == 0) s_done <= 1; \
      end else s_done <= 1; \
    end

  // ================================================================ sequential
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; cur <= '0; kind <= K_SUB; is_sfu <= 0; is_red <= 0; use_vec <= 0; use_imm <= 0; has_val <= 0; has_base <= 0;
      inplace <= 0; row_vec <= 0; write_slice <= 0; t_src <= '0; t_dst <= '0; t_vec <= '0; t_aux <= '0; t_val <= '0;
      cs_mode <= 0; vec_cs <= 0; dst_2d <= 0; n_rg <= 0;
      M <= 0; N <= 0; n_cg <= 0; r0 <= 0; kk <= 0; n_pass <= 0; width_b <= 0; n_ops <= 0; n_slots <= 1; q_off <= 0; k_off <= 0; imm <= 0;
      s_slot <= 0; s_row <= 0; s_cg <= 0; s_op <= 0; s_pass <= 0; Mk <= 0; Nk <= 0; n_cgk <= 0; total_rd <= 0; got_rd <= 0;
      s_vec_pending <= 0; s_done <= 0; tagq[0] <= '0; tagq[1] <= '0; tag_cnt <= 0;
      wq_wr <= 0; wq_rd <= 0; wq_cnt <= 0;
      op0 <= '0; op1 <= '0; partial <= '0; sfu_slice <= '0; acc_slice <= '0; exp_part_in <= '0; rowvec_q <= '0;
      sfu_step <= 0; sfu_busy <= 0; sfu_tag <= '0; exp_sum_pending <= 0; red_fire <= 0; red_row <= 0; topk_push <= 0;
      red_busy <= 0; red_done <= 0; red_lvl <= 0; red_row_q <= 0; red_row_d <= 0; red_result <= '0;
      for (int n = 0; n < L; n++) red_reg[n] <= '0;
      selected <= '0; best_v <= NEG_INF; best_i <= 0; best_valid <= 0; topk_pass <= 0; topk_val_phase <= 0; topk_fire <= 0;
      count_acc <= 0; fault <= 0; fault_code <= F_NONE;
    end else begin
      // ---- write FIFO ----
      wq_cnt <= wq_cnt + (wq_push ? 1 : 0) - (wq_pop ? 1 : 0);
      if (wq_push) begin wq[wq_wr] <= wq_in; wq_wr <= (wq_wr + 1) % WQ; end
      if (wq_pop) wq_rd <= (wq_rd + 1) % WQ;
      // ---- tag queue ----
      tag_cnt <= tag_cnt + (tag_wr ? 2'd1 : 2'd0) - (tag_rd ? 2'd1 : 2'd0);
      if (tag_rd) begin tagq[0] <= tagq[1]; if (tag_wr) tagq[tag_cnt - 2'd1] <= tag_in; end
      else if (tag_wr) tagq[tag_cnt] <= tag_in;
      red_fire <= 0; topk_push <= 0; exp_sum_pending <= 0;
      // ---- row reducer: level 0 in the red_fire cycle, then RED_LEVELS - 1 more levels; the result is held in
      // {red_done, red_result, red_row_d} until the write FIFO takes it (the last level waits while a result is held)
      if (red_fire) begin
        for (int n = 0; n < RED_W; n++) red_reg[n] <= (kind == K_ROWMAX) ? red_max[n] : red_sum[n];
        red_busy <= 1; red_lvl <= 1; red_row_q <= red_row;
      end else if (red_busy) begin
        if (red_lvl + 1 >= RED_LEVELS) begin
          if (!red_done || row_w_push) begin
            red_busy <= 0; red_done <= 1; red_result <= (kind == K_ROWMAX) ? red_max[0] : red_sum[0]; red_row_d <= red_row_q;
          end
        end else begin
          for (int n = 0; n < RED_W; n++) red_reg[n] <= (kind == K_ROWMAX) ? red_max[n] : red_sum[n];
          red_lvl <= red_lvl + 1;
        end
      end
      if (row_w_push && !(red_busy && red_lvl + 1 >= RED_LEVELS)) red_done <= 0;

      case (st)
        // ---------------------------------------------------------- decode
        S_IDLE: if (issue) begin cur <= ins; fault <= 0; fault_code <= F_NONE; st <= S_SETUP; end
        S_SETUP: begin
          logic [15:0] src_t, dst_t; int m, n, nops; logic rv, flt; fault_t fcode; kind_t kd; logic sfu, ws;
          flt = 0; fcode = F_NONE; nops = 1; rv = 0; sfu = 0; ws = 1; kd = K_SUB; dst_t = cur.a; src_t = cur.a;
          use_vec <= cur.flags[FL_VEC]; use_imm <= cur.flags[FL_IMM]; has_val <= cur.flags[FL_VAL]; has_base <= cur.flags[FL_BASE_REG];
          inplace <= cur.flags[FL_INPLACE]; imm <= cur.imm; r0 <= 0; kk <= 0; width_b <= int'(cur.c);
          q_off <= $signed(cur.imm[31:16]); k_off <= $signed(cur.imm[15:0]);
          n_slots <= 1; n_pass <= 1; t_vec <= cur.b; t_aux <= '0; t_val <= cur.c;
          case (cur.op)
            OP_VSUB, OP_VMUL, OP_VADD, OP_VMAX: begin
              kd = (cur.op == OP_VSUB) ? K_SUB : (cur.op == OP_VMUL) ? K_MUL : (cur.op == OP_VADD) ? K_ADD : K_MAX;
              rv = cur.flags[FL_VEC];
            end
            OP_VNEG: kd = K_NEG;
            OP_VSCALE_ACC: begin kd = K_SCALE; rv = 1; n_slots <= cur.flags[FL_WIDE] ? 4 : 1; end
            OP_VGATE: begin
              kd = (cur.qual == QUAL_OP_MUL) ? K_GATE_MUL : (cur.qual == QUAL_OP_ADD) ? K_GATE_ADD : K_GATE_DECAY;
              t_aux <= cur.b; nops = 2;
              if (cur.qual == QUAL_OP_DECAY) sfu = 1;
              if (cur.qual > QUAL_OP_DECAY) begin flt = 1; fcode = F_BAD_QUALIFIER; end
            end
            OP_VCVT: begin
              src_t = cur.b; dst_t = cur.a; r0 <= int'(cur.d);
              if (cur.flags[FL_INPLACE] || cur.a == cur.b) begin kd = K_CVT_BF16; dst_t = cur.b; inplace <= 1; r0 <= 0; end
              else if ((tref_sp(cur.a) == SP_STATE || tref_sp(cur.a) == SP_ACC) && !cur.flags[FL_TO_BF16]) kd = K_CVT_RAW;
              else kd = K_CVT_BF16;
            end
            OP_VCOPY: begin kd = K_COPY; src_t = cur.b; dst_t = cur.a; end
            OP_VMASK: begin
              t_aux <= cur.b;
              case (cur.qual)
                QUAL_KIND_CAUSAL: begin kd = K_MASK_CAUSAL; nops = 0; end
                QUAL_KIND_BAND:   begin kd = K_MASK_BAND; nops = 0; end
                QUAL_KIND_TILE:   begin kd = K_MASK_TILE; nops = 1; end
                default: begin flt = 1; fcode = F_BAD_QUALIFIER; end
              endcase
            end
            OP_VEXP2: begin kd = K_EXP2; rv = cur.flags[FL_VEC]; sfu = 1; end
            OP_VEXP2_SUM: begin kd = K_EXP2_SUM; dst_t = cur.b; t_vec <= cur.c; rv = cur.flags[FL_VEC]; sfu = 1; end
            OP_VRCP, OP_VLOG2, OP_VSIGMOID: begin
              kd = (cur.op == OP_VRCP) ? K_RCP : (cur.op == OP_VLOG2) ? K_LOG2 : K_SIGMOID; sfu = 1;
            end
            OP_VFILL: begin kd = K_FILL; nops = 0; end
            OP_VROWMAX, OP_VROWSUM: begin kd = (cur.op == OP_VROWMAX) ? K_ROWMAX : K_ROWSUM; src_t = cur.b; dst_t = cur.a; ws = 0; end
            OP_VSEL: begin kd = K_SEL; src_t = cur.b; dst_t = cur.a; t_aux <= cur.c; t_vec <= cur.d; nops = 3; end
            OP_VTOPK: begin kd = K_TOPK; src_t = cur.b; dst_t = cur.a; kk <= int'(cur.imm[31:16]); n_pass <= int'(cur.imm[31:16]); ws = 0; end
            OP_VCOUNT: begin kd = K_COUNT; src_t = cur.b; dst_t = cur.b; ws = 0; end
            default: begin flt = 1; fcode = F_BAD_OPCODE; end
          endcase
          kind <= kd; is_sfu <= sfu; is_red <= (kd == K_ROWMAX || kd == K_ROWSUM || kd == K_EXP2_SUM);
          n_ops <= nops; row_vec <= rv; write_slice <= ws; t_src <= src_t; t_dst <= dst_t;
          // E2: column-slice order when every tile operand is a 2-D VEC tile and the kind is lane-independent
          begin
            logic all2d;
            all2d = vec_is_2d(dst_t) && (nops == 0 || vec_is_2d(src_t)) &&
                    (cur.op != OP_VSEL || (vec_is_2d(cur.c) && vec_is_2d(cur.d)));
            cs_mode <= all2d && (kd == K_SUB || kd == K_MUL || kd == K_ADD || kd == K_MAX || kd == K_NEG || kd == K_COPY ||
                                 kd == K_SEL || kd == K_EXP2 || kd == K_RCP || kd == K_LOG2 || kd == K_SIGMOID || kd == K_FILL);
          end
          vec_cs <= vec_is_2d((cur.op == OP_VEXP2_SUM) ? cur.c : (cur.op == OP_VSEL) ? cur.d : cur.b);
          dst_2d <= vec_is_2d(dst_t);
          if (cur.op == OP_VFILL) begin m = int'(cur.c); n = int'(cur.d); end
          else begin
            m = int'(meta[slot_index(tref_sp(src_t), tref_slot(src_t))].rows);
            n = int'(meta[slot_index(tref_sp(src_t), tref_slot(src_t))].cols);
          end
          M <= m; N <= n; n_cg <= n_slices(n); Mk <= m; Nk <= n; n_cgk <= n_slices(n); n_rg <= n_slices(m);
          s_slot <= 0; s_row <= 0; s_cg <= 0; s_op <= 0; s_pass <= 0; s_vec_pending <= rv;
          s_done <= (m == 0 || n == 0 || (kd == K_TOPK && cur.imm[31:16] == 0));
          total_rd <= 0; got_rd <= 0; selected <= '0; best_valid <= 0; best_v <= NEG_INF; best_i <= 0; count_acc <= 0;
          sfu_busy <= 0; sfu_step <= 0; topk_fire <= 0; topk_val_phase <= 0;
          for (int b = 0; b < L; b++) partial[b] <= (kd == K_ROWMAX) ? NEG_INF : POS_ZERO;
          if (kd == K_TOPK && (m != 1 || cur.imm[15:0] != 16'd1)) begin flt = 1; fcode = F_BAD_SHAPE; end
          if (kd == K_TOPK && int'(cur.imm[31:16]) > T_V) begin flt = 1; fcode = F_BAD_SHAPE; end
          if (cur.op == OP_VCVT && !(cur.flags[FL_INPLACE] || cur.a == cur.b) &&
              (int'(cur.d) + m > sp_rows(tref_sp(cur.a), tref_slot(cur.a)) || n > sp_cols(tref_sp(cur.a), tref_slot(cur.a))))
          begin flt = 1; fcode = F_BAD_SHAPE; end
          if (cur.op == OP_VFILL && (m > sp_rows(tref_sp(cur.a), tref_slot(cur.a)) || n > sp_cols(tref_sp(cur.a), tref_slot(cur.a))))
          begin flt = 1; fcode = F_BAD_SHAPE; end
          if (flt) begin fault <= 1; fault_code <= fcode; st <= S_FAULT; end else st <= S_RUN;
        end
        // ---------------------------------------------------------- run
        S_RUN: begin
          // sequencer
          if (!s_done) begin
            if (n_ops == 0) begin
              if (wo_push) begin `ADVANCE_SLICE end
            end else if (tag_wr) begin
              total_rd <= total_rd + 1;
              if (s_vec_pending) s_vec_pending <= 0;
              else if (s_op + 1 < n_ops) s_op <= s_op + 1;
              else begin s_op <= 0; `ADVANCE_SLICE end
            end
          end
          // returned reads
          if (rd_rvalid) begin
            got_rd <= got_rd + 1;
            if (rtag.vec) begin if (vec_cs) rowvec_q <= rd_rdata; else for (int b = 0; b < L; b++) rowvec_q[b] <= rd_rdata[0]; end
            else begin
              if (rtag.op == 2'd0 && !rtag.last) op0 <= rd_rdata;
              if (rtag.op == 2'd1 && !rtag.last) op1 <= rd_rdata;
              if (rtag.last) begin
                case (kind)
                  K_ROWMAX, K_ROWSUM: begin
                    for (int b = 0; b < L; b++) partial[b] <= (kind == K_ROWMAX) ? l_max[b] : l_sum[b];
                    if (int'(rtag.cg) + 1 >= n_cg) begin red_fire <= 1; red_row <= rtag.row; end
                  end
                  K_EXP2, K_EXP2_SUM, K_RCP, K_LOG2, K_SIGMOID: begin
                    sfu_slice <= (kind == K_EXP2 || kind == K_EXP2_SUM) ? exp2_arg : rd_rdata;
                    sfu_busy <= 1; sfu_step <= 0; sfu_tag <= rtag;
                  end
                  K_GATE_DECAY: begin
                    sfu_slice <= op0; acc_slice <= rd_rdata; sfu_busy <= 1; sfu_step <= 0; sfu_tag <= rtag;
                  end
                  K_TOPK: begin
                    logic [31:0] bv; int bi; logic bval;
                    bv = best_v; bi = best_i; bval = best_valid;
                    for (int b = 0; b < L; b++) begin
                      int col; col = int'(rtag.cg) * L + b;
                      if (col < N && !selected[col] && (!bval || ord(rd_rdata[b]) > ord(bv))) begin
                        bv = rd_rdata[b]; bi = col; bval = 1;
                      end
                    end
                    best_v <= bv; best_i <= bi; best_valid <= bval;
                    if (int'(rtag.cg) + 1 >= n_cg) begin topk_fire <= 1; topk_pass <= int'(rtag.pass); end
                  end
                  K_COUNT: begin
                    int c; c = count_acc;
                    for (int b = 0; b < L; b++) begin
                      int col; logic nonneg; col = int'(rtag.cg) * L + b;
                      nonneg = !f_isnan(rd_rdata[b]) && (!rd_rdata[b][31] || rd_rdata[b][30:0] == 0);
                      if (col < N && nonneg && ord(rd_rdata[b]) < ord(bound_f)) c = c + 1;
                    end
                    count_acc <= c;
                  end
                  default: ;
                endcase
              end
            end
          end else if (red_fire) begin
            for (int b = 0; b < L; b++) partial[b] <= (kind == K_ROWMAX) ? NEG_INF : POS_ZERO;
          end
          // SFU step
          if (sfu_busy) begin
            for (int b = 0; b < SFU_LANES; b++) sfu_slice[sfu_step * SFU_LANES + b] <= sfu_out[b];
            if (kind == K_EXP2_SUM) begin
              for (int b = 0; b < SFU_LANES; b++) begin
                int col; col = int'(sfu_tag.cg) * L + sfu_step * SFU_LANES + b;
                exp_part_in[sfu_step * SFU_LANES + b] <= (col < N) ? sfu_out[b] : POS_ZERO;
              end
            end
            if (sfu_step + 1 >= SFU_STEPS) begin sfu_busy <= 0; if (kind == K_EXP2_SUM) exp_sum_pending <= 1; end
            sfu_step <= sfu_step + 1;
          end
          if (exp_sum_pending) begin
            for (int b = 0; b < L; b++) partial[b] <= l_sum[b];
            if (int'(sfu_tag.cg) + 1 >= n_cg) begin red_fire <= 1; red_row <= sfu_tag.row; end
          end
          // VTOPK: after a pass, write idx (and val), mark the column
          if (topk_fire && !topk_push && wq_cnt < WQ && !red_fire) begin
            topk_push <= 1;
            if (!topk_val_phase) begin
              selected[best_i] <= 1'b1;
              if (has_val) topk_val_phase <= 1;
              else begin topk_fire <= 0; best_valid <= 0; best_v <= NEG_INF; best_i <= 0; end
            end else begin
              topk_val_phase <= 0; topk_fire <= 0; best_valid <= 0; best_v <= NEG_INF; best_i <= 0;
            end
          end
          // completion
          if (s_done && got_rd >= total_rd && tag_cnt == 0 && !sfu_busy && !exp_sum_pending && !red_fire && !red_busy && !red_done &&
              !topk_fire && wq_cnt == 0 && !wq_push) begin
            st <= (kind == K_COUNT) ? S_REG : S_META;
            s_slot <= 0;
          end
        end
        S_META: if (meta_ack) begin
          if (kind == K_SCALE && s_slot + 1 < n_slots) s_slot <= s_slot + 1;
          else if (kind == K_EXP2_SUM || (kind == K_TOPK && has_val)) st <= S_META2;
          else st <= S_DONE;
        end
        S_META2: if (meta_ack) st <= S_DONE;
        S_REG: st <= S_DONE;
        S_DONE: st <= S_IDLE;
        S_FAULT: ;
        default: st <= S_IDLE;
      endcase
    end
  end
endmodule
