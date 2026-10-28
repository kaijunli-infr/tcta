// MMA unit (MMA, MMA.T, MMA.MASKED; ISA §5.4, erratum E1): an output-stationary MAC_ROWS x MAC_COLS systolic
// array of mma_pe. A product [M, K] x [K, N] is ceil(M/16) * ceil(N/16) output blocks. Row i of the array
// streams a[i0+i][k] for k = 0..K-1 starting at cycle T_blk + i, column j streams b[k][j0+j] (MMA.T:
// b[j0+j][k]) starting at T_blk + j: the skew makes every operand fetch hit BANKS distinct banks, one read per
// cycle per operand. Blocks start every P = ceil16(max(K, 16)) cycles (the multiple of 16 keeps the two
// in-flight blocks' diagonals on disjoint banks) as long as the result FIFO has room, so the array never stalls.
// Finished rows (one per cycle per block) go through the result FIFO to the writeback stage: mask (add / mul),
// then + old (accumulate), then the write; `wide` and `row0` are address arithmetic there.
module mma_unit
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
(
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
  localparam int BLKQ = 2;                      // per-row / per-column queue of block coordinates

  // ---------------------------------------------------------------- instruction decode
  typedef enum logic [2:0] {S_IDLE, S_SETUP, S_RUN, S_META, S_DONE, S_FAULT} state_t;
  state_t st;
  instr_t cur;
  logic [15:0] acc_t, a_t, b_t, mask_t;
  logic transpose, masked, mask_mul, accumulate, wide;
  int M, K, N, r0, n_ib, n_jb, n_blocks, P;

  // ---------------------------------------------------------------- block scheduler
  int   blk;            // next block to start
  int   pt;             // cycles since the last block start
  int   committed;      // rows started (16 per block) and not yet popped from the result FIFO
  logic started_any;
  logic start_now;
  int   start_i0, start_j0;
  always_comb begin
    start_now = (st == S_RUN) && (blk < n_blocks) && (!started_any || pt >= P) && (committed <= RF_DEPTH - R);
    start_i0 = (blk / n_jb) * R;
    start_j0 = (blk % n_jb) * C;
  end

  // ---------------------------------------------------------------- row / column controllers
  // start tokens travel down the rows (one per cycle) and right along the columns
  logic rs_v [R];  int rs_i0 [R]; int rs_j0 [R];
  logic cs_v [C];  int cs_i0 [C]; int cs_j0 [C];
  logic r_act [R]; int r_k [R]; int r_i0 [R];
  logic c_act [C]; int c_k [C]; int c_j0 [C];
  logic r_act_now [R]; int r_k_now [R]; int r_i0_now [R];
  logic c_act_now [C]; int c_k_now [C]; int c_j0_now [C];
  always_comb begin
    for (int i = 0; i < R; i++) begin
      r_act_now[i] = rs_v[i] || r_act[i];
      r_k_now[i]   = rs_v[i] ? 0 : r_k[i];
      r_i0_now[i]  = rs_v[i] ? rs_i0[i] : r_i0[i];
    end
    for (int j = 0; j < C; j++) begin
      c_act_now[j] = cs_v[j] || c_act[j];
      c_k_now[j]   = cs_v[j] ? 0 : c_k[j];
      c_j0_now[j]  = cs_v[j] ? cs_j0[j] : c_j0[j];
    end
  end

  // ---------------------------------------------------------------- operand fetch
  always_comb begin
    a_req = '0; b_req = '0;
    a_req.sp = tref_sp(a_t); b_req.sp = tref_sp(b_t); a_req.slot = tref_slot(a_t); b_req.slot = tref_slot(b_t);
    for (int i = 0; i < R; i++) if (r_act_now[i]) begin
      a_req.valid = 1'b1;
      a_req.addr[r_k_now[i] % BANKS] = sp_addr(tref_sp(a_t), tref_slot(a_t), r_i0_now[i] + i, r_k_now[i] / BANKS);
    end
    for (int j = 0; j < C; j++) if (c_act_now[j]) begin
      b_req.valid = 1'b1;
      if (transpose) b_req.addr[c_k_now[j] % BANKS] = sp_addr(tref_sp(b_t), tref_slot(b_t), c_j0_now[j] + j, c_k_now[j] / BANKS);
      else           b_req.addr[j] = sp_addr(tref_sp(b_t), tref_slot(b_t), c_k_now[j], (c_j0_now[j] + j) / BANKS);
    end
  end

  // registered fetch state for the response cycle
  logic r_act_q [R]; int r_k_q [R];
  logic c_act_q [C]; int c_k_q [C];
  // array edge registers
  logic [15:0] a_edge [R]; logic v_edge [R]; logic first_edge [R]; logic last_edge [R];
  logic [15:0] b_edge [C];

  // ---------------------------------------------------------------- the array
  logic [15:0] a_w [R][C+1]; logic [15:0] b_w [R+1][C];
  logic v_w [R][C+1]; logic first_w [R][C+1]; logic last_w [R][C+1];
  logic [31:0] res [R][C]; logic res_v [R][C];
  for (genvar i = 0; i < R; i++) begin : g_row
    assign a_w[i][0] = a_edge[i]; assign v_w[i][0] = v_edge[i]; assign first_w[i][0] = first_edge[i]; assign last_w[i][0] = last_edge[i];
    for (genvar j = 0; j < C; j++) begin : g_col
      if (i == 0) begin : g_top
        assign b_w[0][j] = b_edge[j];
      end
      mma_pe u_pe (
        .clk(clk), .rst_n(rst_n),
        .a_in(a_w[i][j]), .b_in(b_w[i][j]), .v_in(v_w[i][j]), .first_in(first_w[i][j]), .last_in(last_w[i][j]),
        .a_out(a_w[i][j+1]), .b_out(b_w[i+1][j]), .v_out(v_w[i][j+1]), .first_out(first_w[i][j+1]), .last_out(last_w[i][j+1]),
        .result(res[i][j]), .result_valid(res_v[i][j]));
    end
  end

  // ---------------------------------------------------------------- per-row block coordinate queues
  int  rq_i0 [R][BLKQ]; int rq_j0 [R][BLKQ]; logic [0:0] rq_rd [R]; logic [0:0] rq_wr [R];

  // ---------------------------------------------------------------- result FIFO
  typedef struct packed { logic [3:0] i; logic [9:0] i0; logic [9:0] j0; slice_t data; } rrow_t;
  rrow_t rf [RF_DEPTH];
  int    rf_wr, rf_rd, rf_cnt;
  logic  rf_push; rrow_t rf_in;
  always_comb begin
    rf_push = 1'b0; rf_in = '0;
    for (int i = 0; i < R; i++) if (res_v[i][C-1]) begin     // rows of one block finish one per cycle
      rf_push = 1'b1;
      rf_in.i = 4'(i); rf_in.i0 = 10'(rq_i0[i][rq_rd[i]]); rf_in.j0 = 10'(rq_j0[i][rq_rd[i]]);
      for (int j = 0; j < C; j++) rf_in.data[j] = res[i][j];
    end
  end

  // ---------------------------------------------------------------- writeback
  typedef enum logic [2:0] {W_IDLE, W_MASK, W_MASK_WAIT, W_MCALC, W_OLD, W_OLD_WAIT, W_WRITE} wstate_t;
  wstate_t wst;
  rrow_t   wrow;
  slice_t  s_masked, s_out, mask_d, old_d;
  int      popped;
  logic    wb_pop;
  logic [31:0] mk_add [BANKS], mk_mul [BANKS], o_add [BANKS];
  for (genvar b = 0; b < BANKS; b++) begin : g_wb
    f32_add u_ma (.a(wrow.data[b]), .b(mask_d[b]), .sub(1'b0), .y(mk_add[b]));
    f32_mul u_mm (.a(wrow.data[b]), .b(mask_d[b]), .y(mk_mul[b]));
    f32_add u_oa (.a(old_d[b]), .b(s_masked[b]), .sub(1'b0), .y(o_add[b]));
  end
  always_comb begin
    for (int b = 0; b < BANKS; b++) s_out[b] = accumulate ? o_add[b] : s_masked[b];
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
    wb_pop = (wst == W_IDLE) && (rf_cnt > 0) && (st == S_RUN);
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
      M <= 0; K <= 0; N <= 0; r0 <= 0; n_ib <= 0; n_jb <= 0; n_blocks <= 0; P <= 16;
      blk <= 0; pt <= 0; committed <= 0; started_any <= 0; mk <= 0; popped <= 0;
      rf_wr <= 0; rf_rd <= 0; rf_cnt <= 0; wst <= W_IDLE; wrow <= '0; s_masked <= '0; mask_d <= '0; old_d <= '0;
      for (int i = 0; i < R; i++) begin
        rs_v[i] <= 0; rs_i0[i] <= 0; rs_j0[i] <= 0; r_act[i] <= 0; r_k[i] <= 0; r_i0[i] <= 0; r_act_q[i] <= 0; r_k_q[i] <= 0;
        a_edge[i] <= '0; v_edge[i] <= 0; first_edge[i] <= 0; last_edge[i] <= 0; rq_rd[i] <= 0; rq_wr[i] <= 0;
        for (int q = 0; q < BLKQ; q++) begin rq_i0[i][q] <= 0; rq_j0[i][q] <= 0; end
      end
      for (int j = 0; j < C; j++) begin
        cs_v[j] <= 0; cs_i0[j] <= 0; cs_j0[j] <= 0; c_act[j] <= 0; c_k[j] <= 0; c_j0[j] <= 0; c_act_q[j] <= 0; c_k_q[j] <= 0;
        b_edge[j] <= '0;
      end
    end else begin
      // ---- instruction FSM ----
      case (st)
        S_IDLE: if (issue) begin cur <= ins; fault <= 0; fault_code <= F_NONE; st <= S_SETUP; end
        S_SETUP: begin
          int m, k, n, pp;
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
          M <= m; K <= k; N <= n;
          n_ib <= (m + R - 1) / R; n_jb <= (n + C - 1) / C; n_blocks <= ((m + R - 1) / R) * ((n + C - 1) / C);
          pp = (k < 16) ? 16 : k; P <= ((pp + 15) / 16) * 16;
          blk <= 0; pt <= 0; committed <= 0; started_any <= 0; popped <= 0; mk <= 0;
          if (k == 0 || tref_sp(cur.a) > SP_STATE || tref_sp(cur.b) > SP_STATE || tref_sp(cur.c) > SP_STATE) begin
            fault <= 1; fault_code <= F_BAD_SHAPE; st <= S_FAULT;
          end else if (cur.op == OP_MMA_MASKED && cur.qual > QUAL_MASK_MODE_MUL) begin
            fault <= 1; fault_code <= F_BAD_QUALIFIER; st <= S_FAULT;
          end else st <= S_RUN;
        end
        S_RUN: begin
          if (popped >= n_blocks * R && wst == W_IDLE && rf_cnt == 0 && blk >= n_blocks) st <= S_META;
        end
        S_META: if (meta_ack) begin
          if (r0 != 0 || !wide || mk == 3) st <= S_DONE; else mk <= mk + 1;
        end
        S_DONE: st <= S_IDLE;
        S_FAULT: ;
        default: st <= S_IDLE;
      endcase

      // ---- scheduler ----
      if (st == S_RUN) begin
        pt <= pt + 1;
        if (start_now) begin blk <= blk + 1; pt <= 1; started_any <= 1; end
      end
      committed <= committed + (start_now ? R : 0) - (wb_pop ? 1 : 0);

      // ---- start tokens ----
      rs_v[0] <= start_now; rs_i0[0] <= start_i0; rs_j0[0] <= start_j0;
      cs_v[0] <= start_now; cs_i0[0] <= start_i0; cs_j0[0] <= start_j0;
      for (int i = 1; i < R; i++) begin rs_v[i] <= rs_v[i-1]; rs_i0[i] <= rs_i0[i-1]; rs_j0[i] <= rs_j0[i-1]; end
      for (int j = 1; j < C; j++) begin cs_v[j] <= cs_v[j-1]; cs_i0[j] <= cs_i0[j-1]; cs_j0[j] <= cs_j0[j-1]; end

      // ---- controllers ----
      for (int i = 0; i < R; i++) begin
        if (r_act_now[i]) begin
          r_k[i] <= r_k_now[i] + 1; r_act[i] <= (r_k_now[i] + 1 < K); r_i0[i] <= r_i0_now[i];
        end
        r_act_q[i] <= r_act_now[i]; r_k_q[i] <= r_k_now[i];
        if (rs_v[i]) begin                       // remember this block's coordinates for the result FIFO
          rq_i0[i][rq_wr[i]] <= rs_i0[i]; rq_j0[i][rq_wr[i]] <= rs_j0[i]; rq_wr[i] <= rq_wr[i] + 1'b1;
        end
        if (res_v[i][C-1]) rq_rd[i] <= rq_rd[i] + 1'b1;
      end
      for (int j = 0; j < C; j++) begin
        if (c_act_now[j]) begin
          c_k[j] <= c_k_now[j] + 1; c_act[j] <= (c_k_now[j] + 1 < K); c_j0[j] <= c_j0_now[j];
        end
        c_act_q[j] <= c_act_now[j]; c_k_q[j] <= c_k_now[j];
      end

      // ---- response -> edge registers ----
      for (int i = 0; i < R; i++) begin
        a_edge[i] <= a_rdata[r_k_q[i] % BANKS][31:16];
        v_edge[i] <= r_act_q[i];
        first_edge[i] <= (r_k_q[i] == 0);
        last_edge[i] <= (r_k_q[i] == K - 1);
      end
      for (int j = 0; j < C; j++) begin
        if (transpose) b_edge[j] <= b_rdata[c_k_q[j] % BANKS][31:16];
        else           b_edge[j] <= b_rdata[j][31:16];
      end

      // ---- result FIFO ----
      if (rf_push) begin rf[rf_wr] <= rf_in; rf_wr <= (rf_wr + 1) % RF_DEPTH; end
      if (wb_pop) rf_rd <= (rf_rd + 1) % RF_DEPTH;
      rf_cnt <= rf_cnt + (rf_push ? 1 : 0) - (wb_pop ? 1 : 0);

      // ---- writeback FSM ----
      case (wst)
        W_IDLE: if (wb_pop) begin
          wrow <= rf[rf_rd]; popped <= popped + 1;
          s_masked <= rf[rf_rd].data;
          if (int'(rf[rf_rd].i0) + int'(rf[rf_rd].i) >= M) wst <= W_IDLE;    // padding row of the last block
          else if (masked) wst <= W_MASK;
          else wst <= accumulate ? W_OLD : W_WRITE;
        end
        W_MASK: if (wb_grant) wst <= W_MASK_WAIT;
        W_MASK_WAIT: if (wb_rvalid) begin mask_d <= wb_rdata; wst <= W_MCALC; end
        W_MCALC: begin
          for (int b = 0; b < BANKS; b++) s_masked[b] <= mask_mul ? mk_mul[b] : mk_add[b];
          wst <= accumulate ? W_OLD : W_WRITE;
        end
        W_OLD: if (wb_grant) wst <= W_OLD_WAIT;
        W_OLD_WAIT: if (wb_rvalid) begin old_d <= wb_rdata; wst <= W_WRITE; end
        W_WRITE: if (wb_grant) wst <= W_IDLE;
        default: wst <= W_IDLE;
      endcase
    end
  end
endmodule
