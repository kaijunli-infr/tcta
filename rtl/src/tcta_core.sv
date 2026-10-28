// TCTA core v0 top: front end, five queues, token network, five scratchpads with their arbiters, the tile
// metadata table, the LOAD / STORE engines, the MMA and VEC units, a COMM stub, the status word and the
// optional token checker (TOKEN_CHECK = 1 reproduces the simulator's UNTOKENED_* faults). Memory, the
// instruction stream and the descriptor table are provided by the testbench through the ports below.
module tcta_core
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
#(
  parameter bit TOKEN_CHECK = 1'b0
) (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              start,
  input  logic [31:0]       t_deadlock,
  // instruction stream
  output logic [31:0]       is_addr,
  input  logic [127:0]      is_data,
  input  logic [31:0]       n_words,
  // descriptor table write port
  input  logic              desc_we,
  input  logic [7:0]        desc_widx,
  input  logic [DESC_W-1:0] desc_wdata,
  // memory
  output logic              mem_rd_valid,
  output logic [ADDR_W-1:0] mem_rd_addr,
  input  logic              mem_rd_rvalid,
  input  logic [MEM_W-1:0]  mem_rd_data,
  input  logic              mem_rd_err,
  output logic              mem_wr_valid,
  output logic [ADDR_W-1:0] mem_wr_addr,
  output logic [MEM_W-1:0]  mem_wr_data,
  output logic [DMA_BYTES-1:0] mem_wr_be,
  // status
  output logic              running,
  output logic              finished,        // FINISH retired with every token FIFO empty
  output logic              fault,
  output fault_t            fault_code,
  output logic [2:0]        fault_q,
  output logic [31:0]       fault_pc,
  output logic [31:0]       cycle,
  // trace (per queue)
  output logic              tr_sync  [5], output logic [5:0] tr_sync_op  [5], output logic [31:0] tr_sync_pc  [5],
  output logic              tr_issue [5], output logic [5:0] tr_issue_op [5], output logic [31:0] tr_issue_pc [5],
  output logic              tr_done  [5], output logic [5:0] tr_done_op  [5], output logic [31:0] tr_done_pc  [5]
);
  // ================================================================ front end
  logic   fe_dq_valid; logic [2:0] fe_dq_q; instr_t fe_dq_ins; logic iq_full [5];
  logic [31:0] regs [N_REGS]; logic reg_we; logic [3:0] reg_idx; logic [31:0] reg_wdata;
  logic   fe_done, fe_progress, fe_fault; fault_t fe_fault_code; logic [31:0] fe_fault_pc;
  front_end u_fe (
    .clk(clk), .rst_n(rst_n), .start(start), .is_addr(is_addr), .is_data(is_data), .n_words(n_words),
    .dq_valid(fe_dq_valid), .dq_q(fe_dq_q), .dq_ins(fe_dq_ins), .iq_full(iq_full),
    .regs(regs), .reg_we(reg_we), .reg_idx(reg_idx), .reg_wdata(reg_wdata),
    .done(fe_done), .progress(fe_progress), .fault(fe_fault), .fault_code(fe_fault_code), .fault_pc(fe_fault_pc));

  // ================================================================ queues + token network
  logic [4:0] push_req, push_ok, pop_req, pop_ok; logic [2:0] push_to [5], pop_from [5];
  logic   q_issue [5]; instr_t q_ins [5]; logic u_busy [5]; logic u_done [5];
  logic   q_empty [5], q_inflight [5], q_progress [5], q_fault [5]; fault_t q_fault_code [5];
  logic [31:0] q_fault_pc [5], q_inflight_pc [5], q_head_pc [5];
  logic   net_all_empty; logic [63:0] net_nonempty;
  for (genvar q = 0; q < 5; q++) begin : g_q
    queue #(.QID(3'(q))) u_q (
      .clk(clk), .rst_n(rst_n),
      .dq_valid(fe_dq_valid && (fe_dq_q == 3'(q))), .dq_ins(fe_dq_ins), .full(iq_full[q]), .empty(q_empty[q]),
      .push_req(push_req[q]), .push_to(push_to[q]), .push_ok(push_ok[q]), .pop_req(pop_req[q]), .pop_from(pop_from[q]), .pop_ok(pop_ok[q]),
      .issue(q_issue[q]), .u_ins(q_ins[q]), .u_busy(u_busy[q]), .u_done(u_done[q]), .inflight(q_inflight[q]),
      .inflight_pc(q_inflight_pc[q]), .head_pc(q_head_pc[q]),
      .progress(q_progress[q]), .fault(q_fault[q]), .fault_code(q_fault_code[q]), .fault_pc(q_fault_pc[q]),
      .tr_sync(tr_sync[q]), .tr_sync_op(tr_sync_op[q]), .tr_sync_pc(tr_sync_pc[q]),
      .tr_issue(tr_issue[q]), .tr_issue_op(tr_issue_op[q]), .tr_issue_pc(tr_issue_pc[q]),
      .tr_done(tr_done[q]), .tr_done_op(tr_done_op[q]), .tr_done_pc(tr_done_pc[q]));
  end
  token_net u_net (
    .clk(clk), .rst_n(rst_n), .push_req(push_req), .push_to(push_to), .push_ok(push_ok),
    .pop_req(pop_req), .pop_from(pop_from), .pop_ok(pop_ok), .all_empty(net_all_empty), .nonempty(net_nonempty));

  // ================================================================ descriptor table
  logic [DESC_W-1:0] desc_mem [N_DESC];
  logic [7:0] ld_desc_idx, st_desc_idx; logic [DESC_W-1:0] ld_desc_data, st_desc_data;
  always_ff @(posedge clk) begin
    if (desc_we) desc_mem[desc_widx] <= desc_wdata;
    ld_desc_data <= desc_mem[ld_desc_idx];
    st_desc_data <= desc_mem[st_desc_idx];
  end

  // ================================================================ scratchpads
  sp_req_t creq [N_CLIENTS];
  logic    cgrant [N_CLIENTS], crvalid [N_CLIENTS]; slice_t crdata [N_CLIENTS];
  logic    g_grant [5][N_CLIENTS], g_rvalid [5][N_CLIENTS]; slice_t g_rdata [5][N_CLIENTS];
  for (genvar s = 0; s < 5; s++) begin : g_sp
    localparam int WORDS = (s == 0) ? INP_WORDS : (s == 1) ? WGT_WORDS : (s == 2) ? ACC_WORDS : (s == 3) ? VEC_WORDS : STATE_WORDS;
    localparam int DW = (s < 2) ? 16 : 32;
    scratchpad #(.WORDS(WORDS), .DW(DW), .SP_ID(4'(s))) u_sp (
      .clk(clk), .rst_n(rst_n), .req(creq), .grant(g_grant[s]), .rvalid(g_rvalid[s]), .rdata(g_rdata[s]));
  end
  always_comb begin
    for (int i = 0; i < N_CLIENTS; i++) begin
      cgrant[i] = 1'b0; crvalid[i] = 1'b0; crdata[i] = '0;
      for (int s = 0; s < 5; s++) begin
        cgrant[i] |= g_grant[s][i];
        crvalid[i] |= g_rvalid[s][i];
        if (g_rvalid[s][i]) crdata[i] = g_rdata[s][i];
      end
    end
  end

  // ================================================================ tile metadata
  tmeta_t meta [N_SLOTS];
  logic   m_we [3]; logic [7:0] m_idx [3]; tmeta_t m_data [3]; logic m_ack [3];
  tile_meta u_meta (.clk(clk), .rst_n(rst_n), .wr_valid(m_we), .wr_idx(m_idx), .wr_data(m_data), .wr_ack(m_ack), .meta(meta));

  // ================================================================ units
  logic   u_fault [5]; fault_t u_fault_code [5];
  dma_load u_load (
    .clk(clk), .rst_n(rst_n), .ins(q_ins[Q_LOAD]), .issue(q_issue[Q_LOAD]), .busy(u_busy[Q_LOAD]), .done(u_done[Q_LOAD]),
    .desc_idx(ld_desc_idx), .desc_data(ld_desc_data),
    .mem_valid(mem_rd_valid), .mem_addr(mem_rd_addr), .mem_rvalid(mem_rd_rvalid), .mem_rdata(mem_rd_data), .mem_rerr(mem_rd_err),
    .sp_req(creq[CL_DMA_LOAD]), .sp_grant(cgrant[CL_DMA_LOAD]), .sp_rvalid(crvalid[CL_DMA_LOAD]), .sp_rdata(crdata[CL_DMA_LOAD]),
    .meta(meta), .meta_we(m_we[0]), .meta_idx(m_idx[0]), .meta_wdata(m_data[0]), .meta_ack(m_ack[0]),
    .fault(u_fault[Q_LOAD]), .fault_code(u_fault_code[Q_LOAD]));
  dma_store u_store (
    .clk(clk), .rst_n(rst_n), .ins(q_ins[Q_STORE]), .issue(q_issue[Q_STORE]), .busy(u_busy[Q_STORE]), .done(u_done[Q_STORE]),
    .desc_idx(st_desc_idx), .desc_data(st_desc_data),
    .mem_valid(mem_wr_valid), .mem_addr(mem_wr_addr), .mem_wdata(mem_wr_data), .mem_be(mem_wr_be),
    .sp_req(creq[CL_DMA_STORE]), .sp_grant(cgrant[CL_DMA_STORE]), .sp_rvalid(crvalid[CL_DMA_STORE]), .sp_rdata(crdata[CL_DMA_STORE]),
    .meta(meta), .fault(u_fault[Q_STORE]), .fault_code(u_fault_code[Q_STORE]));
  mma_unit u_mma (
    .clk(clk), .rst_n(rst_n), .ins(q_ins[Q_MMA]), .issue(q_issue[Q_MMA]), .busy(u_busy[Q_MMA]), .done(u_done[Q_MMA]),
    .a_req(creq[CL_MMA_A]), .a_grant(cgrant[CL_MMA_A]), .a_rdata(crdata[CL_MMA_A]),
    .b_req(creq[CL_MMA_B]), .b_grant(cgrant[CL_MMA_B]), .b_rdata(crdata[CL_MMA_B]),
    .wb_req(creq[CL_MMA_WB]), .wb_grant(cgrant[CL_MMA_WB]), .wb_rvalid(crvalid[CL_MMA_WB]), .wb_rdata(crdata[CL_MMA_WB]),
    .meta(meta), .meta_we(m_we[1]), .meta_idx(m_idx[1]), .meta_wdata(m_data[1]), .meta_ack(m_ack[1]),
    .fault(u_fault[Q_MMA]), .fault_code(u_fault_code[Q_MMA]));
  vec_unit u_vec (
    .clk(clk), .rst_n(rst_n), .ins(q_ins[Q_VEC]), .issue(q_issue[Q_VEC]), .busy(u_busy[Q_VEC]), .done(u_done[Q_VEC]),
    .rd_req(creq[CL_VEC_RD]), .rd_grant(cgrant[CL_VEC_RD]), .rd_rvalid(crvalid[CL_VEC_RD]), .rd_rdata(crdata[CL_VEC_RD]),
    .wr_req(creq[CL_VEC_WR]), .wr_grant(cgrant[CL_VEC_WR]),
    .meta(meta), .meta_we(m_we[2]), .meta_idx(m_idx[2]), .meta_wdata(m_data[2]), .meta_ack(m_ack[2]),
    .regs(regs), .reg_we(reg_we), .reg_idx(reg_idx), .reg_wdata(reg_wdata),
    .fault(u_fault[Q_VEC]), .fault_code(u_fault_code[Q_VEC]));
  // COMM stub: SEND / RECV are BAD_OPCODE in v0 (the link is not part of this phase)
  logic comm_fault;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) comm_fault <= 1'b0;
    else if (q_issue[Q_COMM]) comm_fault <= 1'b1;
  end
  assign u_fault[Q_COMM] = comm_fault;
  assign u_fault_code[Q_COMM] = comm_fault ? F_BAD_OPCODE : F_NONE;
  assign u_busy[Q_COMM] = comm_fault;
  assign u_done[Q_COMM] = 1'b0;

  // ================================================================ token checker (debug)
  logic tc_fault; fault_t tc_fault_code; logic [2:0] tc_fault_q; logic [31:0] tc_fault_pc;
  if (TOKEN_CHECK) begin : g_tc
    localparam int N_EV = N_CLIENTS + 15;      // per-access events + up to three operand tiles per issuing queue
    logic ev_valid [N_EV], ev_we [N_EV]; logic [7:0] ev_slot [N_EV]; logic [2:0] ev_q [N_EV]; logic [31:0] ev_pc [N_EV];
    logic [4:0] pop_fire;
    function automatic logic [2:0] client_q(input int c);
      case (c)
        CL_MMA_A, CL_MMA_B, CL_MMA_WB: return Q_MMA;
        CL_VEC_RD, CL_VEC_WR: return Q_VEC;
        CL_DMA_LOAD: return Q_LOAD;
        default: return Q_STORE;
      endcase
    endfunction
    // the operand tiles an instruction reads at issue (machine.py's _note_read calls), up to three
    function automatic logic [47:0] issue_reads(input instr_t i);   // {valid2, tref2, valid1, tref1, valid0, tref0}
      logic v0, v1, v2; logic [15:0] t0, t1, t2;
      v0 = 0; v1 = 0; v2 = 0; t0 = i.a; t1 = i.b; t2 = i.c;
      case (i.op)
        OP_LOAD_IDX: begin v0 = 1; t0 = i.c; end
        OP_STORE_2D: begin v0 = 1; t0 = i.a; end
        OP_STORE_IDX: begin v0 = 1; t0 = i.a; v1 = 1; t1 = i.c; end
        OP_MMA, OP_MMA_T, OP_MMA_MASKED: begin
          v0 = 1; t0 = i.b; v1 = 1; t1 = i.c;
          if (i.op == OP_MMA_MASKED) begin v2 = 1; t2 = i.d; end
          else if (i.flags[FL_ACCUMULATE]) begin v2 = 1; t2 = i.a; end
        end
        OP_VROWMAX, OP_VROWSUM, OP_VCVT, OP_VCOPY, OP_VTOPK, OP_VCOUNT: begin v0 = 1; t0 = i.b; end
        OP_VSUB, OP_VMUL, OP_VADD, OP_VMAX, OP_VEXP2: begin v0 = 1; t0 = i.a; v1 = i.flags[FL_VEC]; t1 = i.b; end
        OP_VNEG, OP_VRCP, OP_VLOG2, OP_VSIGMOID: begin v0 = 1; t0 = i.a; end
        OP_VSCALE_ACC, OP_VGATE: begin v0 = 1; t0 = i.a; v1 = 1; t1 = i.b; end
        OP_VMASK: begin v0 = 1; t0 = i.a; v1 = (i.qual == QUAL_KIND_TILE); t1 = i.b; end
        OP_VEXP2_SUM: begin v0 = 1; t0 = i.a; v1 = i.flags[FL_VEC]; t1 = i.c; end
        OP_VSEL: begin v0 = 1; t0 = i.b; v1 = 1; t1 = i.c; v2 = 1; t2 = i.d; end
        default: ;
      endcase
      return {v2, t2, v1, t1, v0, t0};
    endfunction
    always_comb begin
      for (int c = 0; c < N_CLIENTS; c++) begin
        ev_valid[c] = creq[c].valid && cgrant[c]; ev_we[c] = creq[c].we;
        ev_slot[c] = 8'(slot_index(creq[c].sp, creq[c].slot)); ev_q[c] = client_q(c); ev_pc[c] = q_inflight_pc[client_q(c)];
      end
      for (int q = 0; q < 5; q++) begin
        logic [47:0] r; r = issue_reads(q_ins[q]);
        for (int k = 0; k < 3; k++) begin
          ev_valid[N_CLIENTS + q*3 + k] = q_issue[q] && r[k*16 + 16];
          ev_we[N_CLIENTS + q*3 + k] = 1'b0;
          ev_slot[N_CLIENTS + q*3 + k] = 8'(slot_index(tref_sp(r[k*16 +: 16]), tref_slot(r[k*16 +: 16])));
          ev_q[N_CLIENTS + q*3 + k] = 3'(q); ev_pc[N_CLIENTS + q*3 + k] = q_ins[q].pc;
        end
      end
      pop_fire = pop_req & pop_ok;
    end
    token_check #(.N_EV(N_EV)) u_tc (
      .clk(clk), .rst_n(rst_n), .ev_valid(ev_valid), .ev_we(ev_we), .ev_slot(ev_slot), .ev_q(ev_q), .ev_pc(ev_pc),
      .pop_valid(pop_fire), .pop_from(pop_from), .fault(tc_fault), .fault_code(tc_fault_code), .fault_q(tc_fault_q), .fault_pc(tc_fault_pc));
  end else begin : g_notc
    assign tc_fault = 1'b0; assign tc_fault_code = F_NONE; assign tc_fault_q = '0; assign tc_fault_pc = '0;
  end

  // ================================================================ status: finish, faults, deadlock
  logic all_idle, any_progress, running_q;
  logic [31:0] idle_cnt;
  always_comb begin
    all_idle = fe_done;
    any_progress = fe_progress;
    for (int q = 0; q < 5; q++) begin
      all_idle &= q_empty[q] && !u_busy[q] && !q_inflight[q];
      any_progress |= q_progress[q] || u_busy[q];
    end
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      running_q <= 0; finished <= 0; fault <= 0; fault_code <= F_NONE; fault_q <= '0; fault_pc <= '0; cycle <= 0; idle_cnt <= 0;
    end else begin
      if (start) begin running_q <= 1; finished <= 0; fault <= 0; fault_code <= F_NONE; cycle <= 0; idle_cnt <= 0; end
      else if (running_q) begin
        cycle <= cycle + 1;
        idle_cnt <= any_progress ? 0 : idle_cnt + 1;
        if (!fault && !finished) begin
          // first fault wins; the order is only a tie-break
          if (tc_fault) begin fault <= 1; fault_code <= tc_fault_code; fault_q <= tc_fault_q; fault_pc <= tc_fault_pc; end
          else if (u_fault[Q_LOAD]) begin fault <= 1; fault_code <= u_fault_code[Q_LOAD]; fault_q <= Q_LOAD; fault_pc <= q_inflight_pc[Q_LOAD]; end
          else if (u_fault[Q_MMA]) begin fault <= 1; fault_code <= u_fault_code[Q_MMA]; fault_q <= Q_MMA; fault_pc <= q_inflight_pc[Q_MMA]; end
          else if (u_fault[Q_VEC]) begin fault <= 1; fault_code <= u_fault_code[Q_VEC]; fault_q <= Q_VEC; fault_pc <= q_inflight_pc[Q_VEC]; end
          else if (u_fault[Q_STORE]) begin fault <= 1; fault_code <= u_fault_code[Q_STORE]; fault_q <= Q_STORE; fault_pc <= q_inflight_pc[Q_STORE]; end
          else if (u_fault[Q_COMM]) begin fault <= 1; fault_code <= u_fault_code[Q_COMM]; fault_q <= Q_COMM; fault_pc <= q_inflight_pc[Q_COMM]; end
          else if (q_fault[0] || q_fault[1] || q_fault[2] || q_fault[3] || q_fault[4]) begin
            fault <= 1;
            for (int q = 4; q >= 0; q--) if (q_fault[q]) begin fault_code <= q_fault_code[q]; fault_q <= 3'(q); fault_pc <= q_fault_pc[q]; end
          end
          else if (fe_fault) begin fault <= 1; fault_code <= fe_fault_code; fault_q <= Q_FE; fault_pc <= fe_fault_pc; end
          else if (idle_cnt > t_deadlock) begin
            fault <= 1; fault_code <= F_TOKEN_TIMEOUT; fault_q <= Q_FE; fault_pc <= '0;
            for (int q = 4; q >= 0; q--) if (!q_empty[q]) begin fault_q <= 3'(q); fault_pc <= q_head_pc[q]; end
          end
          else if (all_idle) begin
            if (net_all_empty) finished <= 1;
            else begin fault <= 1; fault_code <= F_TOKEN_LEAK; fault_q <= Q_FE; fault_pc <= '0; end
          end
        end
      end
    end
  end
  assign running = running_q && !finished && !fault;
endmodule
