// One instruction queue (ISA §8): an in-order buffer of dispatched instructions. Each cycle the head retires
// one leading PUSH / POP that can proceed (POP: a token is available; PUSH: the edge FIFO is not full and no
// work of this queue is in flight -- the trailing PUSH fires at the producer's completion, §8.5) and issues
// at most one work instruction to the unit when it is idle; a sync that retires and a following work
// instruction can go in the same cycle, so syncs cost no issue slot. PUSH / POP on an edge outside §8.1 is
// BAD_EDGE. Deadlock detection (TOKEN_TIMEOUT) is global, in tcta_core.
module queue
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
#(
  parameter logic [2:0] QID = 3'd0
) (
  input  logic     clk,
  input  logic     rst_n,
  // dispatch
  input  logic     dq_valid,
  input  instr_t   dq_ins,
  output logic     full,
  output logic     empty,
  // token network
  output logic     push_req,
  output logic [2:0] push_to,
  input  logic     push_ok,
  output logic     pop_req,
  output logic [2:0] pop_from,
  input  logic     pop_ok,
  // unit
  output logic     issue,
  output instr_t   u_ins,
  input  logic     u_busy,
  input  logic     u_done,
  output logic     inflight,
  output logic [31:0] inflight_pc,
  output logic [31:0] head_pc,
  // status / trace
  output logic     progress,
  output logic     fault,
  output fault_t   fault_code,
  output logic [31:0] fault_pc,
  output logic     tr_sync,  output logic [5:0] tr_sync_op,  output logic [31:0] tr_sync_pc,
  output logic     tr_issue, output logic [5:0] tr_issue_op, output logic [31:0] tr_issue_pc,
  output logic     tr_done,  output logic [5:0] tr_done_op,  output logic [31:0] tr_done_pc
);
  instr_t iq [IQ_DEPTH];
  int rd, wr, cnt;
  logic inflight_q; logic [31:0] inflight_pc_q; logic [5:0] inflight_op_q;

  instr_t h0, h1; logic v0, v1;
  logic h0_sync, h0_push, h0_pop, h1_work;
  logic sync_retire, work_issue_h0, work_issue_h1, unit_free, edge_ok;
  int n_pop;
  always_comb begin
    h0 = iq[rd]; h1 = iq[(rd + 1) % IQ_DEPTH];
    v0 = cnt > 0; v1 = cnt > 1;
    h0_push = v0 && (h0.op == OP_PUSH); h0_pop = v0 && (h0.op == OP_POP); h0_sync = h0_push || h0_pop;
    h1_work = v1 && (h1.op != OP_PUSH) && (h1.op != OP_POP);
    unit_free = !u_busy && (!inflight_q || u_done);
    edge_ok = h0_push ? EDGE_VALID[edge_id(QID, h0.qual[2:0])] : (h0_pop ? EDGE_VALID[edge_id(h0.qual[2:0], QID)] : 1'b1);
    push_req = h0_push && edge_ok && (!inflight_q || u_done) && !fault;
    push_to = h0.qual[2:0];
    pop_req = h0_pop && edge_ok && !fault;
    pop_from = h0.qual[2:0];
    sync_retire = (push_req && push_ok) || (pop_req && pop_ok);
    work_issue_h0 = v0 && !h0_sync && unit_free && !fault;
    work_issue_h1 = sync_retire && h1_work && unit_free && !fault;
    issue = work_issue_h0 || work_issue_h1;
    u_ins = work_issue_h0 ? h0 : h1;
    n_pop = (sync_retire ? 1 : 0) + (work_issue_h0 ? 1 : 0) + (work_issue_h1 ? 1 : 0);
    full = (cnt >= IQ_DEPTH);
    empty = (cnt == 0);
    inflight = inflight_q; inflight_pc = inflight_pc_q; head_pc = h0.pc;
    progress = sync_retire || issue || u_done;
    tr_sync = sync_retire; tr_sync_op = h0.op; tr_sync_pc = h0.pc;
    tr_issue = issue; tr_issue_op = u_ins.op; tr_issue_pc = u_ins.pc;
    tr_done = u_done; tr_done_op = inflight_op_q; tr_done_pc = inflight_pc_q;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd <= 0; wr <= 0; cnt <= 0; inflight_q <= 0; inflight_pc_q <= 0; inflight_op_q <= 0;
      fault <= 0; fault_code <= F_NONE; fault_pc <= 0;
      for (int k = 0; k < IQ_DEPTH; k++) iq[k] <= '0;
    end else begin
      if (dq_valid && !full) begin iq[wr] <= dq_ins; wr <= (wr + 1) % IQ_DEPTH; end
      rd <= (rd + n_pop) % IQ_DEPTH;
      cnt <= cnt + ((dq_valid && !full) ? 1 : 0) - n_pop;
      if (issue) begin inflight_q <= 1; inflight_pc_q <= u_ins.pc; inflight_op_q <= u_ins.op; end
      else if (u_done) inflight_q <= 0;
      if (v0 && h0_sync && !edge_ok && !fault) begin fault <= 1; fault_code <= F_BAD_EDGE; fault_pc <= h0.pc; end
    end
  end
endmodule
