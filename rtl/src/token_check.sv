// Debug twin of the simulator's tile-level token checks (ISA §8.4): per slot the last writer queue and write
// time and the last read time per queue; per edge the last POP time. A read of a slot whose write is newer than
// the last POP on (writer -> reader) -- or of a slot never written -- is UNTOKENED_READ; a write over a slot
// read by another queue more recently than the last POP on (reader -> writer) is UNTOKENED_WRITE. The ISA lets
// hardware skip this (an untokened read returns stale data); tcta_core instantiates it under TOKEN_CHECK so the
// harness reproduces the simulator's negative test. Events arrive per scratchpad access (slot granularity) and,
// like the simulator's `_note_read` at instruction start, once per operand tile when a unit takes an instruction
// (reading a tile's metadata is a read of the tile; a zero-shaped operand is otherwise never touched).
module token_check
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
#(
  parameter int N_EV = N_CLIENTS
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        ev_valid [N_EV],
  input  logic        ev_we    [N_EV],
  input  logic [7:0]  ev_slot  [N_EV],
  input  logic [2:0]  ev_q     [N_EV],
  input  logic [31:0] ev_pc    [N_EV],
  input  logic [4:0]  pop_valid,
  input  logic [2:0]  pop_from [5],
  output logic        fault,
  output fault_t      fault_code,
  output logic [2:0]  fault_q,
  output logic [31:0] fault_pc
);
  logic [TC_TIME_W-1:0] now;
  logic [2:0]           writer   [N_SLOTS];
  logic                 written  [N_SLOTS];
  logic [TC_TIME_W-1:0] write_t  [N_SLOTS];
  logic                 rd_seen  [N_SLOTS][5];
  logic [TC_TIME_W-1:0] rd_t     [N_SLOTS][5];
  logic [TC_TIME_W-1:0] pop_t    [64];
  logic                 pop_seen [64];

  logic f_hit; fault_t f_code; logic [2:0] f_q; logic [31:0] f_pc;
  always_comb begin
    f_hit = 1'b0; f_code = F_NONE; f_q = '0; f_pc = '0;
    for (int u = 0; u < N_EV; u++) begin
      int s, e;
      s = int'(ev_slot[u]); e = 0;
      if (ev_valid[u] && !ev_we[u] && !f_hit) begin
        if (!written[s]) begin f_hit = 1; f_code = F_UNTOKENED_READ; f_q = ev_q[u]; f_pc = ev_pc[u]; end
        else if (writer[s] != ev_q[u]) begin
          e = edge_id(writer[s], ev_q[u]);
          if (!EDGE_VALID[e] || !pop_seen[e] || pop_t[e] < write_t[s]) begin
            f_hit = 1; f_code = F_UNTOKENED_READ; f_q = ev_q[u]; f_pc = ev_pc[u];
          end
        end
      end
      if (ev_valid[u] && ev_we[u] && !f_hit) begin
        for (int rq = 0; rq < 5; rq++) begin
          if (rd_seen[s][rq] && 3'(rq) != ev_q[u] && !f_hit) begin
            e = edge_id(3'(rq), ev_q[u]);
            if (!EDGE_VALID[e] || !pop_seen[e] || pop_t[e] < rd_t[s][rq]) begin
              f_hit = 1; f_code = F_UNTOKENED_WRITE; f_q = ev_q[u]; f_pc = ev_pc[u];
            end
          end
        end
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      now <= '0; fault <= 1'b0; fault_code <= F_NONE; fault_q <= '0; fault_pc <= '0;
      for (int s = 0; s < N_SLOTS; s++) begin
        written[s] <= 1'b0; writer[s] <= '0; write_t[s] <= '0;
        for (int q = 0; q < 5; q++) begin rd_seen[s][q] <= 1'b0; rd_t[s][q] <= '0; end
      end
      for (int e = 0; e < 64; e++) begin pop_seen[e] <= 1'b0; pop_t[e] <= '0; end
    end else begin
      now <= now + 1;
      for (int q = 0; q < 5; q++) if (pop_valid[q]) begin
        pop_seen[edge_id(pop_from[q], 3'(q))] <= 1'b1;
        pop_t[edge_id(pop_from[q], 3'(q))] <= now;
      end
      for (int u = 0; u < N_EV; u++) begin
        if (ev_valid[u] && !ev_we[u]) begin
          rd_seen[ev_slot[u]][ev_q[u]] <= 1'b1;
          rd_t[ev_slot[u]][ev_q[u]] <= now;
        end
      end
      for (int u = 0; u < N_EV; u++) begin
        if (ev_valid[u] && ev_we[u]) begin
          written[ev_slot[u]] <= 1'b1; writer[ev_slot[u]] <= ev_q[u]; write_t[ev_slot[u]] <= now;
          for (int q = 0; q < 5; q++) if (3'(q) != ev_q[u]) rd_seen[ev_slot[u]][q] <= 1'b0;
        end
      end
      if (f_hit && !fault) begin fault <= 1'b1; fault_code <= f_code; fault_q <= f_q; fault_pc <= f_pc; end
    end
  end
endmodule
