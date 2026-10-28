// The token network: one token_fifo per legal edge of ISA §8.1 (edge id = producer*8 + consumer), with a
// push port and a pop port per queue. A queue addresses an edge by (its own id, partner id); illegal edges are
// rejected at decode by the queues (BAD_EDGE), the network only instantiates the legal ones.
module token_net
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
(
  input  logic       clk,
  input  logic       rst_n,
  // producer side: queue q pushes on edge (q -> push_to[q])
  input  logic [4:0] push_req,
  input  logic [2:0] push_to [5],
  output logic [4:0] push_ok,
  // consumer side: queue q pops on edge (pop_from[q] -> q)
  input  logic [4:0] pop_req,
  input  logic [2:0] pop_from [5],
  output logic [4:0] pop_ok,
  output logic       all_empty,
  output logic [63:0] nonempty      // per edge id, for the TOKEN_LEAK status aux
);
  logic [63:0] f_push, f_pop, f_push_ok, f_pop_ok, f_empty, f_full;
  logic [$clog2(FIFO_DEPTH+1)-1:0] f_count [64];

  always_comb begin
    f_push = '0; f_pop = '0;
    for (int q = 0; q < 5; q++) begin
      if (push_req[q]) f_push[edge_id(3'(q), push_to[q])] = 1'b1;
      if (pop_req[q])  f_pop[edge_id(pop_from[q], 3'(q))] = 1'b1;
    end
    for (int q = 0; q < 5; q++) begin
      push_ok[q] = push_req[q] && f_push_ok[edge_id(3'(q), push_to[q])];
      pop_ok[q]  = pop_req[q]  && f_pop_ok[edge_id(pop_from[q], 3'(q))];
    end
    all_empty = &(f_empty | ~EDGE_VALID);
    nonempty = ~f_empty & EDGE_VALID;
  end

  for (genvar e = 0; e < 64; e++) begin : g_edge
    if (EDGE_VALID[e]) begin : g_fifo
      token_fifo #(.DEPTH(FIFO_DEPTH)) u_fifo (
        .clk(clk), .rst_n(rst_n), .push(f_push[e]), .pop(f_pop[e]),
        .push_ok(f_push_ok[e]), .pop_ok(f_pop_ok[e]), .empty(f_empty[e]), .full(f_full[e]), .count(f_count[e]));
    end else begin : g_none
      assign f_push_ok[e] = 1'b0; assign f_pop_ok[e] = 1'b0; assign f_empty[e] = 1'b1; assign f_full[e] = 1'b0; assign f_count[e] = '0;
    end
  end
endmodule
