// One token FIFO (ISA §8.1): tokens carry no payload, so the FIFO is an up/down counter with full and empty.
// PUSH is accepted only when not full; POP only when not empty; a simultaneous push and pop on a non-empty,
// non-full FIFO leaves the count unchanged. Formal target (rtl/formal/token_fifo.sby): count tracks
// pushes - pops, never exceeds DEPTH, never goes below 0, a push into a full FIFO is not accepted.
module token_fifo #(parameter int DEPTH = 8) (
  input  logic clk,
  input  logic rst_n,
  input  logic push,          // producer wants to PUSH this cycle
  input  logic pop,           // consumer wants to POP this cycle
  output logic push_ok,       // the push is accepted this cycle (not full)
  output logic pop_ok,        // the pop is accepted this cycle (not empty)
  output logic empty,
  output logic full,
  output logic [$clog2(DEPTH+1)-1:0] count
);
  always_comb begin
    empty = (count == 0);
    full  = (count == DEPTH);
    push_ok = push && !full;
    pop_ok  = pop && !empty;
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) count <= '0;
    else begin
      case ({push_ok, pop_ok})
        2'b10: count <= count + 1'b1;
        2'b01: count <= count - 1'b1;
        default: count <= count;
      endcase
    end
  end
`ifdef FORMAL
  // counting tokens: the count equals pushes - pops accepted so far and stays within [0, DEPTH]
  logic [31:0] n_push, n_pop;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin n_push <= 0; n_pop <= 0; end
    else begin n_push <= n_push + {31'd0, push_ok}; n_pop <= n_pop + {31'd0, pop_ok}; end
  end
  always_comb begin
    assert (count <= DEPTH);
    assert (32'(count) == n_push - n_pop);
    assert (!(full && push_ok));
    assert (!(empty && pop_ok));
    assert (!(push && !full) || push_ok);    // a push into a non-full FIFO is never dropped
    assert (!(pop && !empty) || pop_ok);     // a pop from a non-empty FIFO is never refused
  end
`endif
endmodule
