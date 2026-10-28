// One scratchpad: BANKS element-wide banks (bank = column mod BANKS), each a dual-port RAM, behind an arbiter
// that grants at most two of the N_CLIENTS requests addressed to this scratchpad per cycle and at most one write
// per cycle (formal property: never two writers in one cycle, the highest-priority requester is always granted).
// A request is one row slice: a per-bank address, a bank-enable mask and, for writes, per-bank data. Reads return
// the cycle after the grant. DW = 16 banks (INP/WGT) hold the BF16 half of the 32-bit lane word ([31:16]).
module scratchpad
  import tcta_pkg::*;
#(
  parameter int WORDS = 1024,
  parameter int DW = 32,
  parameter logic [3:0] SP_ID = 4'd0
) (
  input  logic     clk,
  input  logic     rst_n,
  input  sp_req_t  req    [N_CLIENTS],
  output logic     grant  [N_CLIENTS],
  output logic     rvalid [N_CLIENTS],
  output slice_t   rdata  [N_CLIENTS]
);
  // ---- arbiter: client 0 has the highest priority ----
  logic [N_CLIENTS-1:0] want, gA, gB;
  logic                 wA, wB;
  always_comb begin
    for (int i = 0; i < N_CLIENTS; i++) want[i] = req[i].valid && (req[i].sp == SP_ID);
    gA = '0; gB = '0; wA = 1'b0; wB = 1'b0;
    for (int i = 0; i < N_CLIENTS; i++) begin
      if (want[i] && gA == '0) begin gA[i] = 1'b1; wA = req[i].we; end
    end
    for (int i = 0; i < N_CLIENTS; i++) begin
      if (want[i] && !gA[i] && gB == '0 && !(req[i].we && wA)) begin gB[i] = 1'b1; wB = req[i].we; end
    end
    for (int i = 0; i < N_CLIENTS; i++) grant[i] = gA[i] | gB[i];
  end

  // ---- mux the granted requests onto the two ports ----
  logic              pA_we, pB_we, pA_valid, pB_valid;
  baddr_t            pA_addr, pB_addr;
  logic [BANKS-1:0]  pA_ben, pB_ben;
  slice_t            pA_wdata, pB_wdata;
  slice_t            pA_rdata, pB_rdata;
  always_comb begin
    pA_valid = |gA; pB_valid = |gB; pA_we = 1'b0; pB_we = 1'b0; pA_ben = '0; pB_ben = '0;
    pA_addr = '0; pB_addr = '0; pA_wdata = '0; pB_wdata = '0;
    for (int i = 0; i < N_CLIENTS; i++) begin
      if (gA[i]) begin pA_we = req[i].we; pA_ben = req[i].ben; pA_addr = req[i].addr; pA_wdata = req[i].wdata; end
      if (gB[i]) begin pB_we = req[i].we; pB_ben = req[i].ben; pB_addr = req[i].addr; pB_wdata = req[i].wdata; end
    end
  end

  // ---- banks ----
  logic [DW-1:0] rA [BANKS], rB [BANKS];
  for (genvar b = 0; b < BANKS; b++) begin : g_bank
    logic [DW-1:0] mem [WORDS];
    always_ff @(posedge clk) begin
      rA[b] <= mem[pA_addr[b]];
      rB[b] <= mem[pB_addr[b]];
      if (pA_valid && pA_we && pA_ben[b]) mem[pA_addr[b]] <= pA_wdata[b][31 -: DW];
      if (pB_valid && pB_we && pB_ben[b]) mem[pB_addr[b]] <= pB_wdata[b][31 -: DW];
    end
  end
  always_comb begin
    for (int b = 0; b < BANKS; b++) begin
      pA_rdata[b] = '0; pB_rdata[b] = '0;
      pA_rdata[b][31 -: DW] = rA[b];
      pB_rdata[b][31 -: DW] = rB[b];
    end
  end

  // ---- responses: the cycle after the grant, routed back to the granted (reading) client ----
  logic [N_CLIENTS-1:0] gA_q, gB_q;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin gA_q <= '0; gB_q <= '0; end
    else begin gA_q <= gA & ~{N_CLIENTS{wA}}; gB_q <= gB & ~{N_CLIENTS{wB}}; end
  end
  always_comb begin
    for (int i = 0; i < N_CLIENTS; i++) begin
      rvalid[i] = gA_q[i] | gB_q[i];
      rdata[i]  = gA_q[i] ? pA_rdata : pB_rdata;
    end
  end

`ifdef FORMAL
  always_comb begin
    assert (!(pA_valid && pA_we && pB_valid && pB_we));          // one writer per cycle
    assert ($countones(gA) <= 1 && $countones(gB) <= 1);
    assert ((gA & gB) == '0);                                      // a client gets at most one port
    for (int i = 0; i < N_CLIENTS; i++) assert (!grant[i] || want[i]);
    for (int i = 0; i < N_CLIENTS; i++) begin                      // the highest-priority requester is granted
      logic higher;
      higher = 1'b0;
      for (int j = 0; j < i; j++) higher |= want[j];
      assert (!(want[i] && !higher) || grant[i]);
    end
    // a second reader never starves while a port is free: two pure reads are both granted
    assert (!(want[0] && want[1] && !req[0].we && !req[1].we) || (grant[0] && grant[1]));
  end
`endif
endmodule
