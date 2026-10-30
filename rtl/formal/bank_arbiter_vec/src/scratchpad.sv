// One scratchpad: BANKS element-wide banks (bank = column mod BANKS; the VEC pad's 2-D slots: bank = row mod BANKS,
// erratum E2, see the serial path below), each a dual-port RAM, behind an arbiter
// that grants at most two of the N_CLIENTS requests addressed to this scratchpad per cycle and at most one write
// per cycle (formal property: never two writers in one cycle, the highest-priority requester is always granted).
// A request is one row slice: a per-bank address, a bank-enable mask and, for writes, per-bank data. Reads return
// the cycle after the grant. DW = 16 banks (INP/WGT) hold the BF16 half of the 32-bit lane word ([31:16]).
// READERS / WRITERS (phase 8 L2) are the clients this pad serves (tcta_pkg::SP_READERS / SP_WRITERS, from the ISA's
// access table): the arbiter covers their union, the write-data muxes only the writers, the response path only the
// readers. Every pad took all seven for both before (41 k LUTs for the five pads, 16 banks x 2 ports x 7 sources).
module scratchpad
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
#(
  parameter int WORDS = 1024,
  parameter int DW = 32,
  parameter logic [3:0] SP_ID = 4'd0,
  parameter logic [N_CLIENTS-1:0] READERS = SP_READERS[SP_ID],
  parameter logic [N_CLIENTS-1:0] WRITERS = SP_WRITERS[SP_ID]
) (
  input  logic     clk,
  input  logic     rst_n,
  input  sp_req_t  req    [N_CLIENTS],
  output logic     grant  [N_CLIENTS],
  output logic     rvalid [N_CLIENTS],
  output slice_t   rdata  [N_CLIENTS]
);
  localparam logic [N_CLIENTS-1:0] CLIENTS = READERS | WRITERS;
  // ---- phase 8 K2 (erratum E2): the VEC pad's 2-D slots are banked by row ----
  // A column-slice request (req.cs) is one word in every bank like any other request. A row-slice request into that
  // range (every DMA access and the VEC unit's mixed-shape operands) finds its elements in one bank, bank row % BANKS,
  // at one word per column: port A serves it element by element over its enabled lanes (a read with ben = 0 takes all
  // lanes), grants it with the last element and returns the assembled slice the cycle after; port B keeps serving the
  // other clients meanwhile (never a second writer).
  localparam bit ROW_BANKED = (SP_ID == SP_VEC);
  localparam int CLW = $clog2(N_CLIENTS);
  function automatic logic is_serial(input sp_req_t r);
    return ROW_BANKED && !r.cs && (int'(r.addr) < VEC_2D_WORDS);
  endfunction
  function automatic logic [SP_AW-1:0] ser_word(input logic [SP_AW-1:0] addr, input int j);   // element j (column cg*BANKS + j)
    int a, slot, row, cg;
    a = int'(addr); slot = a / (T_M * VEC_CG); row = (a % (T_M * VEC_CG)) / VEC_CG; cg = a % VEC_CG;
    return SP_AW'(slot * T_M * VEC_CG + (cg * BANKS + j) * VEC_RG + row / BANKS);
  endfunction
  function automatic int ser_bank(input logic [SP_AW-1:0] addr);
    return ((int'(addr) % (T_M * VEC_CG)) / VEC_CG) % BANKS;
  endfunction

  // ---- arbiter: client 0 has the highest priority; a serial burst keeps port A ----
  logic [N_CLIENTS-1:0] want, serial, selA, gA, gB;
  logic                 wA, wB;
  logic                 ser_busy, ser_act, ser_now, ser_lastel;
  logic [CLW-1:0]       ser_cl;
  logic [SP_AW-1:0]     ser_addr;                  // the burst's request: a client that presents a different one
  logic                 ser_we;                    // (dma_load's index read preempts its writer) aborts the burst,
  logic [BANKS-1:0]     ser_rem, ser_lanes_now;    // which restarts when the request comes back (writes are idempotent)
  int                   ser_j, ser_b, selA_i;
  logic [SP_AW-1:0]     aA;                        // port A's request as selected (before the serial element substitution)
  logic [BANKS-1:0]     benA;
  always_comb begin
    for (int i = 0; i < N_CLIENTS; i++) begin
      want[i] = (req[i].we ? WRITERS[i] : READERS[i]) && req[i].valid && (req[i].sp == SP_ID);
      serial[i] = want[i] && is_serial(req[i]);
    end
    ser_act = ser_busy && (int'(ser_cl) < N_CLIENTS) && serial[ser_cl] && (req[ser_cl].addr == ser_addr) && (req[ser_cl].we == ser_we);
    selA = '0; selA_i = 0;
    if (ser_act) begin selA[ser_cl] = 1'b1; selA_i = int'(ser_cl); end             // the burst keeps port A
    else for (int i = N_CLIENTS - 1; i >= 0; i--) if (want[i]) begin selA = '0; selA[i] = 1'b1; selA_i = i; end
    aA = '0; benA = '0; wA = 1'b0;
    for (int i = 0; i < N_CLIENTS; i++) if (CLIENTS[i] && selA[i]) begin aA = req[i].addr; benA = req[i].ben; wA = req[i].we; end
    ser_now = (selA != '0) && (ser_act || (selA & serial) != '0);
    ser_lanes_now = ser_act ? ser_rem : (wA ? benA : ((benA == '0) ? {BANKS{1'b1}} : benA));
    ser_j = 0;
    for (int b = BANKS - 1; b >= 0; b--) if (ser_lanes_now[b]) ser_j = b;
    ser_b = ser_bank(aA);
    ser_lastel = ser_now && ((ser_lanes_now & ~(BANKS'(1) << ser_j)) == '0);
    gA = ser_now ? (ser_lastel ? selA : '0) : selA;
    gB = '0; wB = 1'b0;
    for (int i = 0; i < N_CLIENTS; i++) begin
      if (want[i] && !selA[i] && !serial[i] && gB == '0 && !(req[i].we && wA)) begin
        gB[i] = 1'b1; wB = req[i].we;
      end
    end
    for (int i = 0; i < N_CLIENTS; i++) grant[i] = gA[i] | gB[i];
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin ser_busy <= 1'b0; ser_cl <= '0; ser_rem <= '0; ser_addr <= '0; ser_we <= 1'b0; end
    else if (ser_now && !ser_lastel) begin
      ser_busy <= 1'b1; ser_cl <= CLW'(selA_i); ser_rem <= ser_lanes_now & ~(BANKS'(1) << ser_j);
      ser_addr <= aA; ser_we <= wA;
    end else ser_busy <= 1'b0;
  end

  // ---- mux the granted requests onto the two ports ----
  logic              pA_we, pB_we, pA_valid, pB_valid;
  logic [SP_AW-1:0]  pA_addr, pB_addr;
  logic [BANKS-1:0]  pA_ben, pB_ben;
  slice_t            pA_wdata, pB_wdata;
  slice_t            pA_rdata, pB_rdata;
  logic [31:0]       ser_el;
  always_comb begin
    pA_valid = |selA; pB_valid = |gB; pA_we = 1'b0; pB_we = 1'b0; pA_ben = '0; pB_ben = '0;
    pA_addr = '0; pB_addr = '0; pA_wdata = '0; pB_wdata = '0;
    for (int i = 0; i < N_CLIENTS; i++) if (CLIENTS[i]) begin
      if (selA[i]) pA_addr = req[i].addr;
      if (gB[i]) pB_addr = req[i].addr;
      if (WRITERS[i] && selA[i]) begin pA_we = req[i].we; pA_ben = req[i].ben; pA_wdata = req[i].wdata; end
      if (WRITERS[i] && gB[i]) begin pB_we = req[i].we; pB_ben = req[i].ben; pB_wdata = req[i].wdata; end
    end
    ser_el = pA_wdata[ser_j];                               // the element's data (the writers' mux above; readers carry none)
    if (ser_now) begin                                      // one element: its word, its bank, its lane's data
      pA_addr = ser_word(aA, ser_j);
      pA_ben = pA_we ? (BANKS'(1) << ser_b) : '0;
      for (int b = 0; b < BANKS; b++) pA_wdata[b] = ser_el;
    end
  end

  // ---- banks ----
  // Written for block-RAM inference (Yosys synth_xilinx, phase 7 J7): one process per port, so the two write
  // ports have no priority relation (the arbiter never grants two writes in a cycle, formal property below), a
  // DW-wide read register per bank and port (the RAM's output register), and no read-during-write check across
  // ports (no_rw_check: a read and a write of the same word in one cycle on different ports is a stale read in
  // this design, ISA §8.4). The single-process form mapped to 7680 LUT-RAM cells and 34 k LUTs for the ACC pad.
  logic [DW-1:0] rA [BANKS], rB [BANKS];
  for (genvar b = 0; b < BANKS; b++) begin : g_bank
    /* verilator lint_off MULTIDRIVEN */
    (* ram_style = "block", no_rw_check *) logic [DW-1:0] mem [WORDS];
    /* verilator lint_on MULTIDRIVEN */
    logic [DW-1:0] qA, qB;
    always @(posedge clk) begin
      qA <= mem[pA_addr];
      if (pA_valid && pA_we && pA_ben[b]) mem[pA_addr] <= pA_wdata[b][31 -: DW];
    end
    always @(posedge clk) begin
      qB <= mem[pB_addr];
      if (pB_valid && pB_we && pB_ben[b]) mem[pB_addr] <= pB_wdata[b][31 -: DW];
    end
    assign rA[b] = qA;
    assign rB[b] = qB;
  end
  always_comb begin
    for (int b = 0; b < BANKS; b++) begin
      pA_rdata[b] = '0; pB_rdata[b] = '0;
      pA_rdata[b][31 -: DW] = rA[b];
      pB_rdata[b][31 -: DW] = rB[b];
    end
  end

  // ---- responses: the cycle after the grant, routed back to the granted (reading) client ----
  // A serial read assembles its lanes in ser_rd (element j of the cycle before lands in bank ser_b's register) and
  // returns them with the last element's data the cycle after its grant.
  logic [N_CLIENTS-1:0] gA_q, gB_q;
  logic ser_q, ser_last_q; int ser_j_q, ser_b_q;
  slice_t ser_rd, ser_slice;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin gA_q <= '0; gB_q <= '0; ser_q <= 1'b0; ser_last_q <= 1'b0; ser_j_q <= 0; ser_b_q <= 0; end
    else begin
      gA_q <= gA & ~{N_CLIENTS{wA}}; gB_q <= gB & ~{N_CLIENTS{wB}};
      ser_q <= ser_now && !pA_we; ser_last_q <= ser_lastel && !pA_we; ser_j_q <= ser_j; ser_b_q <= ser_b;
    end
  end
  always_ff @(posedge clk) if (ser_q) ser_rd[ser_j_q] <= pA_rdata[ser_b_q];
  always_comb begin
    ser_slice = ser_rd;
    ser_slice[ser_j_q] = pA_rdata[ser_b_q];
    for (int i = 0; i < N_CLIENTS; i++) begin
      rvalid[i] = READERS[i] && (gA_q[i] | gB_q[i]);
      rdata[i]  = !READERS[i] ? '0 : gA_q[i] ? (ser_last_q ? ser_slice : pA_rdata) : pB_rdata;
    end
  end

`ifdef FORMAL
  always_comb begin
    assert (!(pA_valid && pA_we && pB_valid && pB_we));          // one writer per cycle
    assert ($countones(gA) <= 1 && $countones(gB) <= 1);
    assert ((gA & gB) == '0);                                      // a client gets at most one port
    for (int i = 0; i < N_CLIENTS; i++) assert (!grant[i] || want[i]);
    for (int i = 0; i < N_CLIENTS; i++) begin                      // the highest-priority requester holds port A
      logic higher;                                                // (and is granted unless its row slice is serial)
      higher = 1'b0;
      for (int j = 0; j < i; j++) higher |= want[j];
      assert (ser_act || !(want[i] && !higher) || selA[i]);
      assert (ser_act || !(want[i] && !higher) || serial[i] || grant[i]);
    end
    // a second reader never starves while a port is free: two pure reads are both granted
    assert (ser_act || serial[0] || serial[1] || !(want[0] && want[1] && !req[0].we && !req[1].we) || (grant[0] && grant[1]));
    assert (!ser_act || (want[ser_cl] && selA[ser_cl]));          // a live burst belongs to a requesting client
    // a serial burst: one client on port A, granted only with its last element, never a write on port B meanwhile
    assert (!ser_now || $countones(selA) == 1);
    assert (!(ser_now && !ser_lastel) || gA == '0);
    assert (!(ser_now && pA_we) || !(pB_valid && pB_we));
  end
`endif
endmodule
