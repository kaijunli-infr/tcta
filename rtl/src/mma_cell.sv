// One MMA array cell (phase 8 K0.3, erratum E5): FIX_B_LEVELS signed multiply-accumulates into 48 bits, one per
// B level, sharing the A operand. The pre-shifted operands come from the row (a, with valid / first / last / slot
// travelling along) and the column (b); the cell registers them once (the systolic skew) and passes them on.
// p[l] holds level l's sum from the cycle `result_valid` pulses until the next block's first product lands (the
// block period is >= K, so that cycle is always available to the lane collector). NaN / inf operands contribute 0
// (their significand is 0 in the operand packet) and set the sticky flags.
module mac48 (
  input  logic                     clk,
  input  logic                     rst_n,
  input  logic signed [23:0]       a,        // FIX_AW
  input  logic signed [17:0]       b,        // FIX_BW
  input  logic                     v,
  input  logic                     first,
  input  logic                     last,
  output logic signed [47:0]       p,
  output logic                     pv
);
`ifdef TCTA_DSP48E2
  // DSP48E2: M = A * B (MREG), P = (first ? 0 : P) + M. OPMODE is registered inside (OPMODEREG = 1), so the
  // value presented with the operands applies to their product one cycle later, together with MREG.
  logic v_m, last_m;
  logic [8:0] opmode;
  always_comb opmode = {2'b00, first ? 3'b000 : 3'b010, 4'b0101};
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin v_m <= 1'b0; last_m <= 1'b0; pv <= 1'b0; end
    else begin v_m <= v; last_m <= last; pv <= v_m && last_m; end
  end
  DSP48E2 #(
    .AREG(0), .BREG(0), .ACASCREG(0), .BCASCREG(0), .ADREG(0), .DREG(0), .CREG(0), .INMODEREG(0), .MREG(1), .PREG(1),
    .OPMODEREG(1), .ALUMODEREG(0), .CARRYINREG(0), .CARRYINSELREG(0), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48"),
    .AMULTSEL("A"), .BMULTSEL("B"), .PREADDINSEL("A"), .USE_WIDEXOR("FALSE"), .A_INPUT("DIRECT"), .B_INPUT("DIRECT"),
    .RND(48'd0), .MASK(48'h3fffffffffff), .PATTERN(48'd0), .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN"),
    .USE_PATTERN_DETECT("NO_PATDET"), .AUTORESET_PATDET("NO_RESET"), .AUTORESET_PRIORITY("RESET"), .XORSIMD("XOR24_48_96")
  ) u_dsp (
    .CLK(clk), .A({{6{a[23]}}, a}), .B(b), .C(48'd0), .D(27'd0), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .ALUMODE(4'b0000), .INMODE(5'b00000), .OPMODE(opmode),
    .ACIN(30'd0), .BCIN(18'd0), .PCIN(48'd0), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0), .CEAD(1'b0), .CED(1'b0), .CEC(1'b0), .CEINMODE(1'b0),
    .CEM(v), .CEP(v_m), .CECTRL(1'b1), .CEALUMODE(1'b0), .CECARRYIN(1'b0),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0), .RSTCTRL(1'b0), .RSTALUMODE(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTINMODE(1'b0),
    .P(p), .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .CARRYOUT(), .MULTSIGNOUT(), .OVERFLOW(), .PATTERNBDETECT(),
    .PATTERNDETECT(), .PCOUT(), .UNDERFLOW(), .XOROUT());
`else
  logic signed [41:0] m;
  logic v_m, first_m, last_m;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m <= '0; v_m <= 1'b0; first_m <= 1'b0; last_m <= 1'b0; p <= '0; pv <= 1'b0;
    end else begin
      if (v) m <= a * b;
      v_m <= v; first_m <= first; last_m <= last;
      if (v_m) p <= (first_m ? 48'sd0 : p) + 48'(m);
      pv <= v_m && last_m;
    end
  end
`endif
endmodule

module mma_cell
  import tcta_pkg::*;
(
  input  logic     clk,
  input  logic     rst_n,
  input  fix_a_t   a_in,
  input  fix_b_t   b_in,
  input  logic     v_in,
  input  logic     first_in,
  input  logic     last_in,
  input  logic     slot_in,
  output fix_a_t   a_out,
  output fix_b_t   b_out,
  output logic     v_out,
  output logic     first_out,
  output logic     last_out,
  output logic     slot_out,
  output logic signed [FIX_BITS-1:0] p [FIX_B_LEVELS],
  output logic [2:0] flags,          // {nan, +inf, -inf} sticky over the block
  output logic     result_valid,
  output logic     result_slot
);
  fix_a_t a_q; fix_b_t b_q;
  logic v_q, first_q, last_q, slot_q;
  logic pv [FIX_B_LEVELS];
  for (genvar l = 0; l < FIX_B_LEVELS; l++) begin : g_lvl
    mac48 u_mac (.clk(clk), .rst_n(rst_n), .a(a_q.v), .b(b_q.v[l]), .v(v_q), .first(first_q), .last(last_q), .p(p[l]), .pv(pv[l]));
  end
  // flags: pipelined with the products (2 stages after the operand register) so that they are final with p
  logic is_nan, is_inf, sgn;
  logic [2:0] fl_m, fl_q;
  logic slot_m;
  always_comb begin
    is_nan = a_q.nan | b_q.nan | (a_q.inf & b_q.zero) | (b_q.inf & a_q.zero);
    is_inf = (a_q.inf | b_q.inf) & ~is_nan;
    sgn = a_q.sgn ^ b_q.sgn;
    fl_m = {is_nan, is_inf & ~sgn, is_inf & sgn};
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_q <= '0; b_q <= '0; v_q <= 1'b0; first_q <= 1'b0; last_q <= 1'b0; slot_q <= 1'b0;
      fl_q <= '0; flags <= '0; slot_m <= 1'b0; result_slot <= 1'b0;
    end else begin
      a_q <= a_in; b_q <= b_in; v_q <= v_in; first_q <= first_in; last_q <= last_in; slot_q <= slot_in;
      if (v_q) fl_q <= (first_q ? 3'b000 : fl_q) | fl_m;    // stage 2 (with M)
      if (v_q && last_q) begin flags <= (first_q ? 3'b000 : fl_q) | fl_m; result_slot <= slot_q; end   // stage 3 (with P)
    end
  end
  assign result_valid = pv[0];
  assign a_out = a_q; assign b_out = b_q; assign v_out = v_q; assign first_out = first_q; assign last_out = last_q; assign slot_out = slot_q;
endmodule
