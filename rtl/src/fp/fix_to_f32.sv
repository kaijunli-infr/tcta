// Fixed-point window -> binary32 (phase 8 K0.3, erratum E5): acc * 2^w, |acc| < 2^(FIX_ACC_W-1), rounded to nearest
// even with subnormals and overflow to inf; the sticky flags ({nan, +inf, -inf}) decide the result like IEEE would:
// nan or both infinities -> the canonical quiet NaN, else the signed infinity. Mirrors machine._fixed48_to_f32.
module fix_to_f32
  import tcta_pkg::*;
(
  input  logic signed [FIX_ACC_W-1:0] acc,
  input  logic signed [FIX_W_W-1:0]   w,
  input  logic [2:0]                  flags,
  output logic [31:0]                 y
);
  localparam int AW = FIX_ACC_W;
  logic neg;
  logic [AW-1:0] mag, norm;
  int lz;
  logic signed [11:0] e_lead, biased;
  logic [23:0] q; logic g, s;
  logic [24:0] t, tsh, drop_mask;
  int d;
  logic [24:0] qr;
  logic inc;
  logic [8:0] E;
  always_comb begin
    neg = acc[AW-1];
    mag = neg ? AW'(-acc) : AW'(acc);
    lz = AW;
    for (int i = 0; i < AW; i++) if (mag[i]) lz = AW - 1 - i;        // position of the leading one -> leading zeros
    norm = mag << lz[$clog2(AW+1)-1:0];
    e_lead = 12'(w) + 12'(AW - 1) - 12'(lz);
    biased = e_lead + 12'sd127;
    // 24-bit significand, guard, sticky at the normal position
    q = norm[AW-1 -: 24]; g = norm[AW-25]; s = |norm[AW-26:0];
    E = 9'(biased);
    // subnormal: shift the significand right by 1 - biased, the dropped bits join the sticky
    d = int'(12'sd1 - biased); if (d > 26) d = 26; if (d < 0) d = 0;
    t = {q, g};
    tsh = t >> d;
    drop_mask = (25'd1 << d) - 25'd1;
    if (biased < 12'sd1) begin
      s = s | ((t & drop_mask) != 25'd0);
      q = tsh[24:1]; g = tsh[0];
      E = 9'd0;
    end
    inc = g & (s | q[0]);
    qr = 25'(q) + 25'(inc);
    if (qr[24]) begin qr = 25'd1 << 23; E = E + 9'd1; end             // carry out of the significand
    else if (qr[23] && E == 9'd0) E = 9'd1;                           // subnormal rounded up to the minimum normal
    if (E >= 9'd255) y = {neg, 8'hff, 23'd0};
    else y = {neg, E[7:0], qr[22:0]};
    if (mag == '0) y = 32'd0;
    if (flags[1] && !flags[2]) y = 32'h7f800000;
    if (flags[2] && !flags[1]) y = 32'hff800000;
    if (flags[0] || (flags[1] && flags[2])) y = 32'h7fc00000;
  end
endmodule
