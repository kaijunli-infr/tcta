// IEEE binary32 add/sub, round-to-nearest-even, subnormals, inf/NaN. Combinational.
// Bit-exact with NumPy float32 for finite operands; NaN results are the quiet NaN 0x7fc00000 (the aarch64
// default NaN), inputs that are NaN propagate as quiet NaN.
module f32_add (
  input  logic [31:0] a,
  input  logic [31:0] b,
  input  logic        sub,      // b is negated first
  output logic [31:0] y
);
  logic        sa, sb;
  logic [7:0]  ea, eb;
  logic [22:0] fa, fb;
  logic        a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
  logic [26:0] ma, mb;          // 1.23 significand + 3 guard bits (hidden bit at 26)
  logic [8:0]  eea, eeb;        // effective exponents (subnormal -> 1)
  logic        swap;
  logic [26:0] big, sml;
  logic [8:0]  ebig, esmall;
  logic        sbig, ssmall;
  logic [8:0]  diff;
  logic [26:0] sml_sh, lost;
  logic        sticky;
  logic [27:0] sum;             // one extra bit for carry
  logic        rs;              // result sign
  logic [4:0]  lz;
  logic [27:0] norm;
  logic signed [10:0] en;       // normalised exponent (may go <= 0: subnormal)
  logic [27:0] den;             // denormalised for subnormal result
  logic [23:0] mant_r;          // rounded 1.23 (bit 23 = hidden, may become 2 -> bit 24 via carry)
  logic [24:0] mant_rc;
  logic        g, r, st, inc;
  logic signed [10:0] e_out;
  logic [31:0] shift_n;

  always_comb begin
    den = '0; lost = '0; mant_r = '0; mant_rc = '0; g = 0; r = 0; st = 0; inc = 0; e_out = '0; shift_n = '0;
    sa = a[31]; ea = a[30:23]; fa = a[22:0];
    sb = b[31] ^ sub; eb = b[30:23]; fb = b[22:0];
    a_nan = (ea == 8'hff) && (fa != 0); b_nan = (eb == 8'hff) && (fb != 0);
    a_inf = (ea == 8'hff) && (fa == 0); b_inf = (eb == 8'hff) && (fb == 0);
    a_zero = (ea == 0) && (fa == 0); b_zero = (eb == 0) && (fb == 0);
    ma = {(ea != 0), fa, 3'b000};
    mb = {(eb != 0), fb, 3'b000};
    eea = (ea == 0) ? 9'd1 : {1'b0, ea};
    eeb = (eb == 0) ? 9'd1 : {1'b0, eb};
    // order operands by magnitude (exponent, then significand)
    swap = (eeb > eea) || ((eeb == eea) && (mb > ma));
    big   = swap ? mb : ma;   sml  = swap ? ma : mb;
    ebig  = swap ? eeb : eea; esmall = swap ? eea : eeb;
    sbig  = swap ? sb : sa;   ssmall = swap ? sa : sb;
    diff = ebig - esmall;
    // align the smaller operand; everything shifted past the guard bits folds into a sticky bit
    if (diff >= 9'd27) begin
      sml_sh = 27'd0;
      sticky = (sml != 0);
    end else begin
      sml_sh = sml >> diff;
      lost = sml & ((27'd1 << diff) - 27'd1);          // the bits shifted out below the guard bits
      sticky = (lost != 0);
    end
    sml_sh[0] = sml_sh[0] | sticky;
    if (sbig == ssmall) begin
      sum = {1'b0, big} + {1'b0, sml_sh};
    end else begin
      sum = {1'b0, big} - {1'b0, sml_sh};
    end
    rs = sbig;
    // normalise
    lz = 0;
    for (int i = 27; i >= 0; i--) if (sum[i]) begin lz = 5'(27 - i); break; end
    if (sum == 0) begin
      norm = 0; en = 0;
    end else begin
      norm = sum << lz;                 // hidden bit now at 27
      en = $signed({2'b00, ebig}) + 11'sd1 - $signed({6'b0, lz});
    end
    // subnormal result: shift right so the exponent becomes 1 (stored as 0)
    if (en <= 0) begin
      shift_n = 32'(1 - en);
      if (shift_n >= 28) begin den = {27'd0, (norm != 0)}; end
      else begin
        den = norm >> shift_n;
        den[0] = den[0] | ((norm << (28 - shift_n)) != 0);
      end
      e_out = 0;
    end else begin
      den = norm; e_out = en;
    end
    // round to nearest even: keep bits [27:4] as 1.23, guard = bit 3, round = bit 2, sticky = bits 1:0
    g = den[3]; r = den[2]; st = den[1] | den[0];
    inc = g & (r | st | den[4]);
    mant_rc = {1'b0, den[27:4]} + {24'd0, inc};
    if (mant_rc[24]) begin                   // carry out of the significand
      mant_r = mant_rc[24:1];
      e_out = e_out + 1;
    end else begin
      mant_r = mant_rc[23:0];
    end
    if (e_out == 0 && mant_r[23]) e_out = 1;   // subnormal rounded up into the smallest normal
    // assemble
    if (a_nan || b_nan || (a_inf && b_inf && (sa != sb))) begin
      y = 32'h7fc00000;
    end else if (a_inf) begin
      y = {sa, 8'hff, 23'd0};
    end else if (b_inf) begin
      y = {sb, 8'hff, 23'd0};
    end else if (sum == 0) begin
      // exact zero: +0 unless both operands were -0 (or both negative and cancelled to zero -> +0 in RNE)
      y = {(a_zero && b_zero) ? (sa & sb) : 1'b0, 31'd0};
    end else if (e_out >= 255) begin
      y = {rs, 8'hff, 23'd0};
    end else begin
      y = {rs, e_out[7:0], mant_r[22:0]};
    end
  end
endmodule
