// Small binary32 helpers: compare / max (NumPy semantics: max propagates NaN; max(-0, +0) keeps the first
// operand like np.maximum's "x if x >= v"? -- np.maximum(x, v) returns the larger, and for equal values (±0)
// returns x), BF16 rounding (RNE, NaN preserved), f32 -> int32 round-to-nearest-even, int32 -> f32,
// f32 -> integer index (truncation toward zero, exact for integers).

module f32_max (
  input  logic [31:0] x,
  input  logic [31:0] v,
  output logic [31:0] y
);
  logic x_nan, v_nan, x_gt_v;
  logic signed [31:0] xi, vi;
  always_comb begin
    x_nan = (x[30:23] == 8'hff) && (x[22:0] != 0);
    v_nan = (v[30:23] == 8'hff) && (v[22:0] != 0);
    // sign-magnitude to two's complement order
    xi = x[31] ? -$signed({1'b0, x[30:0]}) : $signed({1'b0, x[30:0]});
    vi = v[31] ? -$signed({1'b0, v[30:0]}) : $signed({1'b0, v[30:0]});
    x_gt_v = (xi > vi);
    if (x_nan || v_nan) y = 32'h7fc00000;
    else if (x_gt_v) y = x;
    else if (xi == vi) y = (x[31] & v[31]) ? x : {1'b0, x[30:0]};   // equal: -0 vs +0 gives +0 (np.maximum)
    else y = v;
  end
endmodule

module f32_to_bf16 (
  input  logic [31:0] x,
  output logic [15:0] y          // the BF16 bits; as an f32 word this is {y, 16'd0}
);
  logic        is_nan;
  logic [31:0] rounded;
  always_comb begin
    is_nan = (x[30:23] == 8'hff) && (x[22:0] != 0);
    rounded = x + 32'h7fff + {31'd0, x[16]};
    y = is_nan ? 16'h7fc0 : rounded[31:16];      // dtypes.to_bf16 writes np.nan (0x7fc00000) for every NaN
  end
endmodule

// round to nearest even (np.rint) then convert to int32 (saturating; NaN -> INT_MIN like numpy on aarch64)
module f32_to_i32_rne (
  input  logic [31:0] x,
  output logic signed [31:0] y
);
  logic        s; logic [7:0] e; logic [23:0] m;
  logic signed [9:0] sh;         // exponent - 150: bits to shift the 24-bit significand left (>=0) / right (<0)
  logic [63:0] wide;
  logic [63:0] intpart;
  logic        g, st, inc;
  logic [31:0] shr;
  always_comb begin
    s = x[31]; e = x[30:23]; m = {(e != 0), x[22:0]};
    y = 0; sh = '0; wide = '0; intpart = '0; g = 0; st = 0; inc = 0; shr = '0;
    if (e == 8'hff) y = 32'sh80000000;
    else if (e < 8'd127 - 8'd24) y = 0;                          // |x| < 2^-24: rounds to 0
    else begin
      sh = $signed({2'b0, e}) - 10'sd150;                         // value = m * 2^sh
      if (sh >= 0) begin
        if (sh > 8) intpart = 64'hffffffffffffffff;              // overflow
        else intpart = {40'd0, m} << sh;
        inc = 0;
      end else begin
        shr = 32'(-sh);
        wide = {40'd0, m};
        intpart = wide >> shr;
        g = (shr <= 24) ? wide[shr - 1] : 1'b0;
        st = (shr >= 2) ? ((wide << (64 - (shr - 1))) != 0) : 1'b0;
        inc = g & (st | intpart[0]);
        intpart = intpart + {63'd0, inc};
      end
      if (intpart > 64'd2147483647) y = s ? 32'sh80000000 : 32'sh7fffffff;
      else y = s ? -$signed(intpart[31:0]) : $signed(intpart[31:0]);
    end
  end
endmodule

// int32 -> f32 (RNE), as numpy astype(float32)
module i32_to_f32 (
  input  logic signed [31:0] x,
  output logic [31:0] y
);
  logic        s; logic [31:0] mag; logic [4:0] lz; logic [31:0] norm;
  logic [24:0] mant_rc; logic [23:0] mant_r; logic g, st, inc; logic [7:0] e;
  always_comb begin
    s = x[31]; mag = s ? 32'(-x) : 32'(x);
    lz = 0;
    for (int i = 31; i >= 0; i--) if (mag[i]) begin lz = 5'(31 - i); break; end
    norm = mag << lz;                       // leading one at bit 31
    g = norm[7]; st = (norm[6:0] != 0);
    inc = g & (st | norm[8]);
    mant_rc = {1'b0, norm[31:8]} + {24'd0, inc};
    e = 8'd127 + 8'd31 - {3'b0, lz};
    if (mant_rc[24]) begin mant_r = mant_rc[24:1]; e = e + 1; end else mant_r = mant_rc[23:0];
    y = (mag == 0) ? 32'd0 : {s, e, mant_r[22:0]};
  end
endmodule

// f32 holding an integer value -> int (truncation; indices are exact integers)
module f32_to_int_trunc (
  input  logic [31:0] x,
  output logic signed [31:0] y
);
  logic s; logic [7:0] e; logic [23:0] m; logic signed [9:0] sh; logic [31:0] mag;
  always_comb begin
    s = x[31]; e = x[30:23]; m = {(e != 0), x[22:0]};
    sh = $signed({2'b0, e}) - 10'sd150;
    if (e == 8'hff) mag = 32'h7fffffff;
    else if (sh >= 0) mag = (sh > 8) ? 32'h7fffffff : (32'(m) << sh);
    else if (sh < -24) mag = 0;
    else mag = 32'(m) >> (-sh);
    y = s ? -$signed(mag) : $signed(mag);
  end
endmodule
