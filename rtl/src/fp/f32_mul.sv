// IEEE binary32 multiply, RNE, subnormals, inf/NaN. Combinational. Bit-exact with NumPy float32.
// MA / MB: number of significant fraction bits actually present in a / b (23 for f32, 7 for a BF16 value held
// in an f32 word); the synthesiser prunes the partial products of the structurally-zero low bits.
module f32_mul #(parameter int MA = 23, parameter int MB = 23) (
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [31:0] y
);
  logic        sa, sb, s;
  logic [7:0]  ea, eb;
  logic [22:0] fa, fb;
  logic        a_nan, b_nan, a_inf, b_inf, a_zero, b_zero;
  logic [23:0] ma, mb;
  logic [47:0] prod;
  logic signed [10:0] e;         // unbiased-then-rebiased exponent of the product's bit 47
  logic [5:0]  lz;
  logic [47:0] norm;
  logic signed [10:0] en;
  logic [47:0] den;
  logic [31:0] shift_n;
  logic [24:0] mant_rc;
  logic [23:0] mant_r;
  logic        g, st, inc;
  logic signed [10:0] e_out;

  always_comb begin
    den = '0; mant_r = '0; mant_rc = '0; g = 0; st = 0; inc = 0; e_out = '0; shift_n = '0;
    sa = a[31]; ea = a[30:23]; fa = a[22:0];
    sb = b[31]; eb = b[30:23]; fb = b[22:0];
    s = sa ^ sb;
    a_nan = (ea == 8'hff) && (fa != 0); b_nan = (eb == 8'hff) && (fb != 0);
    a_inf = (ea == 8'hff) && (fa == 0); b_inf = (eb == 8'hff) && (fb == 0);
    a_zero = (ea == 0) && (fa == 0); b_zero = (eb == 0) && (fb == 0);
    ma = {(ea != 0), fa & ((23'h7fffff >> (23 - MA)) << (23 - MA))};
    mb = {(eb != 0), fb & ((23'h7fffff >> (23 - MB)) << (23 - MB))};
    prod = ma * mb;                                   // 2.46 fixed point: bit 47 = weight 2
    // exponent of bit 47 of prod as a biased f32 exponent: ea' + eb' - 127 + 1
    e = $signed({3'b0, (ea == 0) ? 8'd1 : ea}) + $signed({3'b0, (eb == 0) ? 8'd1 : eb}) - 11'sd127 + 11'sd1;
    lz = 0;
    for (int i = 47; i >= 0; i--) if (prod[i]) begin lz = 6'(47 - i); break; end
    norm = prod << lz;                                // leading one at bit 47
    en = e - $signed({5'b0, lz});
    if (en <= 0) begin
      shift_n = 32'(1 - en);
      if (shift_n >= 48) den = {47'd0, (norm != 0)};
      else begin
        den = norm >> shift_n;
        den[0] = den[0] | ((norm << (48 - shift_n)) != 0);
      end
      e_out = 0;
    end else begin
      den = norm; e_out = en;
    end
    // 1.23 from bits [47:24]; guard bit 23; sticky bits [22:0]
    g = den[23]; st = (den[22:0] != 0);
    inc = g & (st | den[24]);
    mant_rc = {1'b0, den[47:24]} + {24'd0, inc};
    if (mant_rc[24]) begin mant_r = mant_rc[24:1]; e_out = e_out + 1; end
    else mant_r = mant_rc[23:0];
    if (e_out == 0 && mant_r[23]) e_out = 1;
    if (a_nan || b_nan || (a_inf && b_zero) || (b_inf && a_zero)) y = 32'h7fc00000;
    else if (a_inf || b_inf) y = {s, 8'hff, 23'd0};
    else if (prod == 0) y = {s, 31'd0};
    else if (e_out >= 255) y = {s, 8'hff, 23'd0};
    else y = {s, e_out[7:0], mant_r[22:0]};
  end
endmodule
