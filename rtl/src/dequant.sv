// Dequantisation of one DRAM element to the f32 lane value (dtypes.dequant_row, ISA §4 / Appendix A).
// raw: the element's bits right-aligned (bf16 [15:0], f32/i32 [31:0], E4M3 [7:0], E2M1 [3:0]);
// scale8: this element's block scale code (UE8M0 for *.b32, E4M3 for f4.r16); scale_t: the per-tensor f32 scale.
// f8/f4 results are rounded to BF16 like the reference (`to_bf16(v * s)`); bf16/f32/i32 are returned as is.
module dequant_elem
  import tcta_isa_pkg::*;
(
  input  logic [7:0]  dtype,
  input  logic [31:0] raw,
  input  logic [7:0]  scale8,
  input  logic [31:0] scale_t,
  output logic [31:0] y
);
  // E4M3 -> f32 (dtypes._e4m3_table: e == 0 subnormal m * 2^-9, (15, 7) NaN, sign kept for -0)
  function automatic logic [31:0] e4m3_f32(input logic [7:0] b);
    logic s; logic [3:0] e; logic [2:0] m;
    s = b[7]; e = b[6:3]; m = b[2:0];
    if (e == 4'd15 && m == 3'd7) return 32'h7fc00000;
    if (e == 4'd0) begin
      case (m)
        3'd0: return {s, 31'd0};
        3'd1: return {s, 8'(127 - 9), 23'd0};
        3'd2: return {s, 8'(127 - 8), 23'd0};
        3'd3: return {s, 8'(127 - 8), 1'b1, 22'd0};
        3'd4: return {s, 8'(127 - 7), 23'd0};
        3'd5: return {s, 8'(127 - 7), 2'b01, 21'd0};
        3'd6: return {s, 8'(127 - 7), 2'b10, 21'd0};
        default: return {s, 8'(127 - 7), 2'b11, 21'd0};
      endcase
    end
    return {s, 8'(127 - 7) + {4'd0, e}, m, 20'd0};
  endfunction
  // E2M1 -> f32 (dtypes._E2M1: code 8 is the integer literal -0, i.e. +0.0)
  function automatic logic [31:0] e2m1_f32(input logic [3:0] n);
    logic [31:0] mag;
    case (n[2:0])
      3'd0: mag = 32'h00000000;
      3'd1: mag = 32'h3f000000;
      3'd2: mag = 32'h3f800000;
      3'd3: mag = 32'h3fc00000;
      3'd4: mag = 32'h40000000;
      3'd5: mag = 32'h40400000;
      3'd6: mag = 32'h40800000;
      default: mag = 32'h40c00000;
    endcase
    return (n[3] && n[2:0] != 3'd0) ? (mag | 32'h80000000) : mag;
  endfunction
  // UE8M0 -> f32: 2^(e-127); e == 255 reserved -> NaN; e == 0 is the subnormal 2^-127
  function automatic logic [31:0] ue8m0_f32(input logic [7:0] e);
    if (e == 8'hff) return 32'h7fc00000;
    if (e == 8'h00) return 32'h00400000;
    return {1'b0, e, 23'd0};
  endfunction

  logic [31:0] v, s, prod, i2f;
  logic [15:0] bf;
  always_comb begin
    case (dtype)
      DT_F8_T, DT_F8_B32: v = e4m3_f32(raw[7:0]);
      DT_F4_B32, DT_F4_R16: v = e2m1_f32(raw[3:0]);
      default: v = 32'd0;
    endcase
    case (dtype)
      DT_F8_T:  s = scale_t;
      DT_F8_B32, DT_F4_B32: s = ue8m0_f32(scale8);
      DT_F4_R16: s = e4m3_f32(scale8);
      default: s = 32'h3f800000;
    endcase
  end
  f32_mul u_mul (.a(v), .b(s), .y(prod));
  f32_to_bf16 u_bf (.x(prod), .y(bf));
  i32_to_f32 u_i2f (.x($signed(raw)), .y(i2f));
  always_comb begin
    case (dtype)
      DT_BF16: y = {raw[15:0], 16'd0};
      DT_F32:  y = raw;
      DT_I32:  y = i2f;
      default: y = {bf, 16'd0};
    endcase
  end
endmodule
