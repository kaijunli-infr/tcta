// One output-stationary processing element (the ACC_F32 = 1 array of mma_unit; erratum E5 option A): a (BF16) flows right, b (BF16) flows down, the control bits
// (valid, first, last) travel with a. Each cycle with valid data: acc = fl(acc_or_zero + fl(a * b)) -- the
// product is rounded to binary32 first and the sum starts from +0 (erratum E1: fl(0 + p0) turns a -0 product
// into +0 exactly like the simulator's `S = zeros; S += a*b`). On the last product the sum is latched into
// `result` and `result_valid` pulses.
module mma_pe (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [15:0] a_in,
  input  logic [15:0] b_in,
  input  logic        v_in,
  input  logic        first_in,
  input  logic        last_in,
  input  logic        slot_in,
  output logic [15:0] a_out,
  output logic [15:0] b_out,
  output logic        v_out,
  output logic        first_out,
  output logic        last_out,
  output logic        slot_out,
  output logic [31:0] result,
  output logic        result_valid,
  output logic        result_slot
);
  logic [15:0] a_q, b_q;
  logic        v_q, first_q, last_q, slot_q;
  logic [31:0] acc, prod, sum, base;

  f32_mul #(.MA(7), .MB(7)) u_mul (.a({a_q, 16'd0}), .b({b_q, 16'd0}), .y(prod));
  always_comb base = first_q ? 32'd0 : acc;
  f32_add u_add (.a(base), .b(prod), .sub(1'b0), .y(sum));

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_q <= '0; b_q <= '0; v_q <= 1'b0; first_q <= 1'b0; last_q <= 1'b0; slot_q <= 1'b0;
      acc <= '0; result <= '0; result_valid <= 1'b0; result_slot <= 1'b0;
    end else begin
      a_q <= a_in; b_q <= b_in; v_q <= v_in; first_q <= first_in; last_q <= last_in; slot_q <= slot_in;
      if (v_q) acc <= sum;
      result_valid <= v_q && last_q;
      if (v_q && last_q) begin result <= sum; result_slot <= slot_q; end
    end
  end
  assign a_out = a_q; assign b_out = b_q; assign v_out = v_q; assign first_out = first_q; assign last_out = last_q; assign slot_out = slot_q;
endmodule
