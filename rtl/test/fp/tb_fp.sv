// Unit test of the FP library against NumPy: reads pairs from fp_in.hex, writes results to fp_out.hex.
module tb_fp (input logic clk);
  localparam int N = 200000;
  logic [63:0] vin [N];
  logic [31:0] a, b, y_add, y_sub, y_mul, y_mul7, y_max, y_i2f; logic [15:0] y_bf; logic signed [31:0] y_i32, y_trunc;
  int i = 0, fo, n;
  f32_add u_add (.a(a), .b(b), .sub(1'b0), .y(y_add));
  f32_add u_sub (.a(a), .b(b), .sub(1'b1), .y(y_sub));
  f32_mul u_mul (.a(a), .b(b), .y(y_mul));
  f32_mul #(.MA(7), .MB(7)) u_mul7 (.a({a[31:16], 16'd0}), .b({b[31:16], 16'd0}), .y(y_mul7));
  f32_max u_max (.x(a), .v(b), .y(y_max));
  f32_to_bf16 u_bf (.x(a), .y(y_bf));
  f32_to_i32_rne u_i32 (.x(a), .y(y_i32));
  i32_to_f32 u_i2f (.x($signed(a)), .y(y_i2f));
  f32_to_int_trunc u_tr (.x(a), .y(y_trunc));
  initial begin
    if (!$value$plusargs("N=%d", n)) n = N;
    $readmemh("fp_in.hex", vin);
    fo = $fopen("fp_out.hex", "w");
  end
  always_comb begin a = vin[i][63:32]; b = vin[i][31:0]; end
  always_ff @(posedge clk) begin
    $fwrite(fo, "%08x %08x %08x %08x %08x %04x %08x %08x %08x\n", y_add, y_sub, y_mul, y_mul7, y_max, y_bf, y_i32, y_i2f, y_trunc);
    i <= i + 1;
    if (i + 1 >= n) begin $fclose(fo); $finish; end
  end
endmodule
