// MMA operand block buffer (phase 8 K0.3): ROWS memories (one per array row or column), each HALVES x ENTRIES slices
// of LANES BF16 elements, as LUT-RAM lane banks: memory r, lane l, entry (half, cg) holds element k = cg*LANES + l of
// row / column r of the block in `half`. Two write shapes, one slice per cycle:
//   row mode     (INP rows, WGT rows for MMA.T): slice (r = w_row, cg = w_entry), all lanes  -> memory w_row
//   scatter mode (WGT rows k for MMA):           slice element j is column j's element k     -> memory j, lane k % LANES, entry k / LANES
// With SCATTER, lane bank (r, l) always takes slice element r ^ l ^ c (c = w_lane in scatter mode, w_row in row
// mode): one XOR permutation of the slice instead of a 2:1 data mux per bank; without, an entry is one wide word. Reads: every memory returns its element k = r_k[r]
// of half r_half[r] (asynchronous LUT-RAM read, one lane mux per memory).
module mma_opbuf
  import tcta_pkg::*;
#(
  parameter int ROWS    = MAC_ROWS,
  parameter int LANES   = BANKS,
  parameter int HALVES  = 4,
  parameter int ENTRIES = FIX_K_MAX / BANKS,
  parameter bit SCATTER = 1'b1                   // 0: row mode only (no permutation network)
)(
  input  logic                    clk,
  input  logic                    we,
  input  logic                    w_scatter,
  input  logic [$clog2(HALVES)-1:0]  w_half,
  input  logic [$clog2(ENTRIES)-1:0] w_entry,
  input  logic [$clog2(LANES)-1:0]   w_lane,       // scatter mode: the lane written in every memory
  input  logic [$clog2(ROWS)-1:0]    w_row,        // row mode: the memory written
  input  logic [15:0]             w_data [LANES],
  input  logic [$clog2(HALVES)-1:0]  r_half [ROWS],
  input  logic [$clog2(FIX_K_MAX)-1:0] r_k [ROWS],
  output logic [15:0]             r_data [ROWS]
);
  localparam int DEPTH = HALVES * ENTRIES;
  localparam int LW = $clog2(LANES);
  logic [LW-1:0] perm;
  logic [15:0] v [LANES];                        // v[x] = w_data[x ^ perm]
  always_comb begin
    perm = (SCATTER && w_scatter) ? w_lane : LW'(w_row);
    for (int x = 0; x < LANES; x++) v[x] = SCATTER ? w_data[x ^ int'({{(32-LW){1'b0}}, perm})] : w_data[x];
  end
  for (genvar r = 0; r < ROWS; r++) begin : g_row
    logic [15:0] lane_q [LANES];
    if (SCATTER) begin : g_lanes
      // per-lane banks (each with its own write enable): 32 x 16 -> two RAM32M16 per lane
      for (genvar l = 0; l < LANES; l++) begin : g_lane
        (* ram_style = "distributed" *) logic [15:0] mem [DEPTH];
        logic wen;
        always_comb wen = we && (w_scatter ? (w_lane == LW'(l)) : (w_row == r));
        always_ff @(posedge clk) if (wen) mem[{w_half, w_entry}] <= v[r ^ l];
        assign lane_q[l] = mem[{r_half[r], r_k[r][$clog2(FIX_K_MAX)-1:LW]}];
      end
    end else begin : g_wide
      // row mode only: one slice-wide word per entry packs into 14-bit RAM32M16 columns (19 cells instead of 32)
      (* ram_style = "distributed" *) logic [LANES*16-1:0] mem [DEPTH];
      logic [LANES*16-1:0] q;
      always_ff @(posedge clk) if (we && (w_row == r)) for (int l = 0; l < LANES; l++) mem[{w_half, w_entry}][l*16 +: 16] <= w_data[l];
      assign q = mem[{r_half[r], r_k[r][$clog2(FIX_K_MAX)-1:LW]}];
      always_comb for (int l = 0; l < LANES; l++) lane_q[l] = q[l*16 +: 16];
    end
    assign r_data[r] = lane_q[r_k[r][LW-1:0]];
  end
endmodule
