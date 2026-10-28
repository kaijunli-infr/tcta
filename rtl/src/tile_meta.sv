// Per-slot tile metadata (rows_valid, cols_valid, pready), the hardware twin of the simulator's TileState.valid:
// MMA takes M, K, N from its operands' metadata, the VEC pipe its iteration bounds, LOAD.IDX the index tile's
// valid count. Three write ports (DMA, MMA, VEC complete at most one tile each per cycle; collisions are
// resolved MMA > VEC > DMA, the loser retries) and a combinational read of the whole table.
module tile_meta
  import tcta_pkg::*;
(
  input  logic       clk,
  input  logic       rst_n,
  input  logic       wr_valid [3],      // 0 DMA, 1 MMA, 2 VEC
  input  logic [7:0] wr_idx   [3],      // slot_index()
  input  tmeta_t     wr_data  [3],
  output logic       wr_ack   [3],
  output tmeta_t     meta     [N_SLOTS]
);
  always_comb begin
    wr_ack[1] = wr_valid[1];
    wr_ack[2] = wr_valid[2] && !(wr_valid[1] && wr_idx[1] == wr_idx[2]);
    wr_ack[0] = wr_valid[0] && !(wr_valid[1] && wr_idx[1] == wr_idx[0]) && !(wr_valid[2] && wr_idx[2] == wr_idx[0]);
  end
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < N_SLOTS; i++) meta[i] <= '0;
    end else begin
      for (int p = 0; p < 3; p++) if (wr_valid[p] && wr_ack[p]) meta[wr_idx[p]] <= wr_data[p];
    end
  end
endmodule
