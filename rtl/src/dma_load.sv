// LOAD engine (LOAD.2D / LOAD.3D / LOAD.IDX, ISA §5.1-§5.2, §6.1): fetch the rows' bytes (DMA_BYTES per beat, any
// alignment, fixed-latency memory model), dequantise BANKS elements per cycle (dequant_elem), round to BF16 for INP/WGT
// destinations, write one row slice per cycle into the scratchpad. LOAD.IDX reads its indices from a VEC row tile
// (16 per scratchpad read), skips -1 rows with a zero fill and raises IDX_RANGE / IDX_SHORT like the simulator.
//
// Phase 8 K1: rows are pipelined. The engine is a fetch front and a write back end around a row-buffer ring:
//   front  (S_ROW / S_FETCH): per row, resolve the index and the address, allocate a record (one of DMA_ROWS_IN_FLIGHT
//          slots) and the row's beats in the ring, issue the beats one per cycle, go to the next row without waiting
//   arrival: responses return in issue order; a data beat lands in the ring at the row's position, a scale beat
//          (*.b32 / f4.r16: two per row, after its data) in the row's slot of `scmem`; `got_tot` counts them
//   writer (W_WRITE): takes records in order once their beats are in (`got_tot` >= the record's end count), writes
//          one slice per cycle through the dequant lanes (zero slices for -1 rows), then frees the ring space.
// The ring is RB_BANKS beat-wide LUT-RAM banks (beat p in bank p % RB_BANKS) of ROWBUF_BEATS beats in all; a row takes
// its natural number of beats at any offset, so 256-byte rows run eight deep, a 2048-byte row alone (as in v0). A slice
// read is a 4-beat window at the row's offset, each bank at its own address, the outputs rotated into beat order.
// A row costs max(beats, slices + 1) cycles in steady state instead of latency + beats + slices + 2 (v0, row-serial).
//
// LOAD.IDX on block-scaled rows (erratum E6, K1): f4.r16 rows carry their scales (fetched from the row after its
// nibbles); f8.b32 / f4.b32 rows take their scale slice from the scale grid at row idx / 32. Scale codes index from the
// row's column 0 (Appendix A), so any col0 works: code (col0 + c) / 32 for element c of the window.
module dma_load
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
(
  input  logic              clk,
  input  logic              rst_n,
  // instruction
  input  instr_t            ins,
  input  logic              issue,
  output logic              busy,
  output logic              done,
  // descriptor table (1-cycle read)
  output logic [7:0]        desc_idx,
  input  logic [DESC_W-1:0] desc_data,
  // memory read port
  output logic              mem_valid,
  output logic [ADDR_W-1:0] mem_addr,
  input  logic              mem_rvalid,
  input  logic [MEM_W-1:0]  mem_rdata,
  input  logic              mem_rerr,
  // scratchpad client (writes the destination, reads the index tile)
  output sp_req_t           sp_req,
  input  logic              sp_grant,
  input  logic              sp_rvalid,
  input  slice_t            sp_rdata,
  // tile metadata
  input  tmeta_t            meta [N_SLOTS],
  output logic              meta_we,
  output logic [7:0]        meta_idx,
  output tmeta_t            meta_wdata,
  input  logic              meta_ack,
  // fault (sticky until the next issue)
  output logic              fault,
  output fault_t            fault_code
);
  localparam int RB_BANKS = 4;                            // beats per f32 slice
  localparam int RB_DEPTH = ROWBUF_BEATS / RB_BANKS;
  localparam int RB_AW = $clog2(RB_DEPTH);
  localparam int RB_SH = $clog2(RB_BANKS);
  localparam int POS_W = $clog2(ROWBUF_BEATS);            // ring position
  localparam int NSLOT = DMA_ROWS_IN_FLIGHT;
  localparam int SLOT_W = $clog2(NSLOT);
  typedef enum logic [3:0] {S_IDLE, S_DESC, S_SETUP, S_IDX_RD, S_IDX_WAIT, S_ROW, S_FETCH, S_DRAIN, S_META, S_DONE, S_FAULT} state_t;
  typedef enum logic {W_IDLE, W_WRITE} wstate_t;
  state_t  st;
  wstate_t wst;

  instr_t      cur;
  desc_t       d;
  logic        is_idx;
  logic [15:0] dst;
  logic [11:0] idx_slot;
  int          rows, cols, row0, col0, n_el_row, nb, n_beats, n_cg;
  int          i, beat_issue, n_resp;
  logic        need_scales;        // *.b32 / f4.r16: a second fetch of this row's scale codes
  logic        is_r16, is_f4;
  logic [ADDR_W-1:0] row_addr, scale_addr;
  logic [31:0] idx_slice [BANKS];
  int          idx_cg;
  logic        to_bf16;

  // ---- row records (one per slot), the fetch FIFO (rows with beats, for the arrival side) and the counters ----
  logic [POS_W-1:0] rec_start [NSLOT];
  logic [7:0]       rec_nbeats [NSLOT];
  logic [5:0]       rec_ncg [NSLOT];
  logic [7:0]       rec_i [NSLOT];
  logic             rec_zf [NSLOT];
  logic [15:0]      rec_end [NSLOT];                      // got_tot value at which the row is complete
  logic [SLOT_W-1:0] fq [NSLOT];
  logic [SLOT_W:0]  alloc_cnt, wr_cnt, fq_wr, fq_rd;      // one bit wider than the index: full / empty
  logic [POS_W:0]   alloc_pos, free_pos;                  // ring positions, one bit wider: occupancy
  logic [15:0]      iss_tot, got_tot;
  logic [7:0]       arr_cnt;                              // beats of the arriving row received so far
  logic [SLOT_W-1:0] as, ws, fs;
  logic             slot_free, ring_free, w_pending, w_ready, w_grant, zf_now, alloc_now;
  int               wcg;
  always_comb begin
    as = alloc_cnt[SLOT_W-1:0]; ws = wr_cnt[SLOT_W-1:0]; fs = fq[fq_rd[SLOT_W-1:0]];
    slot_free = (alloc_cnt - wr_cnt) < (SLOT_W+1)'(NSLOT);
    ring_free = (ROWBUF_BEATS - int'((POS_W+1)'(alloc_pos - free_pos))) >= n_beats;   // the difference modulo 2^(POS_W+1): size it before the int cast
    w_pending = (wr_cnt != alloc_cnt);
    w_ready   = w_pending && ($signed(got_tot - rec_end[ws]) >= 0);
    w_grant   = sp_grant && (st != S_IDX_RD);
  end

  // ---- descriptor and geometry helpers ----
  always_comb desc_idx = cur.b[7:0];
  // bytes of the first `col` columns of a row (f4 formats: nibbles)
  function automatic int col_bytes(input logic [7:0] dtype, input int col);
    return (dtype == DT_F4_B32 || dtype == DT_F4_R16) ? (col >> 1) : col * el_bytes(dtype);
  endfunction

  // ---- row buffer ring: RB_BANKS LUT-RAM banks of RB_DEPTH beats; beat p lives in bank p % RB_BANKS ----
  logic [MEM_W-1:0] rb_rdata [RB_BANKS];
  logic [RB_AW-1:0] rb_raddr [RB_BANKS];
  int   w0;                                  // first beat of slice wcg's payload within the row
  logic [POS_W-1:0] wpos;                    // its ring position
  logic rb_we, arr_data;
  logic [POS_W-1:0] arr_wpos;
  always_comb begin
    case (d.dtype)
      DT_F32, DT_I32:     w0 = wcg * 4;
      DT_BF16:            w0 = wcg * 2;
      DT_F8_T, DT_F8_B32: w0 = wcg;
      default:            w0 = wcg >> 1;
    endcase
    wpos = rec_start[ws] + POS_W'(w0);
    // bank k holds beat wpos + ((k - wpos) mod RB_BANKS) of the window
    for (int k = 0; k < RB_BANKS; k++) rb_raddr[k] = RB_AW'((int'(wpos) + ((k - int'(wpos)) & (RB_BANKS - 1))) >> RB_SH);
    arr_data = mem_rvalid && (fq_rd != fq_wr) && (arr_cnt < rec_nbeats[fs]);
    arr_wpos = rec_start[fs] + POS_W'(arr_cnt);
    rb_we = arr_data;
  end
  for (genvar k = 0; k < RB_BANKS; k++) begin : g_rb
    (* ram_style = "distributed" *) logic [MEM_W-1:0] mem [RB_DEPTH];
    always_ff @(posedge clk) if (rb_we && (int'(arr_wpos) & (RB_BANKS - 1)) == k) mem[RB_AW'(arr_wpos >> RB_SH)] <= mem_rdata;
    assign rb_rdata[k] = mem[rb_raddr[k]];
  end
  // scale beats per slot (two per row)
  (* ram_style = "distributed" *) logic [MEM_W-1:0] scmem0 [NSLOT];
  (* ram_style = "distributed" *) logic [MEM_W-1:0] scmem1 [NSLOT];
  logic [2*MEM_W-1:0] scw;
  always_ff @(posedge clk) begin
    if (mem_rvalid && (fq_rd != fq_wr) && !arr_data) begin
      if (arr_cnt == rec_nbeats[fs]) scmem0[fs] <= mem_rdata; else scmem1[fs] <= mem_rdata;
    end
  end
  assign scw = {scmem1[ws], scmem0[ws]};

  // ---- element extraction: the 4-beat window in beat order, then the dtype's element spacing ----
  logic [RB_BANKS*MEM_W-1:0] win;
  logic [31:0] raw   [BANKS];
  logic [7:0]  sc8   [BANKS];
  logic [31:0] deq   [BANKS];
  logic [15:0] deq_bf [BANKS];
  slice_t      wslice;
  int          nib0;                         // f4: nibble of element 0 of the slice within the window
  always_comb begin
    for (int j = 0; j < RB_BANKS; j++) win[j*MEM_W +: MEM_W] = rb_rdata[(int'(wpos) + j) & (RB_BANKS - 1)];
    nib0 = (wcg & 1) * BANKS + (col0 & 1);
    for (int e = 0; e < BANKS; e++) begin
      int sc_i;
      case (d.dtype)
        DT_F32, DT_I32:     raw[e] = win[e*32 +: 32];
        DT_BF16:            raw[e] = 32'(win[e*16 +: 16]);
        DT_F8_T, DT_F8_B32: raw[e] = 32'(win[e*8 +: 8]);
        default:            raw[e] = 32'(win[(nib0 + e)*4 +: 4]);
      endcase
      case (d.dtype)
        DT_F8_B32, DT_F4_B32: sc_i = ((col0 & 31) + wcg * BANKS + e) >> 5;   // scale codes fetched from code col0 / 32
        DT_F4_R16:            sc_i = ((col0 & 15) + wcg * BANKS + e) >> 4;   // scale bytes fetched from column col0 & ~15
        default:              sc_i = 0;
      endcase
      sc8[e] = scw[sc_i*8 +: 8];
    end
  end
  for (genvar e = 0; e < BANKS; e++) begin : g_deq
    dequant_elem u_dq (.dtype(d.dtype), .raw(raw[e]), .scale8(sc8[e]), .scale_t(d.scale_t), .y(deq[e]));
    f32_to_bf16  u_bf (.x(deq[e]), .y(deq_bf[e]));
    always_comb wslice[e] = rec_zf[ws] ? 32'd0 : to_bf16 ? {deq_bf[e], 16'd0} : deq[e];
  end

  // ---- index conversion ----
  // one index converter: only index i of the 16-entry index slice is needed per row
  logic signed [31:0] idx_cur;
  f32_to_int_trunc u_idx (.x(idx_slice[i % BANKS]), .y(idx_cur));

  // ---- row address: one 32 x 32 multiplier for the gathered-row (idx * stride0) and 2-D ((row0 + i) * stride0) cases ----
  logic signed [31:0] row_mul_a;
  logic signed [63:0] row_mul;
  logic [ADDR_W-1:0]  row_base;                // this row's first byte (column 0)
  logic [63:0]        grid_addr;               // *.b32: this row's scale codes from column col0 / 32 of scale row (row / 32)
  always_comb begin
    row_mul_a = is_idx ? idx_cur : 32'(row0 + i);
    row_mul = row_mul_a * $signed({1'b0, d.stride0});
    row_base = ADDR_W'(d.base + 64'(cur.base_off) + 64'(row_mul));
    grid_addr = d.scales_base + 64'(cur.scales_off) + 64'($unsigned(row_mul_a) >> 5) * 64'(d.scales_stride) + 64'(col0 / 32);
  end

  // ---- outputs ----
  always_comb begin
    busy = (st != S_IDLE);
    done = (st == S_DONE);
    mem_valid = 1'b0; mem_addr = '0;
    sp_req = '0; meta_we = 1'b0; meta_idx = '0; meta_wdata = '0;
    if (st == S_IDX_RD) sp_req = slice_req(SP_VEC, idx_slot, 0, i / BANKS, 1'b0, '0, '0);   // the slice holding index i
    else if (wst == W_WRITE) sp_req = slice_req(tref_sp(dst), tref_slot(dst), int'(rec_i[ws]), wcg, 1'b1, slice_ben(cols, wcg), wslice);
    if (st == S_FETCH) begin
      mem_valid = 1'b1;
      if (beat_issue < n_beats) mem_addr = row_addr + ADDR_W'(beat_issue * DMA_BYTES);
      else mem_addr = scale_addr + ADDR_W'((beat_issue - n_beats) * DMA_BYTES);
    end
    if (st == S_META) begin
      meta_we = 1'b1; meta_idx = 8'(slot_index(tref_sp(dst), tref_slot(dst)));
      meta_wdata = '{rows: 10'(rows), cols: 10'(cols), pready: 1'b0};
    end
  end

  task automatic set_fault(input fault_t code);
    fault <= 1'b1; fault_code <= code; st <= S_FAULT;
  endtask

  // ---- allocation: a row record (and, unless the row is skipped, its beats in the ring) ----
  always_comb begin
    zf_now = is_idx && (idx_cur == -1);
    alloc_now = (st == S_ROW) && (i < rows) && (n_cg != 0) && !(is_idx && idx_cg != (i / BANKS)) && slot_free &&
                (zf_now || (ring_free && !(is_idx && (idx_cur < 0 || idx_cur >= $signed(d.shape0)))));
  end
  // records: no reset, always written before they are read
  always_ff @(posedge clk) begin
    if (alloc_now) begin
      rec_start[as] <= alloc_pos[POS_W-1:0]; rec_nbeats[as] <= zf_now ? 8'd0 : 8'(n_beats); rec_ncg[as] <= 6'(n_cg);
      rec_i[as] <= 8'(i); rec_zf[as] <= zf_now; rec_end[as] <= iss_tot + (zf_now ? 16'd0 : 16'(n_beats + (need_scales ? 2 : 0)));
      if (!zf_now) fq[fq_wr[SLOT_W-1:0]] <= as;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; wst <= W_IDLE; fault <= 1'b0; fault_code <= F_NONE; cur <= '0; d <= '0; is_idx <= 0; dst <= '0; idx_slot <= '0;
      rows <= 0; cols <= 0; row0 <= 0; col0 <= 0; n_el_row <= 0; nb <= 0; n_beats <= 0; n_cg <= 0;
      i <= 0; wcg <= 0; beat_issue <= 0; n_resp <= 0; need_scales <= 0; is_r16 <= 0; is_f4 <= 0;
      row_addr <= '0; scale_addr <= '0; idx_cg <= 0; to_bf16 <= 0;
      alloc_cnt <= '0; wr_cnt <= '0; fq_wr <= '0; fq_rd <= '0; alloc_pos <= '0; free_pos <= '0; iss_tot <= '0; got_tot <= '0; arr_cnt <= '0;
      for (int k = 0; k < BANKS; k++) idx_slice[k] <= '0;
    end else begin
      // ---- front ----
      case (st)
        S_IDLE: if (issue) begin
          cur <= ins; fault <= 1'b0; fault_code <= F_NONE; st <= S_DESC;
          alloc_cnt <= '0; wr_cnt <= '0; fq_wr <= '0; fq_rd <= '0; alloc_pos <= '0; free_pos <= '0; iss_tot <= '0; got_tot <= '0; arr_cnt <= '0;
          wst <= W_IDLE;
        end
        S_DESC: st <= S_SETUP;                       // desc_data valid now (registered read of desc_idx = cur.b[7:0])
        S_SETUP: begin
          d <= unpack_desc(desc_data);
          begin
            desc_t dd; int rb, ncols, dbytes; logic [15:0] dstr;
            dd = unpack_desc(desc_data);
            dstr = cur.a;
            dst <= dstr;
            to_bf16 <= (tref_sp(dstr) == SP_INP || tref_sp(dstr) == SP_WGT);
            is_r16 <= (dd.dtype == DT_F4_R16);
            is_f4 <= (dd.dtype == DT_F4_B32 || dd.dtype == DT_F4_R16);
            need_scales <= (dd.dtype == DT_F8_B32 || dd.dtype == DT_F4_B32 || dd.dtype == DT_F4_R16);
            i <= 0; idx_cg <= -1;
            ncols = int'(cur.imm[15:0]); cols <= ncols; n_cg <= n_slices(ncols); col0 <= int'(cur.d);
            // f4: an odd col0 puts element 0 in the high nibble of the first byte, so the window is one byte longer
            dbytes = (dd.dtype == DT_F4_B32 || dd.dtype == DT_F4_R16) ? ((ncols + int'(cur.d[0]) + 1) >> 1) : ncols * el_bytes(dd.dtype);
            nb <= dbytes; n_beats <= (dbytes + DMA_BYTES - 1) / DMA_BYTES;
            if (cur.op == OP_LOAD_IDX) begin
              is_idx <= 1'b1;
              rows <= int'(cur.b[15:8]); row0 <= 0; n_el_row <= int'(dd.shape1); idx_slot <= tref_slot(cur.c);
              rb = int'(cur.imm[31:16]);
              if (tref_sp(dstr) == SP_STATE || tref_sp(dstr) > SP_STATE) set_fault(F_BAD_SHAPE);
              else if (rb % 64 != 0 || rb > ROWBUF_BYTES) set_fault(F_BAD_SHAPE);
              else if (tref_sp(cur.c) != SP_VEC || int'(tref_slot(cur.c)) < VEC_ROW_SLOT0) set_fault(F_BAD_SHAPE);
              else if (int'(meta[slot_index(SP_VEC, tref_slot(cur.c))].cols) < int'(cur.b[15:8])) set_fault(F_IDX_SHORT);
              else if (dd.dtype > DT_F4_R16) set_fault(F_BAD_QUALIFIER);
              else if (ncols > sp_cols(tref_sp(dstr), tref_slot(dstr))) set_fault(F_BAD_SHAPE);
              else st <= S_ROW;
            end else begin
              is_idx <= 1'b0;
              rows <= int'(cur.imm[31:16]); row0 <= int'(cur.c); n_el_row <= ncols;
              rb = row_bytes_for(dd.dtype, ncols);
              if (int'(cur.imm[31:16]) > sp_rows(tref_sp(dstr), tref_slot(dstr)) ||
                  ncols > sp_cols(tref_sp(dstr), tref_slot(dstr)) || rb > ROWBUF_BYTES) set_fault(F_BAD_SHAPE);
              else if (dd.dtype > DT_F4_R16) set_fault(F_BAD_QUALIFIER);
              else st <= S_ROW;
            end
          end
        end
        // ---- per row: resolve, allocate, fetch ----
        S_ROW: begin
          if (i >= rows) st <= S_DRAIN;
          else if (n_cg == 0) i <= i + 1;
          else if (is_idx && idx_cg != (i / BANKS)) st <= S_IDX_RD;
          else if (is_idx && idx_cur != -1 && (idx_cur < 0 || idx_cur >= $signed(d.shape0))) set_fault(F_IDX_RANGE);
          else if (alloc_now) begin
            alloc_cnt <= alloc_cnt + 1'b1; i <= i + 1;
            if (!zf_now) begin                         // a skipped row is a record with no beats, written as zeros in its turn
              fq_wr <= fq_wr + 1'b1;
              alloc_pos <= alloc_pos + (POS_W+1)'(n_beats);
              iss_tot <= iss_tot + 16'(n_beats + (need_scales ? 2 : 0));
              row_addr <= row_base + ADDR_W'(col_bytes(d.dtype, col0));
              if (is_r16) scale_addr <= row_base + ADDR_W'((n_el_row >> 1) + (col0 >> 4));
              else scale_addr <= ADDR_W'(grid_addr);
              beat_issue <= 0; n_resp <= n_beats + (need_scales ? 2 : 0);
              st <= S_FETCH;
            end
          end
          // else: no free slot or ring space yet; retry next cycle
        end
        S_IDX_RD: if (sp_grant) begin idx_cg <= i / BANKS; st <= S_IDX_WAIT; end
        S_IDX_WAIT: if (sp_rvalid) begin
          for (int k = 0; k < BANKS; k++) idx_slice[k] <= sp_rdata[k];
          st <= S_ROW;
        end
        S_FETCH: begin
          if (beat_issue + 1 >= n_resp) st <= S_ROW;
          beat_issue <= beat_issue + 1;
        end
        S_DRAIN: if (!w_pending && wst == W_IDLE) st <= S_META;
        S_META: if (meta_ack) st <= S_DONE;
        S_DONE: st <= S_IDLE;
        S_FAULT: ;                                   // sticky: the core halts
        default: st <= S_IDLE;
      endcase

      // ---- arrival (responses in issue order; the data beats land in the ring, see rb_we) ----
      if (mem_rvalid && (fq_rd != fq_wr)) begin
        if (mem_rerr) set_fault(F_PAGE_FAULT);
        got_tot <= got_tot + 1'b1;
        if (int'(arr_cnt) + 1 >= int'(rec_nbeats[fs]) + (need_scales ? 2 : 0)) begin fq_rd <= fq_rd + 1'b1; arr_cnt <= '0; end
        else arr_cnt <= arr_cnt + 1'b1;
      end

      // ---- writer ----
      case (wst)
        W_IDLE: if (w_ready && st != S_IDLE && st != S_FAULT) begin wcg <= 0; wst <= W_WRITE; end
        W_WRITE: if (w_grant) begin
          if (wcg + 1 >= int'(rec_ncg[ws])) begin
            wst <= W_IDLE; wr_cnt <= wr_cnt + 1'b1; free_pos <= free_pos + (POS_W+1)'(rec_nbeats[ws]);
          end
          wcg <= wcg + 1;
        end
        default: wst <= W_IDLE;
      endcase
    end
  end
endmodule
