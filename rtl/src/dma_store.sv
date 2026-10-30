// STORE engine (STORE.2D / STORE.IDX, ISA §5.2): per row, read the source tile's row slices (wide: four consecutive ACC
// slots as [M, 4*T_N]), convert (f32 as is, i32 = rint, bf16 = RNE top half, f4.r16 = quantise, Appendix A), pack into
// a row buffer, then write the row as DMA_BYTES beats with byte enables. STORE.IDX scatters rows to idx[i] (-1 skips,
// IDX_RANGE / IDX_SHORT as the simulator).
//
// Phase 8 K1: rows are pipelined. A read front and a beat-writer back end share a row-buffer ring of ROWBUF_BEATS beats
// (RB_BANKS beat-wide LUT-RAM banks, beat p in bank p % RB_BANKS; rows start at 4-beat boundaries so a converted f32 slice
// is one write of four banks at one address, a bf16 slice two banks):
//   front  (S_RD): per row, resolve the index and the address, allocate a record (DMA_ROWS_IN_FLIGHT slots) and the row's
//          beats, then issue one slice read per cycle; the response lands in the ring the next cycle (the scratchpad
//          returns a read one cycle after its grant, in order) and the last response marks the record complete
//   writer (W_BEATS): takes complete records in order, one beat per cycle to the memory port, then frees the ring.
// A row costs max(slices, beats + 1) cycles in steady state (v0: 2 * slices + beats + 1, row-serial).
//
// f4.r16 egress (erratum E6, K1): a slice (16 channels) is one scale block. The quantiser reproduces the simulator
// (`dtypes.encode_f4_r16_row`) bit for bit: scale = E4M3 nearest of RN(amax / 6) with ties to the smaller magnitude
// (1.0 if amax is 0 or the block has a NaN, 448 at the top), then code = E2M1 nearest of RN(x / scale), ties to the
// smaller magnitude, non-finite quotients to code 0, the sign from `x < 0`. The division is exact in integers: with
// scale = S * 2^es (S the 4-bit E4M3 significand) and midpoints m_t = M_t / 4, RN(x / scale) > m_t iff
// X * 2^(ex - es - 124) > M_t * S * (2^24 + 1), one shifter and seven comparators per lane (R16_LANES lanes, a slice in
// 16 / R16_LANES cycles). The row's nibbles, its scale bytes and the zero padding to 64 B go through a one-beat assembler
// into the ring (the egress path is not on any family's hot loop: 4 lanes).
module dma_store
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
(
  input  logic              clk,
  input  logic              rst_n,
  input  instr_t            ins,
  input  logic              issue,
  output logic              busy,
  output logic              done,
  output logic [7:0]        desc_idx,
  input  logic [DESC_W-1:0] desc_data,
  // memory write port (always accepted)
  output logic              mem_valid,
  output logic [ADDR_W-1:0] mem_addr,
  output logic [MEM_W-1:0]  mem_wdata,
  output logic [DMA_BYTES-1:0] mem_be,
  // scratchpad client (reads)
  output sp_req_t           sp_req,
  input  logic              sp_grant,
  input  logic              sp_rvalid,
  input  slice_t            sp_rdata,
  input  tmeta_t            meta [N_SLOTS],
  output logic              fault,
  output fault_t            fault_code
);
  localparam int RB_BANKS = 4;                            // beats per f32 slice
  localparam int RB_DEPTH = ROWBUF_BEATS / RB_BANKS;
  localparam int RB_AW = $clog2(RB_DEPTH);
  localparam int RB_SH = $clog2(RB_BANKS);
  localparam int POS_W = $clog2(ROWBUF_BEATS);
  localparam int NSLOT = DMA_ROWS_IN_FLIGHT;
  localparam int SLOT_W = $clog2(NSLOT);
  localparam int R16_LANES = 4;                           // f4.r16 quantiser lanes
  localparam int R16_STEPS = BANKS / R16_LANES;
  typedef enum logic [3:0] {S_IDLE, S_DESC, S_SETUP, S_IDX_RD, S_IDX_WAIT, S_ROW, S_RD, S_Q_WAIT, S_Q_AMAX, S_Q_SCALE, S_Q_LANE,
                            S_Q_PUT, S_FLUSH, S_DRAIN, S_DONE, S_FAULT} state_t;
  typedef enum logic {W_IDLE, W_BEATS} wstate_t;
  state_t  st;
  wstate_t wst;
  instr_t      cur;
  desc_t       d;
  logic        is_idx, wide, is_bf, is_r16;
  logic [15:0] src;
  logic [11:0] idx_slot;
  int          rows, cols, row0, col0, nb, n_beats, n_cg, eb, n_alloc;
  int          i, cg, idx_cg;
  logic [31:0] idx_slice [BANKS];
  // one index converter: only index i of the 16-entry index slice is needed per row
  logic signed [31:0] idx_cur;
  f32_to_int_trunc u_idx (.x(idx_slice[i % BANKS]), .y(idx_cur));
  always_comb desc_idx = cur.b[7:0];

  // ---- row records, counters ----
  logic [POS_W-1:0] rec_start [NSLOT];
  logic [7:0]       rec_nbeats [NSLOT];
  logic [11:0]      rec_nb [NSLOT];
  logic [ADDR_W-1:0] rec_addr [NSLOT];
  logic             rec_done [NSLOT];
  logic [SLOT_W:0]  alloc_cnt, wr_cnt;
  logic [POS_W:0]   alloc_pos, free_pos;
  logic [SLOT_W-1:0] as, ws, cur_slot;
  logic             slot_free, ring_free, w_pending, alloc_now;
  int               beat;
  always_comb begin
    as = alloc_cnt[SLOT_W-1:0]; ws = wr_cnt[SLOT_W-1:0];
    slot_free = (alloc_cnt - wr_cnt) < (SLOT_W+1)'(NSLOT);
    ring_free = (ROWBUF_BEATS - int'((POS_W+1)'(alloc_pos - free_pos))) >= n_alloc;   // the difference modulo 2^(POS_W+1): size it before the int cast
    w_pending = (wr_cnt != alloc_cnt);
  end

  // ---- conversion of the returned slice (f32 / i32 / bf16 path: straight into the ring) ----
  logic [31:0] cv_i32 [BANKS];
  logic [15:0] cv_bf  [BANKS];
  for (genvar e = 0; e < BANKS; e++) begin : g_cv
    f32_to_i32_rne u_i (.x(sp_rdata[e]), .y(cv_i32[e]));
    f32_to_bf16    u_b (.x(sp_rdata[e]), .y(cv_bf[e]));
  end
  // the slice read granted last cycle: its response is on sp_rdata now
  logic             pend, pend_last;
  int               pend_cg;
  logic [SLOT_W-1:0] pend_slot;

  // ---- f4.r16 quantiser ----
  slice_t      qs;                                        // the slice being quantised
  logic [30:0] q_amax; logic q_nan;
  logic [7:0]  q_sc;                                      // the block's E4M3 scale code
  logic [63:0] q_nib;                                     // the slice's 8 nibble bytes
  logic [7:0]  sc_bytes [32];                             // the row's scale bytes (slice cg -> byte cg)
  int          qstep;
  // RN(amax / 6) then nearest E4M3 (ties to the smaller magnitude): dtypes.quant_e4m3(amax / 6)
  function automatic logic [27:0] div3(input logic [25:0] x);        // {quotient, remainder}
    logic [25:0] q; logic [1:0] r; logic [2:0] t;
    q = '0; r = '0;
    for (int b = 25; b >= 0; b--) begin
      t = {r, x[b]};
      if (t >= 3'd3) begin q[b] = 1'b1; r = 2'(t - 3'd3); end else begin q[b] = 1'b0; r = t[1:0]; end
    end
    return {q, r};
  endfunction
  function automatic logic [7:0] r16_scale_code(input logic [30:0] amax, input logic nan);
    logic [7:0] ea; logic [23:0] A, q; logic [25:0] x6; logic [27:0] qr; int s, ev;
    if (nan || amax == '0) return 8'h38;                           // scale 1.0
    ea = amax[30:23];
    if (ea == 8'd255) return 8'h7e;                                 // inf -> min(inf, 448)
    if (ea == 8'd0) return 8'h00;                                   // a denormal amax: v far below 2^-10
    A = {1'b1, amax[22:0]};
    s = (A >= 24'hC00000) ? 1 : 2;                                  // q = A * 2^s / 3 has 24 significant bits
    x6 = 26'(A) << s;
    qr = div3(x6); q = qr[25:2];
    if (qr[1:0] == 2'd2) q = q + 24'd1;                             // fraction 2/3 rounds up, 1/3 down; no ties
    ev = int'(ea) - 1 - s;                                          // v = q * 2^(ev - 150), biased exponent ev
    if (ev > 135 || (ev == 135 && q >= 24'hE00000)) return 8'h7e;  // v >= 448
    if (ev >= 121) begin                                            // v >= 2^-6: normal E4M3, e = ev - 120
      logic [7:0] code;
      code = {1'b0, 4'(ev - 120), q[22:20]} + 8'((q[19:0] > 20'h80000) ? 1 : 0);
      return (code > 8'h7e) ? 8'h7e : code;
    end else begin                                                  // subnormal: k = round-half-down(v * 2^9)
      int sh; logic [23:0] rem, half; logic [3:0] k;
      sh = 141 - ev;
      if (sh >= 25) return 8'h00;
      k = 4'(q >> sh);
      rem = q & ((24'd1 << sh) - 24'd1); half = 24'd1 << (sh - 1);
      if (rem > half) k = k + 4'd1;
      return {4'd0, k};
    end
  endfunction
  // scale fields and the seven lane thresholds M_t * S * (2^24 + 1), M_t in {1, 3, 5, 7, 10, 14, 20}
  logic        q_sz; int q_es; logic [3:0] q_S;
  logic [33:0] q_P [7];
  localparam int MT [7] = '{1, 3, 5, 7, 10, 14, 20};
  always_comb begin
    q_sz = (q_sc[6:0] == 7'd0);
    if (q_sc[6:3] == 4'd0) begin q_S = {1'b0, q_sc[2:0]}; q_es = -9; end
    else begin q_S = {1'b1, q_sc[2:0]}; q_es = int'(q_sc[6:3]) - 10; end
    for (int t = 0; t < 7; t++) begin
      logic [8:0] ms;
      ms = 9'(MT[t] * int'(q_S));
      q_P[t] = {1'b0, ms, 24'd0} | 34'(ms);
    end
  end
  // E2M1 code of RN(x / scale), ties to the smaller magnitude; the sign from x < 0 (dtypes.quant_e2m1)
  function automatic logic [3:0] r16_code(input logic [31:0] x, input logic sz, input int es, input logic [33:0] P [7]);
    logic [7:0] ex; logic [22:0] mx; logic nan, inf, zero, neg; logic [23:0] xm; int exe, D; logic [33:0] xs; logic [2:0] c;
    ex = x[30:23]; mx = x[22:0];
    nan = (ex == 8'd255) && (mx != '0); inf = (ex == 8'd255) && (mx == '0); zero = (ex == 8'd0) && (mx == '0);
    neg = x[31] && !nan && !zero;
    if (nan || zero) return 4'd0;
    if (inf || sz) return {neg, 3'd0};                              // inf, or x / 0: a non-finite quotient
    xm = (ex == 8'd0) ? {1'b0, mx} : {1'b1, mx}; exe = (ex == 8'd0) ? 1 : int'(ex);
    D = exe - es - 124;
    c = 3'd0;
    if (D >= 10) c = 3'd7;
    else if (D >= 1) begin
      xs = 34'(xm) << D;
      for (int t = 0; t < 7; t++) if (xs > P[t]) c = c + 3'd1;
    end
    return {neg, c};
  endfunction
  logic [3:0] q_code [R16_LANES];
  always_comb for (int l = 0; l < R16_LANES; l++) q_code[l] = r16_code(qs[qstep * R16_LANES + l], q_sz, q_es, q_P);

  // ---- the one-beat assembler (f4.r16): 8-byte puts at 8-byte offsets of the row, flushed when the beat changes ----
  logic        asm_valid, put_v, flush_v, zero_v;
  logic [7:0]  asm_idx, zidx, put_idx;
  logic [MEM_W-1:0] asm_beat, put_merged;
  int          put_off; logic [63:0] put_data;
  int          fl, n_sc_chunks, nb_nib;
  always_comb begin
    put_idx = 8'(put_off >> 4);
    put_merged = (asm_valid && put_idx == asm_idx) ? asm_beat : '0;
    if (put_off[3]) put_merged[127:64] = put_data; else put_merged[63:0] = put_data;
  end

  // ---- row buffer ring: RB_BANKS LUT-RAM banks ----
  logic [MEM_W-1:0] rb_wdata [RB_BANKS], rb_rdata [RB_BANKS];
  logic [RB_BANKS-1:0] rb_we;
  logic [RB_AW-1:0] rb_waddr, rb_raddr;
  logic [POS_W-1:0] rpos, apos;                           // writer read position; assembler write position
  logic [7:0]       aidx;                                 // the row beat the assembler writes this cycle
  logic [MEM_W-1:0] abeat;
  always_comb begin
    rpos = rec_start[ws] + POS_W'(beat);
    rb_raddr = RB_AW'(rpos >> RB_SH);
    rb_we = '0; rb_waddr = '0;
    for (int k = 0; k < RB_BANKS; k++) rb_wdata[k] = '0;
    // assembler write: a put that leaves the open beat, a flush, or a zero beat
    aidx = zero_v ? zidx : asm_idx;
    abeat = zero_v ? '0 : asm_beat;
    apos = rec_start[cur_slot] + POS_W'(aidx);
    if (is_r16) begin
      if (zero_v || flush_v || (put_v && asm_valid && put_idx != asm_idx)) begin
        rb_waddr = RB_AW'(apos >> RB_SH);
        for (int k = 0; k < RB_BANKS; k++) begin rb_we[k] = ((int'(apos) & (RB_BANKS - 1)) == k); rb_wdata[k] = abeat; end
      end
    end else if (pend && sp_rvalid) begin
      // slice pend_cg of the row in slot pend_slot: f32 / i32 occupy beats 4cg .. 4cg+3 (bank k holds elements 4k .. 4k+3),
      // bf16 beats 2cg, 2cg+1 (banks 2(cg%2), +1, eight elements per beat). Little-endian within the beat.
      rb_waddr = RB_AW'((int'(rec_start[pend_slot]) >> RB_SH) + (is_bf ? (pend_cg >> 1) : pend_cg));
      for (int k = 0; k < RB_BANKS; k++) begin
        rb_we[k] = !is_bf || ((k >> 1) == (pend_cg & 1));
        for (int j = 0; j < 4; j++) rb_wdata[k][j*32 +: 32] = (d.dtype == DT_I32) ? cv_i32[4*k + j] : sp_rdata[4*k + j];
        if (is_bf) for (int j = 0; j < 8; j++) rb_wdata[k][j*16 +: 16] = cv_bf[8*(k & 1) + j];
      end
    end
  end
  for (genvar k = 0; k < RB_BANKS; k++) begin : g_rb
    (* ram_style = "distributed" *) logic [MEM_W-1:0] mem [RB_DEPTH];
    always_ff @(posedge clk) if (rb_we[k]) mem[rb_waddr] <= rb_wdata[k];
    assign rb_rdata[k] = mem[rb_raddr];
  end

  // the read of slice cg of row i: wide tiles continue into the next slot every T_N columns
  function automatic sp_req_t src_req(input int row, input int c);
    logic [11:0] slot; int cgl;
    slot = tref_slot(src) + 12'(wide ? (c * BANKS) / T_N : 0);
    cgl = wide ? (c % (T_N / BANKS)) : c;
    return slice_req(tref_sp(src), slot, row, cgl, 1'b0, slice_ben(cols, c), '0);   // ben on a read: the lanes wanted (E2 pad)
  endfunction

  int rem_bytes;
  always_comb begin
    busy = (st != S_IDLE); done = (st == S_DONE);
    sp_req = '0; mem_valid = 1'b0; mem_addr = '0; mem_wdata = '0; mem_be = '0;
    rem_bytes = int'(rec_nb[ws]) - beat * DMA_BYTES;
    case (st)
      S_IDX_RD: sp_req = slice_req(SP_VEC, idx_slot, 0, i / BANKS, 1'b0, '0, '0);   // the slice holding index i
      S_RD:     sp_req = src_req(i, cg);
      default: ;
    endcase
    if (wst == W_BEATS) begin
      mem_valid = 1'b1; mem_addr = rec_addr[ws] + ADDR_W'(beat * DMA_BYTES);
      mem_wdata = rb_rdata[int'(rpos) & (RB_BANKS - 1)];
      for (int k = 0; k < DMA_BYTES; k++) mem_be[k] = (k < rem_bytes);
    end
  end

  task automatic set_fault(input fault_t code);
    fault <= 1'b1; fault_code <= code; st <= S_FAULT;
  endtask

  // row address: one 32 x 32 multiplier shared by the gathered-row (idx * stride0) and the 2-D ((row0 + i) * stride0) cases
  logic signed [31:0] row_mul_a;
  logic signed [63:0] row_mul;
  always_comb begin
    row_mul_a = is_idx ? idx_cur : 32'(row0 + i);
    row_mul = row_mul_a * $signed({1'b0, d.stride0});
  end

  // ---- allocation ----
  always_comb begin
    alloc_now = (st == S_ROW) && (i < rows) && !(is_idx && idx_cg != (i / BANKS)) && !(is_idx && idx_cur == -1) &&
                !(is_idx && (idx_cur < 0 || idx_cur >= $signed(d.shape0))) && slot_free && ring_free;
    // f4.r16 row end: the scale bytes go out in 8-byte chunks, then the open beat, then zero beats to the 64-byte row
    nb_nib = cols >> 1;
    n_sc_chunks = (n_cg + 7) / 8;
    put_v = 1'b0; flush_v = 1'b0; zero_v = 1'b0; put_off = 0; put_data = '0;
    if (st == S_Q_PUT) begin
      put_v = 1'b1; put_off = cg * 8; put_data = q_nib;
    end else if (st == S_FLUSH) begin
      if (fl < n_sc_chunks) begin
        put_v = 1'b1; put_off = nb_nib + fl * 8;
        for (int j = 0; j < 8; j++) put_data[j*8 +: 8] = (fl * 8 + j < n_cg) ? sc_bytes[fl * 8 + j] : 8'd0;
      end else if (fl == n_sc_chunks) flush_v = asm_valid;
      else zero_v = (int'(zidx) < n_beats);
    end
  end
  // records: no reset, written before they are read
  always_ff @(posedge clk) begin
    if (alloc_now) begin
      rec_start[as] <= alloc_pos[POS_W-1:0]; rec_nbeats[as] <= 8'(n_beats); rec_nb[as] <= 12'(nb);
      rec_addr[as] <= ADDR_W'(d.base + 64'(cur.base_off) + 64'(row_mul) + (is_idx ? 64'd0 : 64'(col0 * eb)));
      rec_done[as] <= 1'b0;
    end
    if (!is_r16 && pend && sp_rvalid && pend_last) rec_done[pend_slot] <= 1'b1;
    if (is_r16 && st == S_FLUSH && fl > n_sc_chunks && int'(zidx) >= n_beats) rec_done[cur_slot] <= 1'b1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= S_IDLE; wst <= W_IDLE; fault <= 0; fault_code <= F_NONE; cur <= '0; d <= '0; is_idx <= 0; wide <= 0; is_bf <= 0; is_r16 <= 0;
      src <= '0; idx_slot <= '0;
      rows <= 0; cols <= 0; row0 <= 0; col0 <= 0; nb <= 0; n_beats <= 0; n_cg <= 0; eb <= 0; n_alloc <= 0; i <= 0; cg <= 0; beat <= 0;
      idx_cg <= -1; alloc_cnt <= '0; wr_cnt <= '0; alloc_pos <= '0; free_pos <= '0; cur_slot <= '0;
      pend <= 0; pend_last <= 0; pend_cg <= 0; pend_slot <= '0;
      qs <= '0; q_amax <= '0; q_nan <= 0; q_sc <= '0; q_nib <= '0; qstep <= 0; asm_valid <= 0; asm_idx <= '0; asm_beat <= '0; zidx <= '0; fl <= 0;
      for (int k = 0; k < BANKS; k++) idx_slice[k] <= '0;
      for (int k = 0; k < 32; k++) sc_bytes[k] <= '0;
    end else begin
      pend <= 1'b0;
      // ---- front ----
      case (st)
        S_IDLE: if (issue) begin
          cur <= ins; fault <= 0; fault_code <= F_NONE; st <= S_DESC;
          alloc_cnt <= '0; wr_cnt <= '0; alloc_pos <= '0; free_pos <= '0; wst <= W_IDLE; asm_valid <= 0;
        end
        S_DESC: st <= S_SETUP;
        S_SETUP: begin
          desc_t dd; int e2, c, ncols, rb;
          dd = unpack_desc(desc_data); d <= dd;
          src <= cur.a; wide <= cur.flags[FL_WIDE] && (cur.op == OP_STORE_2D);
          e2 = el_bytes(dd.dtype); eb <= e2; is_bf <= (dd.dtype == DT_BF16); is_r16 <= (dd.dtype == DT_F4_R16);
          i <= 0; cg <= 0; idx_cg <= -1;
          ncols = int'(cur.imm[15:0]); cols <= ncols; n_cg <= n_slices(ncols);
          // f4.r16: nibbles, one scale byte per 16 channels, zero padding to 64 B (dtypes.encode_f4_r16_row)
          rb = (dd.dtype == DT_F4_R16) ? ((ncols / 2 + ncols / 16 + 63) / 64) * 64 : ncols * e2;
          nb <= rb; n_beats <= (rb + DMA_BYTES - 1) / DMA_BYTES;
          n_alloc <= ((rb + DMA_BYTES - 1) / DMA_BYTES + RB_BANKS - 1) / RB_BANKS * RB_BANKS;
          if (cur.op == OP_STORE_IDX) begin
            is_idx <= 1; rows <= int'(cur.b[15:8]); row0 <= 0; col0 <= 0; idx_slot <= tref_slot(cur.c);
            if (tref_sp(cur.c) != SP_VEC || int'(tref_slot(cur.c)) < VEC_ROW_SLOT0) set_fault(F_BAD_SHAPE);
            else if (int'(meta[slot_index(SP_VEC, tref_slot(cur.c))].cols) < int'(cur.b[15:8])) set_fault(F_IDX_SHORT);
            else if (!(dd.dtype == DT_F32 || dd.dtype == DT_I32 || dd.dtype == DT_BF16)) set_fault(F_BAD_QUALIFIER);
            else st <= S_ROW;
          end else begin
            is_idx <= 0; rows <= int'(cur.imm[31:16]); row0 <= int'(cur.c); col0 <= int'(cur.d);
            c = cur.flags[FL_WIDE] ? 4 * sp_cols(tref_sp(cur.a), tref_slot(cur.a)) : sp_cols(tref_sp(cur.a), tref_slot(cur.a));
            if (int'(cur.imm[31:16]) > sp_rows(tref_sp(cur.a), tref_slot(cur.a)) || ncols > c || rb > ROWBUF_BYTES ||
                (dd.dtype == DT_F4_R16 && ncols % 16 != 0)) set_fault(F_BAD_SHAPE);
            else if (!(dd.dtype == DT_F32 || dd.dtype == DT_I32 || dd.dtype == DT_BF16 || dd.dtype == DT_F4_R16)) set_fault(F_BAD_QUALIFIER);
            else st <= S_ROW;
          end
        end
        S_ROW: begin
          if (i >= rows) st <= S_DRAIN;
          else if (is_idx && idx_cg != (i / BANKS)) st <= S_IDX_RD;
          else if (is_idx && idx_cur == -1) i <= i + 1;
          else if (is_idx && (idx_cur < 0 || idx_cur >= $signed(d.shape0))) set_fault(F_IDX_RANGE);
          else if (alloc_now) begin
            cur_slot <= as; alloc_cnt <= alloc_cnt + 1'b1; alloc_pos <= alloc_pos + (POS_W+1)'(n_alloc);
            cg <= 0; asm_valid <= 0; st <= S_RD;
          end
          // else: no free slot or ring space yet; retry next cycle
        end
        S_IDX_RD: if (sp_grant) begin idx_cg <= i / BANKS; st <= S_IDX_WAIT; end
        S_IDX_WAIT: if (sp_rvalid) begin
          for (int k = 0; k < BANKS; k++) idx_slice[k] <= sp_rdata[k];
          st <= S_ROW;
        end
        S_RD: if (sp_grant) begin                         // one slice read per cycle; the response lands next cycle
          pend <= 1'b1; pend_cg <= cg; pend_slot <= cur_slot; pend_last <= (cg + 1 >= n_cg);
          if (is_r16) st <= S_Q_WAIT;
          else if (cg + 1 >= n_cg) begin i <= i + 1; st <= S_ROW; end
          else cg <= cg + 1;
        end
        // ---- f4.r16: quantise the slice ----
        S_Q_WAIT: if (sp_rvalid) begin qs <= sp_rdata; st <= S_Q_AMAX; end
        S_Q_AMAX: begin
          logic [30:0] m; logic nn;
          m = '0; nn = 1'b0;
          for (int e = 0; e < BANKS; e++) begin
            logic isnan;
            isnan = (qs[e][30:23] == 8'd255) && (qs[e][22:0] != '0);
            nn |= isnan;
            if (!isnan && qs[e][30:0] > m) m = qs[e][30:0];
          end
          q_amax <= m; q_nan <= nn; st <= S_Q_SCALE;
        end
        S_Q_SCALE: begin q_sc <= r16_scale_code(q_amax, q_nan); qstep <= 0; st <= S_Q_LANE; end
        S_Q_LANE: begin
          for (int l = 0; l < R16_LANES; l += 2) q_nib[(qstep * R16_LANES + l) * 4 +: 8] <= {q_code[l + 1], q_code[l]};
          if (qstep + 1 >= R16_STEPS) st <= S_Q_PUT; else qstep <= qstep + 1;
        end
        S_Q_PUT: begin                                    // put_v: the 8 nibble bytes of slice cg (see the assembler)
          sc_bytes[cg % 32] <= q_sc;
          if (cg + 1 >= n_cg) begin fl <= 0; st <= S_FLUSH; end else begin cg <= cg + 1; st <= S_RD; end
        end
        S_FLUSH: begin                                    // scale chunks, the open beat, zero beats (see put_v / flush_v / zero_v)
          if (fl <= n_sc_chunks) begin
            fl <= fl + 1;
            if (fl == n_sc_chunks) zidx <= asm_valid ? asm_idx + 8'd1 : 8'd0;
          end else if (int'(zidx) < n_beats) zidx <= zidx + 8'd1;
          else begin i <= i + 1; st <= S_ROW; end
        end
        S_DRAIN: if (!w_pending && wst == W_IDLE) st <= S_DONE;
        S_DONE: st <= S_IDLE;
        S_FAULT: ;
        default: st <= S_IDLE;
      endcase
      // ---- the assembler state ----
      if (put_v) begin
        asm_beat <= put_merged; asm_idx <= put_idx; asm_valid <= 1'b1;
      end else if (flush_v) asm_valid <= 1'b0;
      // ---- writer ----
      case (wst)
        W_IDLE: if (w_pending && rec_done[ws] && st != S_IDLE && st != S_FAULT) begin beat <= 0; wst <= W_BEATS; end
        W_BEATS: begin
          if (beat + 1 >= int'(rec_nbeats[ws])) begin
            wst <= W_IDLE; wr_cnt <= wr_cnt + 1'b1;
            free_pos <= free_pos + (POS_W+1)'((int'(rec_nbeats[ws]) + RB_BANKS - 1) / RB_BANKS * RB_BANKS);
          end
          beat <= beat + 1;
        end
        default: wst <= W_IDLE;
      endcase
    end
  end
endmodule
