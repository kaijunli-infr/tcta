// TCTA core v0: device parameters (ISA §11, v1 KV260 column) and the internal instruction record.
// Config C is the same design with different parameter values (rtl/README.md §1).
package tcta_pkg;
  import tcta_isa_pkg::*;

  // ---- tile shapes (ISA §2.2) ----
  localparam int T_M  = 64;
  localparam int T_K  = 128;
  localparam int T_N  = 128;
  localparam int T_V  = 64;
  localparam int T_VR = 512;
  localparam int T_S  = 128;

  // ---- slots per scratchpad (v1 KV260; ACC = 2 + one wide of 4 = 6 indices) ----
  localparam int N_INP   = 2;
  localparam int N_WGT   = 4;
  localparam int N_ACC   = 6;
  localparam int N_VEC   = 16;
  localparam int N_STATE = 1;
  localparam int VEC_ROW_SLOT0 = 8;        // VEC slots >= this are row tiles [1, T_VR]

  // ---- units ----
  localparam int MAC_ROWS  = 16;
  localparam int MAC_COLS  = 16;
  localparam int BANKS     = 16;           // = MAC_COLS = VEC_LANES: one row slice per access
  localparam int VEC_LANES = 16;
  localparam int SFU_LANES = 4;
  localparam int DMA_BYTES = 16;           // memory port width (bytes per beat)
  localparam int FIFO_DEPTH = 8;
  localparam int N_REGS   = 16;
  localparam int IQ_DEPTH = 16;            // dispatched-but-unissued instructions per queue
  localparam int RQ_DEPTH = 8;             // issued-but-uncompleted work + trailing pushes per queue
  localparam int N_DESC   = 256;
  localparam int DESC_W   = 384;           // 48-byte descriptor (ISA §12)
  localparam int ADDR_W   = 48;            // DRAM address bits used by the memory model

  // ---- scratchpad geometry: bank = column mod BANKS, word = one element, address = (slot, row, colgroup) ----
  localparam int INP_CG   = T_K / BANKS;   // column groups per row
  localparam int WGT_CG   = T_N / BANKS;
  localparam int ACC_CG   = T_N / BANKS;
  localparam int VEC_CG   = T_V / BANKS;
  localparam int VECR_CG  = T_VR / BANKS;
  localparam int STATE_CG = T_S / BANKS;
  localparam int INP_WORDS   = N_INP * T_M * INP_CG;
  localparam int WGT_WORDS   = N_WGT * T_K * WGT_CG;
  localparam int ACC_WORDS   = N_ACC * T_M * ACC_CG;
  localparam int VEC_WORDS   = VEC_ROW_SLOT0 * T_M * VEC_CG + (N_VEC - VEC_ROW_SLOT0) * VECR_CG;
  localparam int STATE_WORDS = N_STATE * T_S * STATE_CG;
  localparam int SP_AW = 14;               // address width that covers every scratchpad (max 4096 words -> 12; margin)

  // ---- scratchpad ids (= tile reference [15:12]) ----
  typedef logic [3:0] sp_t;

  // ---- fault codes (ISA §10) ----
  typedef enum logic [3:0] {
    F_NONE = 4'd0, F_TOKEN_LEAK = 4'd1, F_TOKEN_TIMEOUT = 4'd2, F_UNTOKENED_READ = 4'd3, F_UNTOKENED_WRITE = 4'd4,
    F_VALID_MISMATCH = 4'd5, F_ACC_NOT_CONVERTED = 4'd6, F_IDX_RANGE = 4'd7, F_IDX_SHORT = 4'd8, F_PAGE_FAULT = 4'd9,
    F_BAD_EDGE = 4'd10, F_BAD_QUALIFIER = 4'd11, F_BAD_SHAPE = 4'd12, F_BAD_OPCODE = 4'd13
  } fault_t;

  // ---- the instruction record a queue holds (decoded §12 word plus the front end's loop-variant results) ----
  typedef struct packed {
    logic [2:0]         q;          // queue (tcta_isa_pkg::Q_*)
    logic [5:0]         op;
    logic [6:0]         qual;
    logic [15:0]        flags;
    logic [15:0]        a;
    logic [15:0]        b;
    logic [15:0]        c;
    logic [15:0]        d;
    logic [31:0]        imm;
    logic signed [47:0] base_off;   // added to the descriptor's base (region.base increments)
    logic signed [31:0] scales_off; // added to the descriptor's scales_base
    logic [31:0]        pc;
    logic               finish;     // FINISH marker: retire when all earlier work is done
  } instr_t;

  // ---- tile metadata kept beside each scratchpad slot (the simulator's TileState.valid / pready) ----
  typedef struct packed {
    logic [9:0] rows;               // rows_valid (<= 512)
    logic [9:0] cols;               // cols_valid
    logic       pready;             // ACC: BF16 payload written in place (ISA §6.3)
  } tmeta_t;

  // ---- per-slot token-check state (debug block, ISA §8.4; TOKEN_CHECK parameter) ----
  localparam int TC_TIME_W = 32;

  // ---- scratchpad clients (fixed priority in this order) and the request / response records ----
  // The two MMA operand fetches are read-only and have the two highest priorities, so they are always granted
  // (two reads fit the two ports; a write can only come from a lower-priority client): the systolic array never
  // stalls. Every other client retries when it loses.
  localparam int N_CLIENTS = 7;
  localparam int CL_MMA_A     = 0;        // read only
  localparam int CL_MMA_B     = 1;        // read only
  localparam int CL_MMA_WB    = 2;        // read (mask, old) and write
  localparam int CL_VEC_RD    = 3;        // read only
  localparam int CL_VEC_WR    = 4;        // write only
  localparam int CL_DMA_LOAD  = 5;        // write (destination) and read (index tile)
  localparam int CL_DMA_STORE = 6;        // read only
  // Which clients each scratchpad serves (phase 8 L2), from the ISA's access table (§1, §2.2) and the operand
  // rules of §5, readers and writers separately: LOAD.2D writes INP/WGT/VEC/STATE and LOAD.IDX INP/WGT/ACC/VEC with
  // the index tile read from VEC; MMA reads a from INP/ACC (§6.3), b from WGT, the mask from VEC/ACC and the old
  // accumulator from ACC/STATE, and writes ACC/STATE; VEC reads ACC/VEC/STATE and writes ACC/VEC/STATE plus INP/WGT
  // (VCVT); STORE reads ACC/VEC. A pad's arbiter covers the union (21 of the 35 pairs), its write-data muxes only the
  // writers (12 pairs) and its response path only the readers (12 pairs); a request outside the table is a
  // simulation error in tcta_core (it would otherwise never be granted and end as TOKEN_TIMEOUT).
  localparam logic [N_CLIENTS-1:0] SP_READERS [5] = '{
    N_CLIENTS'(1 << CL_MMA_A),                                                              // INP
    N_CLIENTS'(1 << CL_MMA_B),                                                              // WGT
    N_CLIENTS'((1 << CL_MMA_A) | (1 << CL_MMA_WB) | (1 << CL_VEC_RD) | (1 << CL_DMA_STORE)), // ACC
    N_CLIENTS'((1 << CL_MMA_WB) | (1 << CL_VEC_RD) | (1 << CL_DMA_LOAD) | (1 << CL_DMA_STORE)), // VEC
    N_CLIENTS'((1 << CL_MMA_WB) | (1 << CL_VEC_RD))                                         // STATE
  };
  localparam logic [N_CLIENTS-1:0] SP_WRITERS [5] = '{
    N_CLIENTS'((1 << CL_VEC_WR) | (1 << CL_DMA_LOAD)),                                      // INP
    N_CLIENTS'((1 << CL_VEC_WR) | (1 << CL_DMA_LOAD)),                                      // WGT
    N_CLIENTS'((1 << CL_MMA_WB) | (1 << CL_VEC_WR) | (1 << CL_DMA_LOAD)),                   // ACC
    N_CLIENTS'((1 << CL_VEC_WR) | (1 << CL_DMA_LOAD)),                                      // VEC
    N_CLIENTS'((1 << CL_MMA_WB) | (1 << CL_VEC_WR) | (1 << CL_DMA_LOAD))                    // STATE
  };
  localparam logic [N_CLIENTS-1:0] SP_CLIENTS [5] = '{
    SP_READERS[0] | SP_WRITERS[0], SP_READERS[1] | SP_WRITERS[1], SP_READERS[2] | SP_WRITERS[2],
    SP_READERS[3] | SP_WRITERS[3], SP_READERS[4] | SP_WRITERS[4]
  };

  typedef logic [BANKS-1:0][31:0]    slice_t;   // one row slice: one element per bank (16-bit banks use [31:16])

  typedef struct packed {
    logic              valid;
    logic              we;
    sp_t               sp;        // which scratchpad
    logic [11:0]       slot;      // the slot the access belongs to (token checker / debug only)
    logic [SP_AW-1:0]  addr;      // the word address, the same in every bank (a request is one row slice; phase 8 L2
                                  // dropped the per-bank addresses, which no client ever set differently)
    logic [BANKS-1:0]  ben;       // bank enables (write) / don't-care (read)
    slice_t            wdata;
  } sp_req_t;

  // a row-slice request: columns cg*BANKS .. cg*BANKS+BANKS-1 of one row, the same address in every bank
  function automatic sp_req_t slice_req(input sp_t sp, input logic [11:0] slot, input int row, input int cg,
                                        input logic we, input logic [BANKS-1:0] ben, input slice_t wdata);
    sp_req_t r;
    r.valid = 1'b1; r.we = we; r.sp = sp; r.slot = slot; r.ben = ben; r.wdata = wdata;
    r.addr = sp_addr(sp, slot, row, cg);
    return r;
  endfunction

  // the number of row slices that cover n columns
  function automatic int n_slices(input int n); return (n + BANKS - 1) / BANKS; endfunction

  // bank enables for the columns [0, n - cg*BANKS) of slice cg
  function automatic logic [BANKS-1:0] slice_ben(input int n, input int cg);
    logic [BANKS-1:0] m;
    for (int b = 0; b < BANKS; b++) m[b] = (cg * BANKS + b) < n;
    return m;
  endfunction

  // ---- memory port (testbench DRAM model): byte-addressed 16-byte beats, any alignment, fixed latency ----
  localparam int MEM_W = DMA_BYTES * 8;
  localparam int ROWBUF_BYTES = 2048;      // the longest DRAM row a LOAD/STORE handles in one piece
  localparam int ROWBUF_BEATS = ROWBUF_BYTES / DMA_BYTES;
  localparam int DMA_ROWS_IN_FLIGHT = 8;   // LOAD / STORE row records in flight (phase 8 K1); the row-buffer ring holds ROWBUF_BEATS beats

  // ---- descriptor fields (ISA §12, 48 bytes little-endian) ----
  typedef struct packed {
    logic [15:0] scales_stride;
    logic [7:0]  ndim;
    logic [7:0]  dtype;
    logic [31:0] scale_t;
    logic [31:0] shape2, shape1, shape0;
    logic [31:0] stride2, stride1, stride0;
    logic [63:0] scales_base;
    logic [63:0] base;
  } desc_t;
  function automatic desc_t unpack_desc(input logic [DESC_W-1:0] d);
    desc_t r;
    r.base = d[63:0]; r.scales_base = d[127:64];
    r.stride0 = d[159:128]; r.stride1 = d[191:160]; r.stride2 = d[223:192];
    r.shape0 = d[255:224]; r.shape1 = d[287:256]; r.shape2 = d[319:288];
    r.scale_t = d[351:320]; r.dtype = d[359:352]; r.ndim = d[367:360]; r.scales_stride = d[383:368];
    return r;
  endfunction

  // bytes per element in DRAM for a dtype (f4 formats: half a byte, handled by the DMA)
  function automatic int el_bytes(input logic [7:0] dtype);
    case (dtype)
      DT_BF16: return 2;
      DT_F32, DT_I32: return 4;
      DT_F8_T, DT_F8_B32: return 1;
      default: return 0;             // f4.b32 / f4.r16: nibbles
    endcase
  endfunction
  // dtypes.row_bytes_for
  function automatic int row_bytes_for(input logic [7:0] dtype, input int ncols);
    case (dtype)
      DT_BF16: return ncols * 2;
      DT_F32, DT_I32: return ncols * 4;
      DT_F8_T, DT_F8_B32: return ncols;
      DT_F4_B32: return ncols / 2;
      DT_F4_R16: return ncols / 2 + ncols / 16;
      default: return 0;
    endcase
  endfunction

  // ---- helpers ----
  function automatic sp_t tref_sp(input logic [15:0] t);  return t[15:12]; endfunction
  function automatic logic [11:0] tref_slot(input logic [15:0] t); return t[11:0]; endfunction

  function automatic int sp_rows(input sp_t sp, input logic [11:0] slot);
    case (sp)
      SP_INP:   return T_M;
      SP_WGT:   return T_K;
      SP_ACC:   return T_M;
      SP_VEC:   return (int'(slot) >= VEC_ROW_SLOT0) ? 1 : T_M;
      SP_STATE: return T_S;
      default:  return 0;
    endcase
  endfunction

  function automatic int sp_cols(input sp_t sp, input logic [11:0] slot);
    case (sp)
      SP_INP:   return T_K;
      SP_WGT:   return T_N;
      SP_ACC:   return T_N;
      SP_VEC:   return (int'(slot) >= VEC_ROW_SLOT0) ? T_VR : T_V;
      SP_STATE: return T_S;
      default:  return 0;
    endcase
  endfunction

  function automatic int sp_cg(input sp_t sp, input logic [11:0] slot);
    return sp_cols(sp, slot) / BANKS;
  endfunction

  // word address of (slot, row, colgroup) inside a scratchpad's banks
  function automatic logic [SP_AW-1:0] sp_addr(input sp_t sp, input logic [11:0] slot, input int row, input int cg);
    int base;
    case (sp)
      SP_INP:   base = int'(slot) * T_M * INP_CG;
      SP_WGT:   base = int'(slot) * T_K * WGT_CG;
      SP_ACC:   base = int'(slot) * T_M * ACC_CG;
      SP_VEC:   base = (int'(slot) >= VEC_ROW_SLOT0) ? (VEC_ROW_SLOT0 * T_M * VEC_CG + (int'(slot) - VEC_ROW_SLOT0) * VECR_CG)
                                                       : int'(slot) * T_M * VEC_CG;
      SP_STATE: base = int'(slot) * T_S * STATE_CG;
      default:  base = 0;
    endcase
    return SP_AW'(base + row * sp_cg(sp, slot) + cg);
  endfunction

  // the slot a word address belongs to (inverse of sp_addr on the slot part; used by the token checker)
  function automatic logic [11:0] addr_slot(input sp_t sp, input logic [SP_AW-1:0] addr);
    int a; a = int'(addr);
    case (sp)
      SP_INP:   return 12'(a / (T_M * INP_CG));
      SP_WGT:   return 12'(a / (T_K * WGT_CG));
      SP_ACC:   return 12'(a / (T_M * ACC_CG));
      SP_VEC:   return (a < VEC_ROW_SLOT0 * T_M * VEC_CG) ? 12'(a / (T_M * VEC_CG))
                                                           : 12'(VEC_ROW_SLOT0 + (a - VEC_ROW_SLOT0 * T_M * VEC_CG) / VECR_CG);
      SP_STATE: return 12'(a / (T_S * STATE_CG));
      default:  return 12'd0;
    endcase
  endfunction

  // flat slot index for the metadata table (INP, WGT, ACC, VEC, STATE in that order)
  localparam int N_SLOTS = N_INP + N_WGT + N_ACC + N_VEC + N_STATE;
  function automatic int slot_index(input sp_t sp, input logic [11:0] slot);
    case (sp)
      SP_INP:   return int'(slot);
      SP_WGT:   return N_INP + int'(slot);
      SP_ACC:   return N_INP + N_WGT + int'(slot);
      SP_VEC:   return N_INP + N_WGT + N_ACC + int'(slot);
      SP_STATE: return N_INP + N_WGT + N_ACC + N_VEC + int'(slot);
      default:  return 0;
    endcase
  endfunction

  // ---- MMA fixed-point accumulation (phase 8 K0.3, erratum E5; mirrors FIX_* in sim/tcta/machine.py) ----
  // An output element is the exact integer sum of pre-shifted BF16 significand products in FIX_B_LEVELS 48-bit
  // DSP48E2 accumulators: A' = +-(ma << (FIX_SA - da)) (ma >> (da - FIX_SA) truncated beyond), da = the row's
  // maximum exponent - ea; B level l = +-(mb << ((l+1) * FIX_SB - db)) for db in (l * FIX_SB, (l+1) * FIX_SB] (from 0 on
  // level 0, truncated beyond on the last level). value = (acc_0 * 2^FIX_SB + acc_1) * 2^(ea_max + eb_max - 268 -
  // FIX_TOP - FIX_SB), rounded once to binary32 in the writeback.
  localparam int FIX_BITS     = 48;
  localparam int FIX_K_MAX    = (T_K > T_S) ? T_K : T_S;                 // K per MMA <= the INP / STATE tile width
  localparam int FIX_TOP      = (FIX_BITS - 1) - 16 - $clog2(FIX_K_MAX); // 24: no accumulator overflow for K <= FIX_K_MAX
  localparam int FIX_SB       = 9;                                       // B pre-shift range per level (18-bit B port)
  localparam int FIX_SA       = FIX_TOP - FIX_SB;                        // 15: A pre-shift range (27-bit A port)
  localparam int FIX_B_LEVELS = 2;
  localparam int FIX_AW       = 8 + FIX_SA + 1;                          // 24: signed A operand
  localparam int FIX_BW       = 8 + FIX_SB + 1;                          // 18: signed B operand
  localparam int FIX_ACC_W    = FIX_BITS + (FIX_B_LEVELS - 1) * FIX_SB;  // 57: the combined accumulator
  localparam int FIX_W_W      = 10;                                      // signed window exponent (-299 .. 207)
  localparam int FIX_W_BIAS   = 2 * 127 + 2 * 7 + FIX_TOP + (FIX_B_LEVELS - 1) * FIX_SB;   // 301

  // decoded BF16: effective exponent e (1 for zero / subnormal), significand m with the hidden bit (the fraction
  // alone for e == 0), the IEEE classes
  typedef struct packed { logic sgn; logic nan; logic inf; logic zero; logic [7:0] e; logic [7:0] m; } bf16_f_t;
  function automatic bf16_f_t bf16_fields(input logic [15:0] x);
    bf16_f_t r;
    r.sgn = x[15]; r.e = (x[14:7] == 8'd0) ? 8'd1 : x[14:7];
    r.m = {x[14:7] != 8'd0, x[6:0]};
    r.nan = (x[14:7] == 8'hff) && (x[6:0] != 7'd0);
    r.inf = (x[14:7] == 8'hff) && (x[6:0] == 7'd0);
    r.zero = (x[14:0] == 15'd0);
    return r;
  endfunction
  // the operands as the array sees them: pre-shifted significands with the sign applied, plus the IEEE classes
  typedef struct packed { logic signed [FIX_AW-1:0] v; logic nan; logic inf; logic zero; logic sgn; } fix_a_t;
  typedef struct packed { logic [FIX_B_LEVELS-1:0][FIX_BW-1:0] v; logic nan; logic inf; logic zero; logic sgn; } fix_b_t;

endpackage
