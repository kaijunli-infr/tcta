// Front end (ISA §6, §12): reads the instruction stream word by word, expands LOOP / ENDLOOP (a stack of
// LOOP_DEPTH levels, static or register count = ceil(R / step), zero count skips the body), applies the
// extension words' loop-variant increments (field += i_L * step, slot += i_L * step, slot += i_L mod step),
// executes SET*, dispatches everything else into the queues' instruction buffers (back-pressure when full)
// and stops at FINISH. `LOOP r` waits until no VCOUNT writing R[r] is outstanding. The instruction index (pc)
// counts instruction words, not extension words, so it matches the simulator's trace and fault pcs.
module front_end
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
#(
  parameter int LOOP_DEPTH = 4
) (
  input  logic         clk,
  input  logic         rst_n,
  input  logic         start,
  // instruction stream (word index -> 128-bit word, one cycle later)
  output logic [31:0]  is_addr,
  input  logic [127:0] is_data,
  input  logic [31:0]  n_words,
  // dispatch
  output logic         dq_valid,
  output logic [2:0]   dq_q,
  output instr_t       dq_ins,
  input  logic         iq_full [5],
  // registers
  output logic [31:0]  regs [N_REGS],
  input  logic         reg_we,
  input  logic [3:0]   reg_idx,
  input  logic [31:0]  reg_wdata,
  // status
  output logic         done,           // FINISH reached (everything before it dispatched)
  output logic         progress,       // did something this cycle
  output logic         fault,
  output fault_t       fault_code,
  output logic [31:0]  fault_pc
);
  typedef enum logic [3:0] {F_IDLE, F_FETCH, F_DEC, F_LOOPR, F_EXT_FETCH, F_EXT_DEC, F_APPLY, F_MOD, F_DISP, F_SKIP_FETCH,
                            F_SKIP_DEC, F_DONE, F_FAULT} state_t;
  state_t st;
  logic [31:0] wp, pc;                 // word index, instruction index
  logic [127:0] w;                     // the instruction word being decoded
  instr_t      ins;                    // the record being built
  logic [1:0]  n_ext, ext_i;           // extension words expected / consumed
  logic [95:0] ents;                   // the current extension word's entries (entry0 at [95:48])
  logic [1:0]  n_ent, ent_i;
  // loop stack
  logic [31:0] l_wp [LOOP_DEPTH], l_pc [LOOP_DEPTH], l_cnt [LOOP_DEPTH], l_i [LOOP_DEPTH];
  int          depth;
  int          skip_depth;
  // LOOP r count and mod computation
  logic signed [31:0] rem; logic [31:0] cnt; logic [31:0] lstep;
  logic [31:0] mod_q; logic [31:0] mod_step;
  // reg pending counters
  logic [7:0]  reg_pending [N_REGS];

  // ---- decode helpers ----
  logic [2:0] w_q; logic [5:0] w_op; logic [6:0] w_qual; logic [15:0] w_flags;
  always_comb begin
    w_q = w[127:125]; w_op = w[124:119]; w_qual = w[118:112]; w_flags = w[111:96];
  end
  function automatic logic op_known(input logic [5:0] op); return int'(op) < N_OPCODES; endfunction

  // ---- current extension entry ----
  logic [47:0] ent; logic [1:0] e_kind; logic [5:0] e_field; logic [3:0] e_lvl; logic signed [31:0] e_step;
  always_comb begin
    ent = (ent_i == 0) ? ents[95:48] : ents[47:0];
    e_kind = ent[47:46]; e_field = ent[45:40]; e_lvl = ent[39:36]; e_step = $signed(ent[31:0]);
  end

  // apply delta to the field at position pos of the record
  function automatic instr_t apply_delta(input instr_t r, input logic [5:0] op, input logic [5:0] field, input logic [1:0] kind,
                                         input logic signed [31:0] delta);
    instr_t o; logic [3:0] pos;
    o = r;
    if (field == IF_REGION_BASE) begin o.base_off = r.base_off + 48'(delta); return o; end
    if (field == IF_REGION_SCALES_BASE) begin o.scales_off = r.scales_off + delta; return o; end
    pos = incr_pos(op, field);
    if (kind == IK_ADD) begin
      case (pos)
        4'd0: o.a = r.a + delta[15:0];
        4'd1: o.b = r.b + delta[15:0];
        4'd2: o.c = r.c + delta[15:0];
        4'd3: o.d = r.d + delta[15:0];
        4'd4: o.imm = r.imm + delta;
        4'd5: o.imm[31:16] = r.imm[31:16] + delta[15:0];
        4'd6: o.imm[15:0] = r.imm[15:0] + delta[15:0];
        4'd7: o.b[7:0] = r.b[7:0] + delta[7:0];
        4'd8: o.b[15:8] = r.b[15:8] + delta[7:0];
        default: ;
      endcase
    end else begin                                  // slot / mod: the low 12 bits of a tile reference
      case (pos)
        4'd0: o.a[11:0] = r.a[11:0] + delta[11:0];
        4'd1: o.b[11:0] = r.b[11:0] + delta[11:0];
        4'd2: o.c[11:0] = r.c[11:0] + delta[11:0];
        4'd3: o.d[11:0] = r.d[11:0] + delta[11:0];
        default: ;
      endcase
    end
    return o;
  endfunction

  // ---- outputs ----
  always_comb begin
    is_addr = (st == F_FETCH || st == F_SKIP_FETCH) ? wp : (st == F_EXT_FETCH ? wp : wp);
    dq_valid = (st == F_DISP) && !iq_full[ins.q];
    dq_q = ins.q; dq_ins = ins;
    done = (st == F_DONE);
    progress = dq_valid || (st == F_DEC && w_q == Q_FE) || (st == F_FETCH) || (st == F_EXT_FETCH) || (st == F_APPLY) ||
               (st == F_SKIP_FETCH) || (st == F_LOOPR) || (st == F_MOD);
  end

  int lvl_i;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= F_IDLE; wp <= 0; pc <= 0; w <= '0; ins <= '0; n_ext <= 0; ext_i <= 0; ents <= '0; n_ent <= 0; ent_i <= 0;
      depth <= 0; skip_depth <= 0; rem <= 0; cnt <= 0; lstep <= 1; mod_q <= 0; mod_step <= 1;
      fault <= 0; fault_code <= F_NONE; fault_pc <= 0; lvl_i <= 0;
      for (int k = 0; k < LOOP_DEPTH; k++) begin l_wp[k] <= 0; l_pc[k] <= 0; l_cnt[k] <= 0; l_i[k] <= 0; end
      for (int k = 0; k < N_REGS; k++) begin regs[k] <= 0; reg_pending[k] <= 0; end
    end else begin
      // register writes from VCOUNT
      if (reg_we) begin
        regs[reg_idx] <= reg_wdata;
        if (!(st == F_DISP && dq_valid && ins.op == OP_VCOUNT && ins.a[3:0] == reg_idx)) reg_pending[reg_idx] <= reg_pending[reg_idx] - 1;
      end
      case (st)
        F_IDLE: if (start) begin wp <= 0; pc <= 0; depth <= 0; st <= F_FETCH; end
        F_FETCH: begin
          if (wp >= n_words) begin fault <= 1; fault_code <= F_BAD_OPCODE; fault_pc <= pc; st <= F_FAULT; end   // ran off the stream
          else st <= F_DEC;
        end
        F_DEC: begin
          w <= is_data;
          begin
            logic [127:0] x; logic [2:0] q; logic [5:0] op; logic [6:0] qual; logic [15:0] flags;
            x = is_data; q = x[127:125]; op = x[124:119]; qual = x[118:112]; flags = x[111:96];
            if (op == OP_EXT || !op_known(op) || q > Q_FE) begin
              fault <= 1; fault_code <= F_BAD_OPCODE; fault_pc <= pc; st <= F_FAULT;
            end else if (q == Q_FE) begin
              case (op)
                OP_LOOP: begin
                  if (qual == QUAL_LOOP_FORM_R) begin
                    if (reg_pending[x[83:80]] != 0) st <= F_FETCH;         // wait: refetch the word later
                    else begin
                      rem <= $signed(regs[x[83:80]]); cnt <= 0; lstep <= (x[63:48] == 0) ? 32'd1 : 32'(x[63:48]);
                      st <= F_LOOPR;
                    end
                  end else begin
                    if (x[31:0] == 0) begin skip_depth <= 0; wp <= wp + 1; pc <= pc + 1; st <= F_SKIP_FETCH; end
                    else if (depth >= LOOP_DEPTH) begin fault <= 1; fault_code <= F_BAD_SHAPE; fault_pc <= pc; st <= F_FAULT; end
                    else begin
                      l_wp[depth] <= wp + 1; l_pc[depth] <= pc + 1; l_cnt[depth] <= x[31:0]; l_i[depth] <= 0;
                      depth <= depth + 1; wp <= wp + 1; pc <= pc + 1; st <= F_FETCH;
                    end
                  end
                end
                OP_ENDLOOP: begin
                  if (depth == 0) begin fault <= 1; fault_code <= F_BAD_OPCODE; fault_pc <= pc; st <= F_FAULT; end
                  else if (l_i[depth-1] + 1 < l_cnt[depth-1]) begin
                    l_i[depth-1] <= l_i[depth-1] + 1; wp <= l_wp[depth-1]; pc <= l_pc[depth-1]; st <= F_FETCH;
                  end else begin
                    depth <= depth - 1; wp <= wp + 1; pc <= pc + 1; st <= F_FETCH;
                  end
                end
                OP_FINISH: st <= F_DONE;
                OP_SETSCALE, OP_SETBASE, OP_SETMASK: begin
                  regs[x[83:80]] <= x[31:0]; wp <= wp + 1; pc <= pc + 1; st <= F_FETCH;
                end
                default: begin fault <= 1; fault_code <= F_BAD_OPCODE; fault_pc <= pc; st <= F_FAULT; end
              endcase
            end else begin
              // a work / sync instruction: build the record
              ins <= '{q: q, op: op, qual: qual, flags: flags, a: x[95:80], b: x[79:64], c: x[63:48], d: x[47:32], imm: x[31:0],
                       base_off: 48'sd0, scales_off: 32'sd0, pc: pc, finish: 1'b0};
              n_ext <= flags[15:14]; ext_i <= 0;
              if (flags[15:14] == 0) st <= F_DISP;
              else begin wp <= wp + 1; st <= F_EXT_FETCH; end
            end
          end
        end
        F_LOOPR: begin                                  // count = ceil(R / step) for R > 0, by repeated subtraction
          if (rem > 0) begin rem <= rem - $signed(lstep); cnt <= cnt + 1; end
          else begin
            if (cnt == 0) begin skip_depth <= 0; wp <= wp + 1; pc <= pc + 1; st <= F_SKIP_FETCH; end
            else if (depth >= LOOP_DEPTH) begin fault <= 1; fault_code <= F_BAD_SHAPE; fault_pc <= pc; st <= F_FAULT; end
            else begin
              l_wp[depth] <= wp + 1; l_pc[depth] <= pc + 1; l_cnt[depth] <= cnt; l_i[depth] <= 0;
              depth <= depth + 1; wp <= wp + 1; pc <= pc + 1; st <= F_FETCH;
            end
          end
        end
        F_EXT_FETCH: st <= F_EXT_DEC;
        F_EXT_DEC: begin
          if (is_data[124:119] != OP_EXT) begin fault <= 1; fault_code <= F_BAD_OPCODE; fault_pc <= pc; st <= F_FAULT; end
          else begin
            ents <= is_data[95:0]; n_ent <= (is_data[118:112] == 7'd2) ? 2'd2 : 2'd1; ent_i <= 0; st <= F_APPLY;
          end
        end
        F_APPLY: begin
          // level e_lvl counted from the innermost loop; inactive levels are skipped
          if (int'(e_lvl) < depth) begin
            int i_val; i_val = int'(l_i[depth - 1 - int'(e_lvl)]);
            if (e_kind == IK_MOD) begin
              mod_q <= 32'(i_val); mod_step <= (e_step == 0) ? 32'd1 : 32'(e_step); st <= F_MOD;
            end else begin
              ins <= apply_delta(ins, ins.op, e_field, e_kind, 32'(i_val) * e_step);
              if (ent_i + 1 < n_ent) ent_i <= ent_i + 1;
              else if (ext_i + 1 < n_ext) begin ext_i <= ext_i + 1; wp <= wp + 1; st <= F_EXT_FETCH; end
              else st <= F_DISP;
            end
          end else begin
            if (ent_i + 1 < n_ent) ent_i <= ent_i + 1;
            else if (ext_i + 1 < n_ext) begin ext_i <= ext_i + 1; wp <= wp + 1; st <= F_EXT_FETCH; end
            else st <= F_DISP;
          end
        end
        F_MOD: begin                                    // i mod step by repeated subtraction
          if (mod_q >= mod_step) mod_q <= mod_q - mod_step;
          else begin
            ins <= apply_delta(ins, ins.op, e_field, IK_SLOT, $signed(mod_q));
            if (ent_i + 1 < n_ent) begin ent_i <= ent_i + 1; st <= F_APPLY; end
            else if (ext_i + 1 < n_ext) begin ext_i <= ext_i + 1; wp <= wp + 1; st <= F_EXT_FETCH; end
            else st <= F_DISP;
          end
        end
        F_DISP: if (dq_valid) begin
          if (ins.op == OP_VCOUNT && !(reg_we && reg_idx == ins.a[3:0])) reg_pending[ins.a[3:0]] <= reg_pending[ins.a[3:0]] + 1;
          wp <= wp + 1; pc <= pc + 1; st <= F_FETCH;
        end
        F_SKIP_FETCH: begin
          if (wp >= n_words) begin fault <= 1; fault_code <= F_BAD_OPCODE; fault_pc <= pc; st <= F_FAULT; end
          else st <= F_SKIP_DEC;
        end
        F_SKIP_DEC: begin
          logic [5:0] op; op = is_data[124:119];
          if (op == OP_EXT) begin wp <= wp + 1; st <= F_SKIP_FETCH; end
          else if (op == OP_LOOP && is_data[127:125] == Q_FE) begin skip_depth <= skip_depth + 1; wp <= wp + 1; pc <= pc + 1; st <= F_SKIP_FETCH; end
          else if (op == OP_ENDLOOP && is_data[127:125] == Q_FE) begin
            if (skip_depth == 0) begin wp <= wp + 1; pc <= pc + 1; st <= F_FETCH; end
            else begin skip_depth <= skip_depth - 1; wp <= wp + 1; pc <= pc + 1; st <= F_SKIP_FETCH; end
          end else begin wp <= wp + 1; pc <= pc + 1; st <= F_SKIP_FETCH; end
        end
        F_DONE: ;
        F_FAULT: ;
        default: st <= F_IDLE;
      endcase
    end
  end
endmodule
