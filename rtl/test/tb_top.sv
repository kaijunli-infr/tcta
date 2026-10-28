// Testbench (Verilator) for tcta_core: loads a vector directory (rtl/test/gen_vectors.py), models DRAM (byte
// addressed, DMA_BYTES per beat at any alignment, fixed read latency), the instruction stream and the
// descriptor table, runs to FINISH / fault / the cycle cap, and writes status.txt, rtl_trace.txt (one line per
// queue event: cycle queue event op pc), dram_out.hex (the pages of pages.txt, same format as expect_dram.hex)
// and sp_dump.txt (every scratchpad word: sp bank word hex). The clock comes from tb_main.cpp.
//   +VEC=<dir> [+MEMLAT=n] [+MAXCYC=n] [+TDEADLOCK=n] [+OUT=<dir>]
module tb_top
  import tcta_isa_pkg::*;
  import tcta_pkg::*;
(
  input logic clk
);
  localparam int DRAM_BYTES = 64 << 20;
  localparam int MAX_WORDS = 65536;
  localparam int MAX_LAT = 64;

  logic [7:0]   dram [DRAM_BYTES];
  logic [127:0] stream [MAX_WORDS];
  logic [DESC_W-1:0] descs [N_DESC];

  string  vec_dir, out_dir;
  int     mem_lat, max_cyc, n_desc, n_words_i, t_deadlock_i;
  int     tick = 0;
  logic   rst_n = 1'b0, start = 1'b0;
  logic [31:0] t_deadlock;

  // ---------------------------------------------------------------- core
  logic [31:0] is_addr; logic [127:0] is_data; logic [31:0] n_words;
  logic desc_we = 1'b0; logic [7:0] desc_widx = '0; logic [DESC_W-1:0] desc_wdata = '0;
  logic mem_rd_valid; logic [ADDR_W-1:0] mem_rd_addr; logic mem_rd_rvalid; logic [MEM_W-1:0] mem_rd_data; logic mem_rd_err;
  logic mem_wr_valid; logic [ADDR_W-1:0] mem_wr_addr; logic [MEM_W-1:0] mem_wr_data; logic [DMA_BYTES-1:0] mem_wr_be;
  logic running, finished, fault; fault_t fault_code; logic [2:0] fault_q; logic [31:0] fault_pc, cycle;
  logic tr_sync [5], tr_issue [5], tr_done [5];
  logic [5:0] tr_sync_op [5], tr_issue_op [5], tr_done_op [5];
  logic [31:0] tr_sync_pc [5], tr_issue_pc [5], tr_done_pc [5];

  tcta_core #(.TOKEN_CHECK(`TOKEN_CHECK)) u_core (
    .clk(clk), .rst_n(rst_n), .start(start), .t_deadlock(t_deadlock),
    .is_addr(is_addr), .is_data(is_data), .n_words(n_words),
    .desc_we(desc_we), .desc_widx(desc_widx), .desc_wdata(desc_wdata),
    .mem_rd_valid(mem_rd_valid), .mem_rd_addr(mem_rd_addr), .mem_rd_rvalid(mem_rd_rvalid), .mem_rd_data(mem_rd_data), .mem_rd_err(mem_rd_err),
    .mem_wr_valid(mem_wr_valid), .mem_wr_addr(mem_wr_addr), .mem_wr_data(mem_wr_data), .mem_wr_be(mem_wr_be),
    .running(running), .finished(finished), .fault(fault), .fault_code(fault_code), .fault_q(fault_q), .fault_pc(fault_pc), .cycle(cycle),
    .tr_sync(tr_sync), .tr_sync_op(tr_sync_op), .tr_sync_pc(tr_sync_pc),
    .tr_issue(tr_issue), .tr_issue_op(tr_issue_op), .tr_issue_pc(tr_issue_pc),
    .tr_done(tr_done), .tr_done_op(tr_done_op), .tr_done_pc(tr_done_pc));

  // ---------------------------------------------------------------- stream (1-cycle read)
  always_ff @(posedge clk) is_data <= (is_addr < MAX_WORDS) ? stream[is_addr] : '0;
  assign n_words = 32'(n_words_i);

  // ---------------------------------------------------------------- memory model
  logic        rq_v [MAX_LAT]; logic [ADDR_W-1:0] rq_a [MAX_LAT];
  initial for (int k = 0; k < MAX_LAT; k++) begin rq_v[k] = 0; rq_a[k] = '0; end
  always_ff @(posedge clk) begin
    // read pipeline
    for (int k = MAX_LAT - 1; k > 0; k--) begin rq_v[k] <= rq_v[k-1]; rq_a[k] <= rq_a[k-1]; end
    rq_v[0] <= mem_rd_valid && rst_n; rq_a[0] <= mem_rd_addr;
    mem_rd_rvalid <= rq_v[mem_lat - 1];
    mem_rd_err <= (rq_a[mem_lat - 1] + DMA_BYTES > DRAM_BYTES);
    for (int k = 0; k < DMA_BYTES; k++)
      mem_rd_data[k*8 +: 8] <= (rq_a[mem_lat - 1] + k < DRAM_BYTES) ? dram[rq_a[mem_lat - 1] + k] : 8'h00;
    // writes
    if (mem_wr_valid && rst_n) begin
      for (int k = 0; k < DMA_BYTES; k++)
        if (mem_wr_be[k] && mem_wr_addr + k < DRAM_BYTES) dram[mem_wr_addr + k] <= mem_wr_data[k*8 +: 8];
    end
  end

  // ---------------------------------------------------------------- names
  function automatic string qname(input int q);
    case (q) 0: return "LOAD"; 1: return "MMA"; 2: return "VEC"; 3: return "STORE"; 4: return "COMM"; default: return "FE"; endcase
  endfunction
  function automatic string opname(input logic [5:0] op);
    case (op)
      OP_PUSH: return "PUSH"; OP_POP: return "POP"; OP_LOAD_2D: return "LOAD.2D"; OP_LOAD_3D: return "LOAD.3D"; OP_LOAD_IDX: return "LOAD.IDX";
      OP_STORE_2D: return "STORE.2D"; OP_STORE_IDX: return "STORE.IDX"; OP_MMA: return "MMA"; OP_MMA_T: return "MMA.T"; OP_MMA_MASKED: return "MMA.MASKED";
      OP_VROWMAX: return "VROWMAX"; OP_VROWSUM: return "VROWSUM"; OP_VSUB: return "VSUB"; OP_VMUL: return "VMUL"; OP_VADD: return "VADD";
      OP_VMAX: return "VMAX"; OP_VNEG: return "VNEG"; OP_VSCALE_ACC: return "VSCALE_ACC"; OP_VGATE: return "VGATE"; OP_VCVT: return "VCVT";
      OP_VCOPY: return "VCOPY"; OP_VMASK: return "VMASK"; OP_VEXP2: return "VEXP2"; OP_VEXP2_SUM: return "VEXP2.SUM"; OP_VRCP: return "VRCP";
      OP_VLOG2: return "VLOG2"; OP_VSIGMOID: return "VSIGMOID"; OP_VFILL: return "VFILL"; OP_VTOPK: return "VTOPK"; OP_VCOUNT: return "VCOUNT";
      OP_VSEL: return "VSEL"; OP_SEND: return "SEND"; OP_RECV: return "RECV"; OP_RECV_ANY: return "RECV.ANY"; OP_LOOP: return "LOOP";
      OP_ENDLOOP: return "ENDLOOP"; OP_FINISH: return "FINISH"; OP_SETSCALE: return "SETSCALE"; OP_SETBASE: return "SETBASE"; OP_SETMASK: return "SETMASK";
      default: return "?";
    endcase
  endfunction
  function automatic string fname(input fault_t f);
    case (f)
      F_NONE: return "NONE"; F_TOKEN_LEAK: return "TOKEN_LEAK"; F_TOKEN_TIMEOUT: return "TOKEN_TIMEOUT"; F_UNTOKENED_READ: return "UNTOKENED_READ";
      F_UNTOKENED_WRITE: return "UNTOKENED_WRITE"; F_VALID_MISMATCH: return "VALID_MISMATCH"; F_ACC_NOT_CONVERTED: return "ACC_NOT_CONVERTED";
      F_IDX_RANGE: return "IDX_RANGE"; F_IDX_SHORT: return "IDX_SHORT"; F_PAGE_FAULT: return "PAGE_FAULT"; F_BAD_EDGE: return "BAD_EDGE";
      F_BAD_QUALIFIER: return "BAD_QUALIFIER"; F_BAD_SHAPE: return "BAD_SHAPE"; F_BAD_OPCODE: return "BAD_OPCODE"; default: return "?";
    endcase
  endfunction

  // ---------------------------------------------------------------- sequencing
  int ftr, fst, fdr, fsp, fpg;
  logic dump_now = 1'b0, ended = 1'b0;
  logic [31:0] pages [4096]; int n_pages;
  initial begin
    if (!$value$plusargs("VEC=%s", vec_dir)) vec_dir = "vectors/gemm";
    if (!$value$plusargs("OUT=%s", out_dir)) out_dir = vec_dir;
    if (!$value$plusargs("MEMLAT=%d", mem_lat)) mem_lat = 24;
    if (!$value$plusargs("MAXCYC=%d", max_cyc)) max_cyc = 20_000_000;
    if (!$value$plusargs("TDEADLOCK=%d", t_deadlock_i)) t_deadlock_i = 1_000_000;
    if (!$value$plusargs("NDESC=%d", n_desc)) n_desc = 0;
    if (!$value$plusargs("NWORDS=%d", n_words_i)) n_words_i = 0;
    if (mem_lat < 1) mem_lat = 1; if (mem_lat > MAX_LAT) mem_lat = MAX_LAT;
    t_deadlock = 32'(t_deadlock_i);
    $readmemh({vec_dir, "/dram.hex"}, dram);
    if (n_words_i > 0) $readmemh({vec_dir, "/stream.hex"}, stream);
    if (n_desc > 0) $readmemh({vec_dir, "/desc.hex"}, descs);
    n_pages = 0;
    fpg = $fopen({vec_dir, "/pages.txt"}, "r");
    if (fpg != 0) begin
      while (!$feof(fpg) && n_pages < 4096) begin
        logic [31:0] a;
        if ($fscanf(fpg, "%h\n", a) == 1) begin pages[n_pages] = a; n_pages++; end
      end
      $fclose(fpg);
    end
    ftr = $fopen({out_dir, "/rtl_trace.txt"}, "w");
  end

  always_ff @(posedge clk) begin
    tick <= tick + 1;
    desc_we <= 0; start <= 0;
    if (tick < 4) rst_n <= 0;
    else if (tick < 4 + n_desc) begin rst_n <= 1; desc_we <= 1; desc_widx <= 8'(tick - 4); desc_wdata <= descs[tick - 4]; end
    else if (tick == 4 + n_desc) begin rst_n <= 1; start <= 1; end
    else rst_n <= 1;
  end

  // trace
  always_ff @(posedge clk) begin
    if (running) begin
      for (int q = 0; q < 5; q++) begin
        if (tr_sync[q])  $fwrite(ftr, "%0d %s SYNC %s %0d\n", cycle, qname(q), opname(tr_sync_op[q]), tr_sync_pc[q]);
        if (tr_issue[q]) $fwrite(ftr, "%0d %s ISSUE %s %0d\n", cycle, qname(q), opname(tr_issue_op[q]), tr_issue_pc[q]);
        if (tr_done[q])  $fwrite(ftr, "%0d %s DONE %s %0d\n", cycle, qname(q), opname(tr_done_op[q]), tr_done_pc[q]);
      end
      // front-end dispatch into a queue (J5: dispatch rate, loop overhead, VCOUNT -> LOOP r); compare.py ignores it
      if (u_core.fe_dq_valid && !u_core.iq_full[u_core.fe_dq_q])
        $fwrite(ftr, "%0d FE DISP %s %0d\n", cycle, opname(u_core.fe_dq_ins.op), u_core.fe_dq_ins.pc);
    end
  end

  // end of run: status, DRAM pages, scratchpads
  always_ff @(posedge clk) begin
    if (!ended && tick > 8 && (finished || fault || cycle > max_cyc)) begin
      ended <= 1; dump_now <= 1;
      fst = $fopen({out_dir, "/status.txt"}, "w");
      if (finished) $fwrite(fst, "ok %0d\n", cycle);
      else if (fault) $fwrite(fst, "fault %s %s %0d %0d\n", fname(fault_code), qname(fault_q), fault_pc, cycle);
      else $fwrite(fst, "timeout %0d\n", cycle);
      $fclose(fst);
      fdr = $fopen({out_dir, "/dram_out.hex"}, "w");
      for (int p = 0; p < n_pages; p++) begin
        $fwrite(fdr, "@%0h\n", pages[p]);
        for (int k = 0; k < 4096; k++) $fwrite(fdr, "%02x\n", dram[pages[p] + k]);
      end
      $fclose(fdr);
      fsp = $fopen({out_dir, "/sp_dump.txt"}, "w");
    end else if (dump_now) begin
      dump_now <= 0;
      $fclose(fsp);
      $fclose(ftr);
      $finish;
    end
  end
  // scratchpad dump (hierarchical, one block per bank)
  for (genvar s = 0; s < 5; s++) begin : g_dump_sp
    for (genvar b = 0; b < BANKS; b++) begin : g_dump_b
      always_ff @(posedge clk) if (dump_now) begin
        for (int w = 0; w < u_core.g_sp[s].WORDS; w++)
          $fwrite(fsp, "%0d %0d %0d %0h\n", s, b, w, u_core.g_sp[s].u_sp.g_bank[b].mem[w]);
      end
    end
  end
endmodule
