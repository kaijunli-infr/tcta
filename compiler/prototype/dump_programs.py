"""Regenerate the program listings in compiler/prototype/dumps/ (E4, F1, F2, G1, G2, GEMM).

    python compiler/prototype/dump_programs.py
"""
import os
from dataclasses import replace
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim"))

import gemm_kernel  # noqa: E402
import kda_kernel  # noqa: E402
import kda_kernel_roundtrip  # noqa: E402
import mla_kernel  # noqa: E402
import moe_data  # noqa: E402
import moe_kernel  # noqa: E402
from cut import lower_to_tile_tir  # noqa: E402
from tcta import DeviceConfig  # noqa: E402
from tcta.isa import DEVICE_CONFIG_C, SP  # noqa: E402
from tile_tir_to_program import listing, tile_tir_to_program  # noqa: E402

OUT = os.path.join(HERE, "dumps")
JOBS = [
    # (file, kernel builder result, scope map, bx, dram layout, device config, title)
    ("program_moe_e0.txt", moe_kernel.build_moe_expert_kernel(), moe_kernel.SCOPE_MAP, 0, moe_data.DRAM, None,
     "E4/F2: expert 0, f4.b32 weights, LOOP r step 64, STORE.IDX"),
    ("program_moe_bf16_e0.txt", moe_kernel.build_moe_expert_kernel(weight_format="bf16"), moe_kernel.SCOPE_MAP, 0, moe_data.DRAM, None,
     "E4: expert 0, bf16 weights"),
    ("program_moe_persistent.txt", moe_kernel.build_moe_persistent_kernel(), moe_kernel.SCOPE_MAP, 0, moe_data.DRAM, None,
     "F2: all 32 experts in one program"),
    ("program_moe_persistent_noprefetch.txt", moe_kernel.build_moe_persistent_kernel(prefetch=False), moe_kernel.SCOPE_MAP, 0,
     moe_data.DRAM, None, "F2 without the count prefetch: LOOP n=E { count; LOOP r }"),
    ("program_moe_combine_t0.txt", moe_kernel.build_moe_combine_kernel(), moe_kernel.COMBINE_SCOPE_MAP, 0, moe_data.DRAM, None,
     "F1: combine, token tile 0"),
    ("program_router_t0.txt", moe_kernel.build_router_kernel(), moe_kernel.ROUTER_SCOPE_MAP, 0, moe_data.DRAM, None,
     "E4: router, token 0"),
    ("program_gemm.txt", gemm_kernel.build_gemm_kernel(), gemm_kernel.SCOPE_MAP, 0, gemm_kernel.DRAM, None,
     "Gate 1 GEMM family: C[50, 96] = A[50, 100] B[100, 96], partial tiles"),
    ("program_mla.txt", mla_kernel.build_mla_kernel(), mla_kernel.SCOPE_MAP, 0, mla_kernel.DRAM, None,
     "G1: MLA decode absorbed, 64 stacked rows, LOOP n=16 over KV tiles, wide ACC O[64, 512]"),
    ("program_kda.txt", kda_kernel.build_kda_kernel(), kda_kernel.SCOPE_MAP, 0, kda_kernel.DRAM, kda_kernel.CONFIG_C,
     "G2: KDA chunkwise on v0.1.3, LOOP n=4 over chunks, computed B operands via VCVT dst=WGT, state in STATE via VCVT row0, config C"),
    ("program_kda_roundtrip.txt", kda_kernel_roundtrip.build_kda_kernel(), kda_kernel_roundtrip.SCOPE_MAP, 0, kda_kernel_roundtrip.DRAM,
     kda_kernel.CONFIG_C, "G2 on v0.1.2 (kept for the phase-5 latency sweep): 13 DRAM round trips + state mirror per chunk, config C"),
    ("program_mla_inp6.txt", mla_kernel.build_mla_kernel(inp_slots=6), mla_kernel.SCOPE_MAP, 0, mla_kernel.DRAM,
     DeviceConfig(slots={SP.INP: 6, SP.WGT: 4, SP.ACC: 6, SP.VEC: 16, SP.STATE: 1}),
     "G1 with Q resident in 5 INP slots (phase-5 INP sweep): the Q loads leave the KV loop"),
    ("program_moe_flat_ring12.txt", moe_kernel.build_moe_persistent_kernel(num_stages=12, flat_w13=True), moe_kernel.SCOPE_MAP, 0,
     moe_data.DRAM, replace(DEVICE_CONFIG_C, fifo_depth=32),
     "F2 at the phase-6 ring knee: [W1|W3] streamed as one flat 16-tile ring of depth 12 per token tile, x resident in 2 INP slots, config C (FIFO 32)"),
    ("program_mla_kv2.txt", mla_kernel.build_mla_kernel(inp_slots=5, kv_unroll=2), mla_kernel.SCOPE_MAP, 0, mla_kernel.DRAM,
     replace(DEVICE_CONFIG_C, fifo_depth=32),
     "G1 with Q resident (INP 5) and the K ring over two KV tiles (10 WGT slots, phase-6 ring sweep), config C (FIFO 32)"),
]

if __name__ == "__main__":
    for name, func, smap, bx, dram, cfg, title in JOBS:
        f = lower_to_tile_tir(func, scope_map=smap)
        prog, rep = tile_tir_to_program(f, bx=bx, dram=dram, cfg=cfg)
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(f"# {title}\n# n_instrs={rep.n_instrs} loops={rep.loops if len(rep.loops) < 4 else str(len(rep.loops)) + ' loops'} "
                     f"edge_counts={rep.edge_counts}\n# primed={rep.primed_pushes} hoisted={rep.hoisted_pops} body_end={rep.body_end_pops} "
                     f"drained={rep.drained} max_fifo_level={rep.max_fifo_level}\n# collapsed={rep.collapsed} shared_slots={rep.shared_slots}\n"
                     f"# slots={rep.slots}\n")
            fh.write(listing(prog) + "\n")
        print(name, rep.n_instrs)
