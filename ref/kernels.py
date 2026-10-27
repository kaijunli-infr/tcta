"""Reference generators for the five kernel families.

Each `gen_*` returns a Fixture with inputs, DRAM images, expected outputs and
meta. All randomness is seeded; call `gen_all()` to write every fixture.

Conventions
- Activations and weights are BF16-rounded before the reference computes,
  so the fixture's float32 arrays are exactly what the device sees.
- Softmax uses exp2 with the log2(e)/sqrt(d) scale folded in, matching
  `VEXP2` (ISA §5.3) so the device need not implement exp.
"""
from __future__ import annotations

import numpy as np

from common import Fixture, bf16, bf16_bytes, f4_b32_bytes, f4_r16_rows, f8_t_bytes, rowmax_err

LOG2E = np.float32(np.log2(np.e))


# --------------------------------------------------------------------------
# 1. GEMM with partial tiles (also the "identity move" used by other tests)
# --------------------------------------------------------------------------
def gen_gemm(root, seed=0, M=64, K=128, N=128, M_valid=50, K_valid=100, N_valid=96):
    rng = np.random.default_rng(seed)
    A = bf16(rng.standard_normal((M, K)))
    B = bf16(rng.standard_normal((K, N)))
    C = (A[:M_valid, :K_valid].astype(np.float64) @ B[:K_valid, :N_valid].astype(np.float64)).astype(np.float32)
    fx = Fixture("gemm_partial", root)
    fx.add(A=A, B=B, A_bytes=bf16_bytes(A), B_bytes=bf16_bytes(B), C=C)
    fx.meta.update(kind="gemm", M=M, K=K, N=N, valid=[M_valid, K_valid, N_valid],
                   layout={"A": "bf16 row-major [M,K]", "B": "bf16 row-major [K,N]"})
    return fx


# --------------------------------------------------------------------------
# 2. FlashAttention-2 prefill with GQA and causal mask
# --------------------------------------------------------------------------
def fa_reference(Qh, Kh, Vh, scale, causal, q_start=0):
    """Qh [Lq,d], Kh/Vh [Lk,d] (BF16 values). Returns O [Lq,d] float32, and per-row m, l."""
    S = (Qh.astype(np.float64) @ Kh.astype(np.float64).T) * float(scale)   # log2 domain
    if causal:
        qi = np.arange(S.shape[0])[:, None] + q_start
        kj = np.arange(S.shape[1])[None, :]
        S = np.where(kj > qi, -np.inf, S)
    m = S.max(axis=1, keepdims=True)
    P = np.exp2(S - m)
    l = P.sum(axis=1, keepdims=True)
    Pb = bf16(P.astype(np.float32)).astype(np.float64)          # device feeds BF16 P to the array
    O = (Pb @ Vh.astype(np.float64)) / l
    return O.astype(np.float32), m[:, 0].astype(np.float32), l[:, 0].astype(np.float32)


def gen_flash_attention(root, seed=1, Lq=256, Lk=256, d=64, n_q=8, G=4, causal=False):
    """GQA: n_q query heads, n_q/G KV heads. Llama-3.2-1B-class: d=64, G=4."""
    rng = np.random.default_rng(seed)
    n_kv = n_q // G
    Q = bf16(rng.standard_normal((n_q, Lq, d)))
    K = bf16(rng.standard_normal((n_kv, Lk, d)))
    V = bf16(rng.standard_normal((n_kv, Lk, d)))
    scale = np.float32(LOG2E / np.sqrt(d))
    O = np.empty((n_q, Lq, d), np.float32); m = np.empty((n_q, Lq), np.float32); l = np.empty((n_q, Lq), np.float32)
    for h in range(n_q):
        O[h], m[h], l[h] = fa_reference(Q[h], K[h // G], V[h // G], scale, causal)
    fx = Fixture("flash_attn_gqa", root)
    fx.add(Q=Q, K=K, V=V, O=O, m=m, l=l,
           Q_bytes=bf16_bytes(Q), K_bytes=bf16_bytes(K), V_bytes=bf16_bytes(V))
    fx.meta.update(kind="flash_attention", Lq=Lq, Lk=Lk, d=d, n_q=n_q, n_kv=n_kv, G=G, causal=causal,
                   scale_log2=float(scale), tile={"Bq": 64, "Bk": 64},
                   notes="O = softmax(QK^T * scale) V with exp2; m,l are final per-row stats for debugging the online pass. "
                         "GQA schedule stacks G query heads into one [G*Bq, d] INP tile against one K/V tile.")
    return fx


# --------------------------------------------------------------------------
# 3. MLA decode, weight-absorbed (DeepSeek-V2-Lite class heads)
# --------------------------------------------------------------------------
def gen_mla_decode(root, seed=2, n_h=16, Bq=4, L=1024, d_latent=512, d_rope=64):
    """Absorbed form: per head q_abs = [q_nope W_UK^T | q_rope] (576), K = [c_kv | k_rope] (576), V = c_kv (512).

    All heads share one latent cache (num_key_value_heads = 1). Bq = 4 decode rows per head
    (one token x 4 MTP/draft slots) -> stacked across heads to fill the 64-row tile.
    """
    rng = np.random.default_rng(seed)
    dk = d_latent + d_rope
    q_abs = bf16(rng.standard_normal((n_h, Bq, dk)) / np.sqrt(dk))
    c_kv = bf16(rng.standard_normal((L, d_latent)))
    k_rope = bf16(rng.standard_normal((L, d_rope)))
    K = np.concatenate([c_kv, k_rope], axis=1)                    # [L, 576]
    scale = np.float32(LOG2E / np.sqrt(dk))
    O = np.empty((n_h, Bq, d_latent), np.float32)
    for h in range(n_h):
        O[h], _, _ = fa_reference(q_abs[h], K, c_kv, scale, causal=False)
    fx = Fixture("mla_decode_absorbed", root)
    fx.add(q_abs=q_abs, K=K, V=c_kv, O=O, K_bytes=bf16_bytes(K), q_bytes=bf16_bytes(q_abs))
    fx.meta.update(kind="mla_decode", n_h=n_h, Bq=Bq, L=L, d_k=dk, d_v=d_latent, scale_log2=float(scale),
                   tile={"Bq_stacked": n_h * Bq, "Bk": 64, "k_tiles_per_entry": dk // 64, "acc_wide": [64, 512]},
                   notes="K reduction is 9 x 64-deep tiles into a [64,64] S tile; O accumulates into the [64,512] wide ACC.")
    return fx


# --------------------------------------------------------------------------
# 4. KDA (Kimi Delta Attention) -- recurrent reference, chunk size 64
# --------------------------------------------------------------------------
def kda_recurrent(q, k, v, alpha, beta, S0=None):
    """Per-token delta rule with per-channel decay (frozen reference form):
        S_t = diag(a_t) S_{t-1} - b_t k_t (k_t^T diag(a_t) S_{t-1}) + b_t k_t v_t^T
        o_t = S_t^T q_t
    q,k,v: [T, d]; alpha: [T, d] in (0,1]; beta: [T] in (0,1]. S: [d_k, d_v].
    """
    T, dk = q.shape; dv = v.shape[1]
    S = np.zeros((dk, dv), np.float64) if S0 is None else S0.astype(np.float64).copy()
    o = np.empty((T, dv), np.float64)
    for t in range(T):
        Sd = alpha[t][:, None] * S
        kt = k[t].astype(np.float64)
        S = Sd - beta[t] * np.outer(kt, kt @ Sd) + beta[t] * np.outer(kt, v[t])
        o[t] = S.T @ q[t]
    return o.astype(np.float32), S.astype(np.float32)


def gen_kda(root, seed=3, T=256, d=128, C=64):
    rng = np.random.default_rng(seed)
    q = bf16(rng.standard_normal((T, d)) / np.sqrt(d))
    k = bf16(rng.standard_normal((T, d)) / np.sqrt(d))
    k = bf16(k / np.linalg.norm(k, axis=1, keepdims=True))        # L2-normalized keys (standard for delta rule)
    v = bf16(rng.standard_normal((T, d)))
    alpha = bf16(np.exp(-rng.uniform(0.0, 0.1, (T, d))))          # per-channel decay in (0.9, 1]
    beta = bf16(rng.uniform(0.3, 1.0, T))
    o, S_final = kda_recurrent(q, k, v, alpha, beta)
    # per-chunk state checkpoints so the chunkwise kernel can be checked chunk by chunk
    S_chunks = np.empty((T // C, d, d), np.float32)
    S = None
    for c in range(T // C):
        sl = slice(c * C, (c + 1) * C)
        _, S = kda_recurrent(q[sl], k[sl], v[sl], alpha[sl], beta[sl], S)
        S_chunks[c] = S
    fx = Fixture("kda_chunk64", root)
    fx.add(q=q, k=k, v=v, alpha=alpha, beta=beta, o=o, S_final=S_final, S_chunks=S_chunks)
    fx.meta.update(kind="kda", T=T, d_k=d, d_v=d, C=C,
                   notes="Recurrent form is normative. S_chunks[c] is the state after chunk c; the chunkwise kernel "
                         "(3 intra-chunk MMAs + masked MMA + Neumann triangular solve, Appendix B) must match each checkpoint. "
                         "Gate placement follows the pinned Kimi Linear reference commit; change kda_recurrent if the pin differs.")
    return fx


# --------------------------------------------------------------------------
# 5. MoE layer: sigmoid/sqrt-softplus router, top-k, grouped GEMM, SwiGLU, shared expert
# --------------------------------------------------------------------------
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def router(logits, k, score_func="sqrtsoftplus", norm_topk=True, routed_scaling=1.5):
    if score_func == "sqrtsoftplus":
        s = np.sqrt(softplus(logits.astype(np.float64)))
    elif score_func == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    else:
        raise ValueError(score_func)
    idx = np.argsort(-s, axis=1, kind="stable")[:, :k]
    w = np.take_along_axis(s, idx, axis=1)
    if norm_topk:
        w = w / w.sum(axis=1, keepdims=True)
    return idx.astype(np.int32), (w * routed_scaling).astype(np.float32)


def swiglu(x, W1, W3, W2):
    h = (x @ W1) * (1.0 / (1.0 + np.exp(-(x @ W1)))) * (x @ W3)   # silu(xW1) * xW3
    return h @ W2


def gen_moe(root, seed=4, B=256, d=512, d_ff=256, E=32, k=6, score_func="sqrtsoftplus"):
    """MoE layer: router + grouped SwiGLU experts in MXFP4 (f4.b32) + BF16 shared expert.

    Expert weights are stored as f4.b32 payload + UE8M0 scales (Appendix A) and ALSO as the
    dequantized BF16 values the reference consumed (W*_deq), so both the DMA path and the
    numerics can be checked. Combine layout: `y_parts[token, slot, :]` is each expert's weighted
    contribution at the token's routing slot (what STORE.IDX writes at row token*k + slot);
    `y = shared + sum_slots y_parts`.
    """
    rng = np.random.default_rng(seed)
    x = bf16(rng.standard_normal((B, d)))
    Wg = bf16(rng.standard_normal((d, E)) / np.sqrt(d))
    Ws1 = bf16(rng.standard_normal((d, d_ff)) / np.sqrt(d)); Ws3 = bf16(rng.standard_normal((d, d_ff)) / np.sqrt(d))
    Ws2 = bf16(rng.standard_normal((d_ff, d)) / np.sqrt(d_ff))
    logits = x.astype(np.float64) @ Wg.astype(np.float64)
    idx, w = router(logits, k, score_func)
    y_shared = swiglu(x.astype(np.float64), Ws1, Ws3, Ws2)
    y_parts = np.zeros((B, k, d), np.float64)
    pay, sc, deq = {n: [] for n in "132"}, {n: [] for n in "132"}, {n: [] for n in "132"}
    tokens_per_expert = []
    for e in range(E):
        p1, s1, d1 = f4_b32_bytes(rng.standard_normal((d, d_ff)).astype(np.float32) / np.sqrt(d))
        p3, s3, d3 = f4_b32_bytes(rng.standard_normal((d, d_ff)).astype(np.float32) / np.sqrt(d))
        p2, s2, d2 = f4_b32_bytes(rng.standard_normal((d_ff, d)).astype(np.float32) / np.sqrt(d_ff))
        for n, (pp, ss, dd) in zip("132", ((p1, s1, d1), (p3, s3, d3), (p2, s2, d2))):
            pay[n].append(pp); sc[n].append(ss); deq[n].append(dd)
        rows, slots = np.nonzero(idx == e)
        tokens_per_expert.append(rows.astype(np.int32))
        if rows.size:
            ye = swiglu(x[rows].astype(np.float64), d1, d3, d2)
            y_parts[rows, slots] = w[rows, slots][:, None] * ye
    y = y_shared + y_parts.sum(axis=1)
    counts = np.array([t.size for t in tokens_per_expert], np.int32)
    maxc = int(counts.max())
    idx_lists = -np.ones((E, maxc), np.int32)                         # -1 = skip (ISA §6.1)
    for e, t in enumerate(tokens_per_expert):
        idx_lists[e, :t.size] = t
    # scatter destination rows for STORE.IDX: token*k + slot, aligned with idx_lists
    slot_of = -np.ones((E, maxc), np.int32)
    for e, t in enumerate(tokens_per_expert):
        slot_of[e, :t.size] = [int(np.nonzero(idx[tok] == e)[0][0]) for tok in t]
    dst_rows = np.where(idx_lists >= 0, idx_lists * k + slot_of, -1).astype(np.int32)
    fx = Fixture("moe_grouped", root)
    fx.add(x=x, Wg=Wg, Ws1=Ws1, Ws3=Ws3, Ws2=Ws2,
           W1_payload=np.stack(pay["1"]), W1_scales=np.stack(sc["1"]), W1_deq=np.stack(deq["1"]),
           W3_payload=np.stack(pay["3"]), W3_scales=np.stack(sc["3"]), W3_deq=np.stack(deq["3"]),
           W2_payload=np.stack(pay["2"]), W2_scales=np.stack(sc["2"]), W2_deq=np.stack(deq["2"]),
           logits=logits.astype(np.float32), topk_idx=idx, topk_w=w, counts=counts, idx_lists=idx_lists,
           dst_rows=dst_rows, y_shared=y_shared.astype(np.float32), y_parts=y_parts.astype(np.float32),
           y=y.astype(np.float32), x_bytes=bf16_bytes(x))
    fx.meta.update(kind="moe", B=B, d=d, d_ff=d_ff, E=E, k=k, score_func=score_func, norm_topk=True,
                   routed_scaling=1.5, max_tokens_per_expert=maxc,
                   formats={"experts": "f4.b32 (payload [E,K,N/2] + scales [E,K/32,N/32]); W*_deq is the dequantized value"},
                   notes="Schedule: VTOPK on logits -> idx_lists; per expert LOOP r=count,step=64 over token tiles via "
                         "LOAD.IDX (-1 rows zero-filled); SwiGLU in VEC/SFU; weighted outputs scattered with STORE.IDX "
                         "to rows dst_rows (= token*k + slot) of a [B*k, d] buffer; a reduce kernel sums the k rows per token.")
    return fx


# --------------------------------------------------------------------------
# 6. DeepSeek-V4.1-Flash decoder-layer slice at real shapes, reduced expert count
# --------------------------------------------------------------------------
def gen_v41_flash_slice(root, seed=5, B=64, L=4096, E=32, k=6, topk=512, d=5120, n_h=64,
                        hd=512, rope=64, q_lora=1280, o_lora=1024, o_groups=8, d_ff=512):
    """One decoder layer of V4.1-Flash shape with synthetic weights.

    Real: d=5120, 64 heads, head_dim 512 (+64 rope), 1 KV head, q_lora 1280, o_lora 1024 in 8 groups,
    384 experts x 2304, top-6, sparse attention over top-512 of the shared FP4 cache.
    Reduced: E=32 experts (routing shape preserved), d_ff=512 (tile shapes unchanged, fewer N tiles),
    L=4096 context, B=64 decode tokens.
    Formats: experts f4.b32 (MXFP4, 32x32 UE8M0 blocks), attention weights f8.t, KV f4.r16.
    What this validates: every kernel at the real tile shapes and the real DRAM formats. Not the
    real model's numerics -- that is level 3 with real weights.

    Expert weights are stored ONLY as f4.b32 payload + scales; the normative values are their
    dequantization (dtypes.dequant_row), which is what the reference below used.
    """
    rng = np.random.default_rng(seed)
    dk = hd + rope
    f32 = lambda a: np.asarray(a, np.float32)
    # ---- inputs ----
    h_in = bf16(rng.standard_normal((B, d)))
    # ---- attention weights, absorbed MLA form ----
    W_qa = bf16(rng.standard_normal((d, q_lora)) / np.sqrt(d))
    W_qb = bf16(rng.standard_normal((q_lora, n_h * dk)) / np.sqrt(q_lora))
    W_kv = bf16(rng.standard_normal((d, dk)) / np.sqrt(d))
    W_oa = bf16(rng.standard_normal((o_groups, (n_h // o_groups) * hd, o_lora)) / np.sqrt(hd))
    W_ob = bf16(rng.standard_normal((o_groups, o_lora, d)) / np.sqrt(o_lora))
    # ---- KV cache: L entries of 576, FP4 r16 ----
    kv_rows, kv_deq = f4_r16_rows(rng.standard_normal((L, dk)).astype(np.float32))
    # ---- sparse indices: top-512 per token (synthetic indexer: random distinct, sorted) ----
    sel = np.stack([np.sort(rng.choice(L, topk, replace=False)) for _ in range(B)]).astype(np.int32)
    # ---- shared expert + router ----
    Wg = bf16(rng.standard_normal((d, E)) / np.sqrt(d))
    Ws1 = bf16(rng.standard_normal((d, d_ff)) / np.sqrt(d)); Ws3 = bf16(rng.standard_normal((d, d_ff)) / np.sqrt(d))
    Ws2 = bf16(rng.standard_normal((d_ff, d)) / np.sqrt(d_ff))

    # ================= reference forward (float32; inputs are BF16-exact) =================
    x = f32(h_in)
    q = bf16(bf16(x @ W_qa) @ W_qb).reshape(B, n_h, dk)
    kv_new = bf16(x @ W_kv)
    scale = np.float32(LOG2E / np.sqrt(dk))
    attn = np.empty((B, n_h, hd), np.float32)
    for b in range(B):
        Kb = np.concatenate([kv_deq[sel[b]], kv_new[b:b + 1]], axis=0)         # [513, 576]
        Vb = Kb[:, :hd]
        o, _, _ = fa_reference(q[b], Kb, Vb, scale, causal=False)              # all heads at once: [n_h, hd]
        attn[b] = o
    attn = bf16(attn.reshape(B, n_h * hd))
    o_proj = np.zeros((B, d), np.float32)
    gsz = (n_h // o_groups) * hd
    for g in range(o_groups):
        o_proj += bf16(attn[:, g * gsz:(g + 1) * gsz] @ W_oa[g]) @ W_ob[g]
    h_mid = bf16(x + o_proj)
    # MoE: experts generated, quantized and consumed one at a time to bound memory
    logits = h_mid @ Wg
    idx, w = router(logits, k, "sqrtsoftplus")
    y = swiglu(h_mid, Ws1, Ws3, Ws2).astype(np.float32)
    counts = np.zeros(E, np.int32)
    W1_pay, W1_sc, W3_pay, W3_sc, W2_pay, W2_sc = [], [], [], [], [], []
    for e in range(E):
        p1, s1, d1 = f4_b32_bytes(rng.standard_normal((d, d_ff)).astype(np.float32) / np.sqrt(d))
        p3, s3, d3 = f4_b32_bytes(rng.standard_normal((d, d_ff)).astype(np.float32) / np.sqrt(d))
        p2, s2, d2 = f4_b32_bytes(rng.standard_normal((d_ff, d)).astype(np.float32) / np.sqrt(d_ff))
        W1_pay.append(p1); W1_sc.append(s1); W3_pay.append(p3); W3_sc.append(s3); W2_pay.append(p2); W2_sc.append(s2)
        rows, slots = np.nonzero(idx == e)
        counts[e] = rows.size
        if rows.size:
            y[rows] += w[rows, slots][:, None] * swiglu(h_mid[rows], d1, d3, d2)
    h_out = bf16(h_mid + y)

    fx = Fixture("v41_flash_decoder_slice", root)
    fx.add(h_in=h_in, h_mid=h_mid, h_out=h_out, attn=attn,
           W_qa=W_qa, W_qb=W_qb, W_kv=W_kv, W_oa=W_oa, W_ob=W_ob, Wg=Wg, Ws1=Ws1, Ws3=Ws3, Ws2=Ws2,
           kv_rows=kv_rows, kv_new=kv_new, sel=sel,
           W1_payload=np.stack(W1_pay), W1_scales=np.stack(W1_sc),
           W3_payload=np.stack(W3_pay), W3_scales=np.stack(W3_sc),
           W2_payload=np.stack(W2_pay), W2_scales=np.stack(W2_sc),
           topk_idx=idx, topk_w=w, counts=counts)
    for name, W in (("W_qa", W_qa), ("W_qb", W_qb), ("W_kv", W_kv)):
        codes, s, _ = f8_t_bytes(W)
        fx.add(**{name + "_f8": codes}); fx.meta[name + "_scale"] = float(s)
    fx.meta.update(kind="v41_flash_slice", B=B, L=L, E=E, k=k, topk=topk, d=d, n_h=n_h, head_dim=hd, rope=rope,
                   q_lora=q_lora, o_lora=o_lora, o_groups=o_groups, d_ff=d_ff, scale_log2=float(scale),
                   formats={"experts": "f4.b32 (payload [E,K,N/2] uint8 + scales [E,K/32,N/32] UE8M0)",
                            "attention_weights": "bf16 arrays, plus f8.t codes with per-tensor scale in meta",
                            "kv_cache": "f4.r16, 384 B per entry, stride 384; dequant with dtypes.dequant_row"},
                   notes="Sparse attention over sel[b] (512 entries via LOAD.IDX row_bytes=384) plus the token's own "
                         "kv_new entry. Pre-norms omitted. Expert values are the dequantized payload; the reference used "
                         "exactly those. d_ff reduced 2304->512 to bound fixture size; tile shapes are unchanged.")
    return fx


# --------------------------------------------------------------------------
def gen_all(root="fixtures"):
    out = []
    for g in (gen_gemm, gen_flash_attention, gen_mla_decode, gen_kda, gen_moe, gen_v41_flash_slice):
        fx = g(root)
        out.append(fx.save())
        print("wrote", out[-1])
    return out


if __name__ == "__main__":
    import sys
    gen_all(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
