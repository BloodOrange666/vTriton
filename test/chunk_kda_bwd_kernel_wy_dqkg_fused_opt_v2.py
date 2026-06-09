import torch
import torch.nn as nn
import triton
import triton.language as tl
import inspect

FLA_CACHE_RESULTS = True
SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)
autotune_cache_kwargs = (
    {"cache_results": FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}
)

@triton.autotune(
    configs=[triton.Config({'BK': 32, 'BV': 32}, num_warps=2, num_stages=3)],
    key=['BT', 'TRANSPOSE_STATE'],
    **autotune_cache_kwargs,
)
@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T', 'stride_hz'])
def chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2(
    q,              # [B, T, H, K]
    k,              # [B, T, H, K]
    v,              # [B, T, H, V]
    v_new,          # [B, T, H, V]
    g,              # [B, T, H, K]
    beta,           # [B, T, H]
    A,              # [B, T, H, BT]
    h,              # [1, 128, H, K, V]
    do,             # [B, T, H, V]
    dh,             # [B, 128, H, K, V]
    dq,             # [B, T, H, K]
    dk,             # [B, T, H, K]
    dv,             # [B, T, H, V], input
    dv2,            # [B, T, H, V], output
    dg,             # [B, T, H, K]
    db,             # [B, T, H]
    dA,             # [B, T, H, BT]
    cu_seqlens,     # [4]
    chunk_indices,  # [128, 2]
    scale,          # float
    T,              # int
    scalar,         # float
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    stride_hz,      # int, = B*T (non-varlen) or total_len (varlen)
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (i_h * stride_hz + bos) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (i_h * stride_hz + bos) * V
    g += (bos * H + i_h) * K
    beta += i_h * stride_hz + bos
    A += (i_h * stride_hz + bos) * BT
    h += (i_tg * H + i_h) * K*V
    do += (i_h * stride_hz + bos) * V
    dh += (i_tg * H + i_h) * K*V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    db += i_h * stride_hz + bos
    dA += (i_h * stride_hz + bos) * BT

    BT_arange = i_t * BT + tl.arange(0, BT)
    BT_arange_zero_offset = tl.arange(0, BT)
    BT_mask = (BT_arange < T) & (BT_arange >= 0)
    BT_mask_zero_offset = (BT_arange_zero_offset < BT) & (BT_arange_zero_offset >= 0)
    p_beta = beta + BT_arange
    b_beta = tl.load(p_beta, mask = BT_mask, other = 0.)

    p_A = A + BT_arange_zero_offset[:, None] + BT * BT_arange
    b_A = tl.load(p_A, mask = BT_mask_zero_offset[:, None] & BT_mask[None, :], other = 0.)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        BK_arange = i_k * BK + tl.arange(0, BK)
        BK_mask = (BK_arange < K) & (BK_arange >= 0)
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * H*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)
        b_gn = tl.where(m_k, b_gn, 0)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            BV_arange = i_v * BV + tl.arange(0, BV)
            BV_mask = (BV_arange < V) & (BV_arange >= 0)
            p_v_new = v_new + BT_arange[:, None] * V + BV_arange
            p_do = do + BT_arange[:, None] * V + BV_arange
            if TRANSPOSE_STATE:
                p_h = tl.make_block_ptr(h, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
                p_dh = tl.make_block_ptr(dh, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
                p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dv = dv + BT_arange[:, None] * H*V + BV_arange
            # [BT, BV]
            b_v_new = tl.load(p_v_new, mask= BT_mask[:,None] & BV_mask[None, :])
            b_v_new = tl.where(BT_mask[:,None] & BV_mask[None, :], b_v_new, 0)
            b_do = tl.load(p_do, mask= BT_mask[:,None] & BV_mask[None, :])
            b_do = tl.where(BT_mask[:,None] & BV_mask[None, :], b_do, 0)
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            tl.extra.cann.extension.compile_hint(b_h, "mayDiscretememaccess")
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            tl.extra.cann.extension.compile_hint(b_dh, "mayDiscretememaccess")
            # [BT, BV]
            b_dv = tl.load(p_dv, mask= BT_mask[:,None] & BV_mask[None, :])
            tl.extra.cann.extension.compile_hint(b_dv, "mayDiscretememaccess")

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype)) * scalar
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype)) * scalar
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype)) * scalar
            tl.debug_barrier()  # DO NOT REMOVE THIS LINE!
            if i_k == 0:
                p_v = v + BT_arange[:, None] * H*V + BV_arange
                p_dv2_block = tl.make_block_ptr(dv2, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

                b_v = tl.load(p_v, mask = BT_mask[:, None] & BV_mask[None, :], other = 0.0)

                b_dA += tl.dot(b_dv, tl.trans(b_v))  * scalar

                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)
                casted_b_dv2 = b_dv2.to(p_dv2_block.dtype.element_ty)
                tl.store(p_dv2_block, casted_b_dv2, boundary_check=(0, 1))

        b_gk_exp = tl.math.exp2(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= tl.math.exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], tl.math.exp2(b_gn[None, :] - b_g), 0)

        b_kg = b_k * b_gk_exp

        b_dw = -b_dw.to(b_A.dtype)
        b_dA += tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype))) * scalar

        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg, 1)

        p_q = tl.make_block_ptr(q, (T, K), (K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_beta[:, None]
        b_dk = b_dk + b_dkgb * b_gb

        p_dq_block = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk_block = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg_block = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        casted_b_dq = b_dq.to(p_dq_block.dtype.element_ty)
        tl.store(p_dq_block, casted_b_dq, boundary_check=(0, 1))
        casted_b_dk = b_dk.to(p_dk_block.dtype.element_ty)
        tl.store(p_dk_block, casted_b_dk, boundary_check=(0, 1))
        casted_b_dg = b_dg.to(p_dg_block.dtype.element_ty)
        tl.store(p_dg_block, casted_b_dg, boundary_check=(0, 1))

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    p_dA = tl.make_block_ptr(dA, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (1,), (i_t * BT,), (BT,), (0,))
    casted_b_dA = b_dA.to(p_dA.dtype.element_ty)
    tl.store(p_dA, casted_b_dA, boundary_check=(0, 1))
    casted_b_db = b_db.to(p_db.dtype.element_ty)
    tl.store(p_db, casted_b_db, boundary_check=(0,))


def build_inputs():
    device = "npu"
    torch.manual_seed(42)

    B = 32
    T = 8192
    H = 32
    K = 128
    V = 128
    BT = 64

    return {
        # layouts exactly as pickle
        "q":     torch.randn((32, 1, 8192, 128), device=device, dtype=torch.bfloat16),
        "k":     torch.randn((1, 8192, 32, 128), device=device, dtype=torch.bfloat16),
        "v":     torch.randn((1, 8192, 32, 128), device=device, dtype=torch.bfloat16),
        "v_new": torch.randn((32, 1, 8192, 128), device=device, dtype=torch.bfloat16),
        "g":     torch.randn((1, 8192, 32, 128), device=device, dtype=torch.float32),
        "beta":  torch.randn((32, 1, 8192),      device=device, dtype=torch.float32),
        "A":     torch.randn((32, 1, 8192, 64),  device=device, dtype=torch.bfloat16),
        "h":     torch.randn((1, 128, 32, 128, 128), device=device, dtype=torch.bfloat16),
        "do":    torch.randn((32, 1, 8192, 128), device=device, dtype=torch.bfloat16),
        "dh":    torch.randn((1, 128, 32, 128, 128), device=device, dtype=torch.bfloat16),

        "dq":    torch.zeros((1, 8192, 32, 128), device=device, dtype=torch.float32),
        "dk":    torch.zeros((1, 8192, 32, 128), device=device, dtype=torch.float32),
        "dv":    torch.randn((1, 8192, 32, 128), device=device, dtype=torch.bfloat16),
        "dv2":   torch.zeros((1, 8192, 32, 128), device=device, dtype=torch.bfloat16),
        "dg":    torch.zeros((1, 8192, 32, 128), device=device, dtype=torch.float32),
        "db":    torch.zeros((32, 1, 8192),      device=device, dtype=torch.float32),
        "dA":    torch.zeros((32, 1, 8192, 64),  device=device, dtype=torch.float32),

        "cu_seqlens":   torch.tensor([0, 2048, 6144, 8192], device=device, dtype=torch.int64),
        "chunk_indices":torch.randint(0, 128, (128, 2), device=device, dtype=torch.int64),

        "scale": 0.08838834764831845,
        "scalar": 1.0,
        "T": T,
        "H": H,
        "K": K,
        "V": V,
        "BT": BT,
        "TRANSPOSE_STATE": False,
        "stride_hz": 8192,
        "grid": (128, 32),
    }

# =========================
# Model wrapper
# =========================
class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        grid = data.pop("grid")
        chunk_kda_bwd_kernel_wy_dqkg_fused_opt_v2[grid](**data)
        return (
            data["dq"], data["dk"], data["dv2"],
            data["db"], data["dg"], data["dA"]
        )

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    data = build_inputs()
    model = Model()
    outputs = model.forward(data)

    print("✅ Kernel launched successfully")
    for t in outputs:
        print(t.shape, t.dtype)