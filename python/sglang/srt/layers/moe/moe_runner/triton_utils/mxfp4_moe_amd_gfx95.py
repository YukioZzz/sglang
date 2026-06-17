"""Native MXFP4 (W4A4, 1x32 block, E8M0 scale) MoE for AMD CDNA4 (gfx950).

Mirrors ``mxfp8_moe_amd_gfx95.py`` but for FP4 (``e2m1``) operands: a single
grouped ``tl.dot_scaled`` kernel consumes the packed FP4 weights + their E8M0
block scales directly (no dequant-to-BF16), and activations are MXFP4-quantized
on the fly (aiter ``dynamic_mxfp4_quant``).

  * tokens are sorted by expert with ``moe_align_block_size``;
  * GEMM1 reads the activation by token-id indirection (``a_row = token // top_k``)
    so the hidden states are MXFP4-quantized exactly ONCE (not top_k times);
  * the SwiGLU-OAI activation is the shared split-layout helper (matches the
    MiniMax-M3 dense MLP), re-quantized to MXFP4 before GEMM2;
  * GEMM2 applies the top-k weight inside the kernel and writes each route to a
    distinct output row (no atomics); the final reduction is a strided sum.

This is the native path for models whose MoE activation is SwiGLU-OAI
(uninterleaved), e.g. MiniMax-M3: the AITER W4A4 CK MoE does not implement that
activation, so SGLang otherwise has no native FP4 MoE for it.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    swiglu_no_interleaved_with_alpha_and_limit,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)


def _mxfp4_quant(x: torch.Tensor):
    """Per-group MXFP4 quant -> (packed uint8 [.., K//2], E8M0 uint8 [.., K//32]).

    Uses aiter's ``dynamic_mxfp4_quant`` (the same kernel used for online weight
    quantization in ``QuarkW4A4MXFp4MoE``), so the activation and weight scales
    share the OCP E8M0 convention that ``dot_scaled`` expects.
    """
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    return dynamic_mxfp4_quant(x.contiguous())


@triton.jit
def _mxfp4_grouped_gemm_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    E,
    N,
    K,
    num_valid_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_asm,
    stride_ask,
    stride_be,
    stride_bn,
    stride_bk,
    stride_bse,
    stride_bsn,
    stride_bsk,
    stride_cm,
    stride_cn,
    A_DIV: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # logical K per step (packed FP4 bytes = BLOCK_K // 2)
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_post = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_post:
        return

    offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_tid).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    valid_expert = (off_e >= 0) & (off_e < E)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kp = tl.arange(0, BLOCK_K // 2)   # packed FP4 bytes
    offs_sk = tl.arange(0, BLOCK_K // 32)  # E8M0 scale groups
    a_row = offs_token // A_DIV

    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_kp[None, :] * stride_ak
    as_ptrs = a_scale_ptr + a_row[:, None] * stride_asm + offs_sk[None, :] * stride_ask
    b_ptrs = (
        b_ptr
        + off_e * stride_be
        + offs_n[:, None] * stride_bn
        + offs_kp[None, :] * stride_bk
    )
    bs_ptrs = (
        b_scale_ptr
        + off_e * stride_bse
        + offs_n[:, None] * stride_bsn
        + offs_sk[None, :] * stride_bsk
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_mask = offs_n < N
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0)
        b = tl.load(b_ptrs, mask=valid_expert & n_mask[:, None], other=0)
        asc = tl.load(as_ptrs, mask=token_mask[:, None], other=0)
        bsc = tl.load(bs_ptrs, mask=valid_expert & n_mask[:, None], other=0)
        acc += tl.dot_scaled(a, asc, "e2m1", b.T, bsc, "e2m1")

        a_ptrs += (BLOCK_K // 2) * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
        as_ptrs += (BLOCK_K // 32) * stride_ask
        bs_ptrs += (BLOCK_K // 32) * stride_bsk

    if MUL_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        acc.to(c_ptr.dtype.element_ty),
        mask=token_mask[:, None] & n_mask[None, :],
    )


def _grouped_gemm_mxfp4(
    a_q: torch.Tensor,       # [M, K//2] uint8 (packed e2m1)
    a_scale: torch.Tensor,   # [M, K//32] uint8 (E8M0)
    w: torch.Tensor,         # [E, N, K//2] uint8 (packed e2m1)
    w_scale: torch.Tensor,   # [E, N, K//32] uint8 (E8M0)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    top_k: int,
    block_m: int,
    K: int,                  # logical K
    out_dtype: torch.dtype,
    a_div: int,
    mul_weight_by: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    M_routed = num_valid_tokens
    E, N, Kp = w.shape
    assert K == Kp * 2, f"logical K {K} != packed {Kp}*2"
    assert K % 128 == 0, f"MXFP4 native MoE requires K%128==0, got K={K}"
    out = torch.zeros((M_routed, N), dtype=out_dtype, device=a_q.device)
    if a_div == top_k and M_routed <= 32 and K >= 3072:
        BLOCK_N = 64
        num_warps = 4
    else:
        BLOCK_N = 128
        num_warps = 8
    BLOCK_K = 128
    grid = (triton.cdiv(sorted_token_ids.shape[0], block_m), triton.cdiv(N, BLOCK_N))
    _mxfp4_grouped_gemm_kernel[grid](
        a_q,
        a_scale,
        w,
        w_scale,
        out,
        mul_weight_by if mul_weight_by is not None else a_q,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        E,
        N,
        K,
        num_valid_tokens,
        top_k,
        a_q.stride(0),
        a_q.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        out.stride(0),
        out.stride(1),
        A_DIV=a_div,
        MUL_WEIGHT=mul_weight_by is not None,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out


def fused_moe_mxfp4_native(
    hidden_states: torch.Tensor,  # [T, H] bf16
    w13: torch.Tensor,            # [E, 2I, H//2] uint8 packed e2m1
    w13_scale: torch.Tensor,      # [E, 2I, H//32] uint8 E8M0
    w2: torch.Tensor,             # [E, H, I//2] uint8 packed e2m1
    w2_scale: torch.Tensor,       # [E, H, I//32] uint8 E8M0
    topk_weights: torch.Tensor,   # [T, top_k]
    topk_ids: torch.Tensor,       # [T, top_k] (local expert ids; -1 for non-local EP)
    *,
    alpha: float,
    limit: Optional[float],
    no_combine: bool = False,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    T, H = hidden_states.shape
    top_k = topk_ids.shape[1]
    M = T * top_k
    local_num_experts = w13.shape[0]

    if expert_map is not None:
        valid_global = (topk_ids >= 0) & (topk_ids < expert_map.numel())
        topk_ids = expert_map[topk_ids.clamp(0, expert_map.numel() - 1).long()].to(
            torch.int32
        )
        topk_ids.masked_fill_(
            ~valid_global | (topk_ids < 0) | (topk_ids >= local_num_experts), -1
        )
    else:
        topk_ids = topk_ids.to(torch.int32, copy=True)
        topk_ids.masked_fill_((topk_ids < 0) | (topk_ids >= local_num_experts), -1)

    block_m = 64
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, block_m, local_num_experts
    )

    # GEMM1: x (mxfp4) @ w13^T -> [M, 2I]. Activation quantized ONCE over the T
    # hidden rows; the kernel gathers per route via a_row = token // top_k.
    a_q, a_s = _mxfp4_quant(hidden_states)
    g1 = _grouped_gemm_mxfp4(
        a_q, a_s, w13, w13_scale,
        sorted_ids, expert_ids, num_post, M, top_k, block_m,
        K=H, out_dtype=hidden_states.dtype, a_div=top_k,
    )  # [M, 2I]

    # SwiGLU-OAI (split layout) -- identical to the MiniMax-M3 dense MLP path.
    act = swiglu_no_interleaved_with_alpha_and_limit(g1, alpha, limit)  # [M, I] bf16
    act_q, act_s = _mxfp4_quant(act)
    I = act.shape[1]

    if no_combine:
        g2 = _grouped_gemm_mxfp4(
            act_q, act_s, w2, w2_scale,
            sorted_ids, expert_ids, num_post, M, top_k, block_m,
            K=I, out_dtype=hidden_states.dtype, a_div=1,
        )
        return g2.view(T, top_k, H)

    # GEMM2: act (mxfp4) @ w2^T -> [M, H], weighted by topk_weights, then reduce.
    g2 = _grouped_gemm_mxfp4(
        act_q, act_s, w2, w2_scale,
        sorted_ids, expert_ids, num_post, M, top_k, block_m,
        K=I, out_dtype=torch.float32, a_div=1,
        mul_weight_by=topk_weights.reshape(-1).to(torch.float32),
    )  # [M, H] == [T*top_k, H]

    return g2.view(T, top_k, H).sum(dim=1).to(hidden_states.dtype)


def fused_experts_mxfp4(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    activation: str = "silu",
    is_gated: bool = True,
    no_combine: bool = False,
    inplace: bool = False,
    apply_router_weight_on_input: bool = False,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    interleaved: bool = True,
    expert_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Native MXFP4 MoE entry (CDNA4 ``dot_scaled`` e2m1).

    Mirrors ``fused_experts_mxfp8``; only the MiniMax-M3 SwiGLU-OAI (split,
    uninterleaved, gated silu with ``gemm1_alpha``/``gemm1_limit``) config is
    supported.
    """
    if not (activation == "silu" and is_gated):
        raise NotImplementedError(
            f"native MXFP4 MoE only supports gated swiglu-oai, got "
            f"{activation=} {is_gated=}."
        )
    if apply_router_weight_on_input:
        raise NotImplementedError(
            "native MXFP4 MoE does not support apply_router_weight_on_input."
        )
    if interleaved:
        raise NotImplementedError(
            "native MXFP4 MoE expects uninterleaved (split) gate/up layout."
        )

    alpha = 1.702 if gemm1_alpha is None else float(gemm1_alpha)
    limit = None if gemm1_limit is None else float(gemm1_limit)

    out = fused_moe_mxfp4_native(
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        alpha=alpha,
        limit=limit,
        no_combine=no_combine,
        expert_map=expert_map,
    )

    if no_combine:
        return out
    if inplace:
        hidden_states.copy_(out)
        return hidden_states
    return out
