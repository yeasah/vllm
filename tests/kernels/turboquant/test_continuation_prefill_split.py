# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The invariant that lets TurboQuant's continuation prefill be chunked.

``_continuation_prefill`` attends ``q_len`` queries to ``cached_len + q_len``
keys under one causal flash-attention call, which requires the whole cached
context dequantized into one buffer -- the allocation that makes prefill VRAM
scale with context rather than with the chunk.

Splitting the keys removes that, and the split point is not arbitrary. Every
query attends to *all* of the cached prefix, so the prefix is unmasked and may
be cut anywhere; only the current chunk is causal, and it is causal against
itself. These tests pin that reading: the prefix may be sliced into slabs of
any size and merged by log-sum-exp, and the result must match the single call
that materializes everything.

The oracle is fp32 SDPA under the explicit mask the SDPA fallback builds
(``k_pos <= q_pos`` with ``q_pos`` offset by ``cached_len``), so a mistake in
the causal alignment fails against arithmetic rather than against another
flash-attention call that could share the error.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Needs a GPU.", allow_module_level=True)

from vllm.v1.attention.backends.fa_utils import (  # noqa: E402
    is_flash_attn_varlen_func_available,
)

if not is_flash_attn_varlen_func_available():
    pytest.skip("Needs flash_attn_varlen_func.", allow_module_level=True)

from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func  # noqa: E402
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states  # noqa: E402

DTYPE = torch.bfloat16
HQ, HK, D = 8, 2, 128
Q_LEN = 64


def _fa(q, k, v, causal, scale, lse=False):
    n = k.shape[0]
    dev = q.device
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.tensor([0, q.shape[0]], device=dev, dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, n], device=dev, dtype=torch.int32),
        max_seqlen_q=q.shape[0],
        max_seqlen_k=n,
        softmax_scale=scale,
        causal=causal,
        return_softmax_lse=lse,
    )


def _oracle(q, k, v, cached_len, scale):
    """fp32 SDPA under the mask the SDPA fallback builds."""
    q_len = q.shape[0]
    dev = q.device
    q_t = q.float().transpose(0, 1).unsqueeze(0)
    k_t = k.float().transpose(0, 1).unsqueeze(0)
    v_t = v.float().transpose(0, 1).unsqueeze(0)
    q_pos = torch.arange(q_len, device=dev).unsqueeze(1) + cached_len
    k_pos = torch.arange(k.shape[0], device=dev).unsqueeze(0)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=k_pos <= q_pos, scale=scale, enable_gqa=(HK < HQ)
    )
    return out[0].transpose(0, 1)


def _split(q, k, v, cached_len, scale, slab):
    """Prefix in slabs (unmasked) + current chunk (causal), merged by LSE.

    The accumulator is fp32 on purpose: merging rounds to the accumulator's
    dtype, so a bf16 one would compound one rounding step per slab.
    """
    acc_o = acc_lse = None
    for s in range(0, cached_len, slab):
        e = min(s + slab, cached_len)
        o_i, lse_i = _fa(q, k[s:e], v[s:e], False, scale, lse=True)
        if acc_o is None:
            acc_o, acc_lse = o_i.float(), lse_i
        else:
            n_o = torch.empty_like(acc_o)
            n_lse = torch.empty_like(acc_lse)
            merge_attn_states(n_o, acc_o, acc_lse, o_i.float(), lse_i, output_lse=n_lse)
            acc_o, acc_lse = n_o, n_lse
    suf_o, suf_lse = _fa(q, k[cached_len:], v[cached_len:], True, scale, lse=True)
    out32 = torch.empty(q.shape[0], HQ, D, device=q.device, dtype=torch.float32)
    merge_attn_states(out32, acc_o, acc_lse, suf_o.float(), suf_lse)
    return out32.to(q.dtype)


def _ulp(ref):
    """One bf16 quantum at ``ref``'s magnitude.

    Tolerances here must scale with the output, not be absolute: attention
    over a short context averages fewer values and so produces larger ones,
    and a fixed bound that fits cached_len=4096 is four times too tight at
    512.
    """
    return ref.abs().max().item() * 2.0**-8


def _inputs(cached_len, seed=0):
    torch.manual_seed(seed)
    seq_len = cached_len + Q_LEN
    dev = "cuda"
    q = torch.randn(Q_LEN, HQ, D, device=dev, dtype=DTYPE) * 0.5
    k = torch.randn(seq_len, HK, D, device=dev, dtype=DTYPE) * 0.5
    v = torch.randn(seq_len, HK, D, device=dev, dtype=DTYPE) * 0.5
    return q, k, v, D**-0.5


@pytest.mark.parametrize("cached_len", [512, 1000, 4096])
def test_single_call_matches_the_masked_oracle(cached_len):
    """Guards the reading of causal alignment the split depends on.

    flash-attn aligns a shorter query to the *end* of the key sequence, so
    query i sees keys [0, cached_len + i]. If that were top-left aligned
    instead, the prefix would not be fully unmasked and no split would be
    valid.
    """
    q, k, v, scale = _inputs(cached_len)
    got = _fa(q, k, v, True, scale)
    ref = _oracle(q, k, v, cached_len, scale)
    assert torch.allclose(got.float(), ref, atol=2e-3, rtol=0), (
        f"max {(got.float() - ref).abs().max():.3e}"
    )


@pytest.mark.parametrize("cached_len", [512, 1000, 4096])
@pytest.mark.parametrize("slab", [256, 1024, 8192])
def test_split_matches_the_single_call(cached_len, slab):
    """Any slab size, including one larger than the prefix and one that does
    not divide it, must reproduce the call that materializes everything."""
    q, k, v, scale = _inputs(cached_len)
    ref = _fa(q, k, v, True, scale)
    got = _split(q, k, v, cached_len, scale, slab)
    delta = (got.float() - ref.float()).abs().max().item()
    # A few bf16 quanta. The fp32 accumulator is what keeps this from scaling
    # with the number of slabs -- see test_split_does_not_drift_with_slab_count.
    tol = 4 * _ulp(ref.float())
    assert delta <= tol, f"max abs {delta:.3e} = {delta / _ulp(ref.float()):.1f} ULP"

    # And it must be no worse than the single call is against real arithmetic,
    # which is the claim that actually matters and is free of any tolerance.
    oracle = _oracle(q, k, v, cached_len, scale)
    err_single = (ref.float() - oracle).abs().max().item()
    err_split = (got.float() - oracle).abs().max().item()
    assert err_split <= 2 * err_single, f"split {err_split:.3e} vs single {err_single:.3e}"


def test_split_does_not_drift_with_slab_count():
    """The merge must not accumulate error as slabs get smaller -- otherwise
    the slab size becomes a numerics knob rather than a memory one."""
    cached_len = 4096
    q, k, v, scale = _inputs(cached_len)
    ref = _fa(q, k, v, True, scale)
    deltas = {
        slab: (_split(q, k, v, cached_len, scale, slab).float() - ref.float())
        .abs()
        .max()
        .item()
        for slab in (4096, 1024, 256)
    }
    assert deltas[256] <= 4 * deltas[4096] + _ulp(ref.float()), deltas
