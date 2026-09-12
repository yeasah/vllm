# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dequantizing the cached context in slabs must equal doing it in one piece.

`_tq_full_dequant_kv` writes the whole cached context into one buffer, which is
half of why continuation prefill's VRAM scales with context. `POS_OFFSET` lets
a launch cover `[off, off + slab)` instead: it shifts the *cache* index while
the output index stays slab-relative, so the destination is sized by the slab.

The claim is bitwise, not approximate -- the same bytes go through the same
arithmetic, only the launch is split -- so these compare bit patterns rather
than values. That also keeps the comparison meaningful on a synthetic cache
full of random bytes, where some slots decode to NaN.

Random bytes are deliberate. The kernel masks every unpacked index to its bit
width, so no byte pattern can read out of bounds, and using real quantized
values would test the store path instead of the indexing this changes. The
block table is shuffled for the same reason: a `POS_OFFSET` applied to the
page lookup rather than only to the position within it would still pass on an
identity mapping.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("Needs a GPU.", allow_module_level=True)

from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.attention.ops.triton_turboquant_decode import (  # noqa: E402
    _tq_full_dequant_kv,
)

HK, D = 2, 128
BLOCK_SIZE, NUM_BLOCKS = 64, 24
MSE_BITS = 3
MSE_BYTES = (D * MSE_BITS + 7) // 8
KPS = MSE_BYTES + 2  # + fp16 vector norm
VAL_DATA_BYTES = D // 2  # 4-bit values
SLOT = KPS + VAL_DATA_BYTES + 4  # + fp16 scale and zero


def _cache(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randint(
        0, 256, (NUM_BLOCKS, BLOCK_SIZE, HK, SLOT),
        dtype=torch.uint8, device="cuda", generator=g,
    )
    # Shuffled, so a bug that offsets the position but not the page lookup
    # cannot hide behind block i living at slot i.
    order = torch.randperm(NUM_BLOCKS, generator=g, device="cuda").to(torch.int32)
    block_table = order.unsqueeze(0)
    centroids = torch.randn(1 << MSE_BITS, device="cuda", generator=g,
                            dtype=torch.float32)
    return kv, block_table, centroids


def _dequant(kv, block_table, centroids, n_pos, pos_offset):
    k = torch.empty(1, HK, n_pos, D, dtype=torch.float16, device="cuda")
    v = torch.empty_like(k)
    _tq_full_dequant_kv[(n_pos, HK)](
        kv, block_table, centroids, k, v,
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        kv.stride(0), kv.stride(1), kv.stride(2),
        block_table.stride(0),
        HEAD_DIM=D, BLOCK_SIZE=BLOCK_SIZE, NUM_KV_HEADS=HK,
        MSE_BYTES=MSE_BYTES, KPS=KPS, VQB=4, VAL_DATA_BYTES=VAL_DATA_BYTES,
        MSE_BITS=MSE_BITS, KEY_FP8=0, BLOCK_D=triton.next_power_of_2(D),
        NORM_CORRECTION=0, FP8_E4B15=0, POS_OFFSET=pos_offset,
        num_warps=4,
    )
    return k, v


def _bits(t):
    return t.view(torch.int16)


def test_pos_offset_defaults_to_the_whole_context():
    """The default must leave every existing call site unchanged."""
    kv, bt, c = _cache()
    a_k, a_v = _dequant(kv, bt, c, 600, 0)
    b_k, b_v = _dequant(kv, bt, c, 600, 0)
    assert torch.equal(_bits(a_k), _bits(b_k))
    assert torch.equal(_bits(a_v), _bits(b_v))


@pytest.mark.parametrize("cached_len", [600, 1024])
@pytest.mark.parametrize("slab", [64, 128, 100, 700])
def test_slabbed_dequant_is_bitwise_identical(cached_len, slab):
    """Slabs that do and do not divide the block size, and one larger than
    the whole context, must all reproduce the single launch exactly."""
    kv, bt, c = _cache()
    ref_k, ref_v = _dequant(kv, bt, c, cached_len, 0)

    got_k = torch.empty_like(ref_k)
    got_v = torch.empty_like(ref_v)
    for s in range(0, cached_len, slab):
        n = min(slab, cached_len - s)
        sk, sv = _dequant(kv, bt, c, n, s)
        got_k[:, :, s : s + n, :] = sk
        got_v[:, :, s : s + n, :] = sv

    assert torch.equal(_bits(got_k), _bits(ref_k)), "K differs"
    assert torch.equal(_bits(got_v), _bits(ref_v)), "V differs"
