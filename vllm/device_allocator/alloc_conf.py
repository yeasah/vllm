# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for reading and editing PyTorch's allocator configuration.

Torch's allocator config is parsed as a whole: ``parseArgs`` resets every
option that is not explicitly present in the string it is handed. Writing a
single field therefore silently drops the rest of the user's configuration,
for the remainder of the process:

    >>> # PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
    >>> torch._C._accelerator_setAllocatorSettings("expandable_segments:False")
    >>> # max_split_size is now SIZE_MAX, not 512 MiB -- and writing
    >>> # "expandable_segments:True" back does not bring it home either.

So anything that wants to toggle one option has to read the current config,
flip just that field, and write the whole string back.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

import torch

# Checked in torch's own precedence order: the accelerator-agnostic variable
# wins, then the CUDA one, then the HIP one (which is what ROCm users set).
ALLOC_CONF_ENV_VARS = (
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
)

EXPANDABLE_SEGMENTS = "expandable_segments"


def alloc_conf_from_env(
    environ: "os._Environ[str] | dict[str, str] | None" = None,
) -> str:
    """The allocator config string the process was started with, if any."""
    env = os.environ if environ is None else environ
    for name in ALLOC_CONF_ENV_VARS:
        conf = env.get(name)
        if conf:
            return conf
    return ""


def live_alloc_conf() -> str | None:
    """The allocator config string currently in effect, or ``None``.

    Read from the live allocator rather than the environment, so that a
    *runtime* write by another component is visible.
    """
    try:
        settings = torch.cuda.memory._snapshot().get("allocator_settings")
        if settings is not None:
            for name in ALLOC_CONF_ENV_VARS:
                conf = settings.get(name)
                if conf is not None:
                    return conf
    except Exception:  # noqa: BLE001
        pass
    return None


def current_alloc_conf() -> str:
    """The live allocator config, falling back to the environment."""
    live = live_alloc_conf()
    return live if live is not None else alloc_conf_from_env()


def conf_flag_enabled(conf: str, key: str) -> bool:
    """Parse ``<key>:<bool>`` out of an allocator config string."""
    for field in conf.split(","):
        name, _, value = field.partition(":")
        if name.strip() == key:
            return value.strip().lower() in ("true", "1")
    return False


def with_conf_flag(conf: str, key: str, enabled: bool) -> str:
    """Return ``conf`` with only ``key`` flipped, every other field verbatim."""
    want = f"{key}:{'True' if enabled else 'False'}"
    fields = [f for f in (f.strip() for f in conf.split(",")) if f]
    for i, field in enumerate(fields):
        if field.partition(":")[0].strip() == key:
            fields[i] = want
            return ",".join(fields)
    fields.append(want)
    return ",".join(fields)


def expandable_segments_enabled_from_env(
    environ: "os._Environ[str] | dict[str, str] | None" = None,
) -> bool:
    return conf_flag_enabled(alloc_conf_from_env(environ), EXPANDABLE_SEGMENTS)


def expandable_segments_enabled() -> bool | None:
    """Live ``expandable_segments`` state, or ``None`` if it cannot be read.

    ``None`` is deliberately distinct from ``False``: the environment can never
    reflect a runtime write, so a caller verifying a write it just issued must
    not read "cannot tell" as "the write did not take".
    """
    try:
        settings = torch.cuda.memory._snapshot().get("allocator_settings")
        if settings is not None and EXPANDABLE_SEGMENTS in settings:
            return bool(settings[EXPANDABLE_SEGMENTS])
    except Exception:  # noqa: BLE001
        pass
    return None


def set_alloc_conf(conf: str) -> None:
    """Write a complete allocator config string."""
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is not None:
        setter(conf)
        return
    torch.cuda.memory._set_allocator_settings(conf)


@contextmanager
def non_expandable_allocations() -> Generator[None, None, None]:
    """Run the body with ``expandable_segments`` off, then restore the config."""
    conf = current_alloc_conf()
    if not conf_flag_enabled(conf, EXPANDABLE_SEGMENTS):
        yield
        return
    set_alloc_conf(with_conf_flag(conf, EXPANDABLE_SEGMENTS, False))
    try:
        yield
    finally:
        set_alloc_conf(conf)


#: pytorch/pytorch@3fda599, the fix for PyTorch #196258 that our tools/torch-c10cuda-fix.sh applies to the
#: installed torch, leaving `libc10_cuda.so.fix-<this sha>` beside the library with the patched build's sha256
EXPANDABLE_SEGMENTS_FIX = "3fda5993c2572eca77dd53033f057dc0290f287f"


def expandable_segments_fix_present() -> bool:
    """Whether torch's libc10_cuda.so carries the PyTorch #196258 fix, as its stamp records: with expandable
    segments, empty_cache() unmapped freed pages on every device but synchronized only the current one."""
    import hashlib

    lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libc10_cuda.so")
    try:
        with open(f"{lib}.fix-{EXPANDABLE_SEGMENTS_FIX}") as f:
            want = f.read().split()[0]
        h = hashlib.sha256()
        with open(lib, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest() == want
    except (OSError, IndexError):
        return False


def maybe_default_expandable_segments(tensor_parallel_size: int = 1) -> bool:
    """Turn expandable segments on when the user left them unset, torch carries the #196258 fix and there is no
    tensor parallelism; returns whether it did. Call before the process allocates, so every segment is expandable.

    They let torch's reservation track what is allocated instead of caching one segment per shape: on 16 GiB
    cards the extensible KV cache's measured reservation growth fell from 0.83 to 0.20 GiB (Qwen3.6-35B-A3B
    EXL3) and from 0.75 to 0.27 (Laguna-XS), memory it no longer has to keep free. An explicit
    expandable_segments setting, either way, stands.

    Not under tensor parallelism: custom all-reduce registers CUDA-graph buffers with the legacy
    cudaIpcGetMemHandle, which cannot export expandable segments' virtual-memory allocations, so graph capture
    fails on any box with peer access ('invalid argument', csrc/custom_all_reduce.cuh), and the TP group's other
    IPC backends are unchecked under them.

    Not under WSL either: WDDM's driver path fails expandable segments' memory mapping with CUDA_ERROR_NOT_READY
    ("CUDA driver error: device not ready") on some allocation patterns -- gemma-4-12B EXL3 on an RTX 3060 failed
    to load every time with them on, and loaded with them off."""
    if tensor_parallel_size > 1:
        return False
    from vllm.platforms.interface import in_wsl

    if in_wsl():
        return False
    if EXPANDABLE_SEGMENTS in alloc_conf_from_env() or not expandable_segments_fix_present():
        return False
    set_alloc_conf(with_conf_flag(current_alloc_conf(), EXPANDABLE_SEGMENTS, True))
    return True
