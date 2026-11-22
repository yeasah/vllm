# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.device_allocator import alloc_conf

pytestmark = pytest.mark.cpu_test

ES = alloc_conf.EXPANDABLE_SEGMENTS


@pytest.mark.parametrize(
    "environ,expected",
    [
        ({}, False),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:True"}, True),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:true"}, True),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:1"}, True),
        ({"PYTORCH_ALLOC_CONF": "expandable_segments:False"}, False),
        ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}, True),
        # ROCm users set this one; it was previously not looked at.
        ({"PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}, True),
        (
            {"PYTORCH_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True"},
            True,
        ),
        ({"PYTORCH_ALLOC_CONF": "max_split_size_mb:512"}, False),
        ({"PYTORCH_ALLOC_CONF": "not_expandable_segments_really:True"}, False),
        ({"PYTORCH_ALLOC_CONF": ""}, False),
        # torch's precedence: the accelerator-agnostic variable wins.
        (
            {
                "PYTORCH_ALLOC_CONF": "expandable_segments:False",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            False,
        ),
    ],
)
def test_expandable_segments_from_env(environ, expected):
    assert alloc_conf.expandable_segments_enabled_from_env(environ) is expected


@pytest.mark.parametrize(
    "conf,key,expected",
    [
        ("", ES, False),
        ("expandable_segments:True", ES, True),
        ("expandable_segments:TRUE", ES, True),
        ("expandable_segments:1", ES, True),
        ("expandable_segments:0", ES, False),
        (" expandable_segments : True ", ES, True),
        ("max_split_size_mb:512", ES, False),
        ("max_split_size_mb:512", "max_split_size_mb", False),
        ("garbage_collection_threshold:0.9", ES, False),
    ],
)
def test_conf_flag_enabled(conf, key, expected):
    assert alloc_conf.conf_flag_enabled(conf, key) is expected


@pytest.mark.parametrize(
    "conf,enabled,expected",
    [
        ("", False, "expandable_segments:False"),
        ("expandable_segments:True", False, "expandable_segments:False"),
        ("expandable_segments:False", True, "expandable_segments:True"),
        (
            "max_split_size_mb:512,expandable_segments:True",
            False,
            "max_split_size_mb:512,expandable_segments:False",
        ),
        (
            "garbage_collection_threshold:0.9,max_split_size_mb:512",
            False,
            (
                "garbage_collection_threshold:0.9,max_split_size_mb:512,"
                "expandable_segments:False"
            ),
        ),
    ],
)
def test_with_conf_flag_preserves_other_fields(conf, enabled, expected):
    assert alloc_conf.with_conf_flag(conf, ES, enabled) == expected


def test_with_conf_flag_round_trips():
    """The regression this guards: torch resets every option absent from the
    string it is handed, so a one-field write drops the rest permanently."""
    conf = (
        "max_split_size_mb:512,expandable_segments:True,"
        "garbage_collection_threshold:0.9"
    )
    off = alloc_conf.with_conf_flag(conf, ES, False)
    assert "max_split_size_mb:512" in off
    assert "garbage_collection_threshold:0.9" in off
    assert alloc_conf.with_conf_flag(off, ES, True) == conf


def test_alloc_conf_from_env_precedence():
    assert (
        alloc_conf.alloc_conf_from_env(
            {
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
                "PYTORCH_HIP_ALLOC_CONF": "max_split_size_mb:128",
            }
        )
        == "max_split_size_mb:256"
    )
    assert alloc_conf.alloc_conf_from_env({}) == ""


def test_current_alloc_conf_prefers_live_state(monkeypatch):
    monkeypatch.setattr(alloc_conf, "live_alloc_conf", lambda: "max_split_size_mb:64")
    monkeypatch.setattr(alloc_conf, "alloc_conf_from_env", lambda: "stale:True")
    assert alloc_conf.current_alloc_conf() == "max_split_size_mb:64"

    monkeypatch.setattr(alloc_conf, "live_alloc_conf", lambda: None)
    assert alloc_conf.current_alloc_conf() == "stale:True"


def test_expandable_segments_enabled_returns_none_when_unreadable(monkeypatch):
    """'cannot read' must stay distinct from 'disabled'."""

    class _NoSnapshot:
        @staticmethod
        def _snapshot():
            raise RuntimeError("no allocator here")

    monkeypatch.setattr(alloc_conf.torch.cuda, "memory", _NoSnapshot)
    assert alloc_conf.expandable_segments_enabled() is None


def test_non_expandable_allocations(monkeypatch):
    conf = "max_split_size_mb:512,expandable_segments:True"
    writes: list[str] = []
    monkeypatch.setattr(
        alloc_conf, "current_alloc_conf", lambda: writes[-1] if writes else conf
    )
    monkeypatch.setattr(alloc_conf, "set_alloc_conf", writes.append)
    with pytest.raises(RuntimeError), alloc_conf.non_expandable_allocations():
        assert writes == ["max_split_size_mb:512,expandable_segments:False"]
        with alloc_conf.non_expandable_allocations():  # already off: no-op
            pass
        raise RuntimeError("capture failed")
    assert writes == ["max_split_size_mb:512,expandable_segments:False", conf]


def test_expandable_segments_default_needs_the_fix_and_an_unset_flag(monkeypatch):
    """Expandable segments go on by default only where torch carries the PyTorch #196258 fix (its stamp
    verifies) and the user left the flag unset; an explicit setting either way stands."""
    from vllm.device_allocator import alloc_conf

    written = []
    monkeypatch.setattr(alloc_conf, "set_alloc_conf", written.append)
    monkeypatch.setattr(alloc_conf, "current_alloc_conf", lambda: "max_split_size_mb:512")
    for var in alloc_conf.ALLOC_CONF_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setattr(alloc_conf, "expandable_segments_fix_present", lambda: False)
    assert not alloc_conf.maybe_default_expandable_segments() and written == []

    monkeypatch.setattr(alloc_conf, "expandable_segments_fix_present", lambda: True)
    assert alloc_conf.maybe_default_expandable_segments()
    assert alloc_conf.conf_flag_enabled(written[-1], "expandable_segments")
    assert "max_split_size_mb:512" in written[-1], "the rest of the config survives"

    written.clear()
    assert not alloc_conf.maybe_default_expandable_segments(tensor_parallel_size=2) and written == [], (
        "custom all-reduce's legacy IPC cannot export expandable segments"
    )
    for value in ("False", "True"):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", f"expandable_segments:{value}")
        assert not alloc_conf.maybe_default_expandable_segments() and written == []


def test_expandable_segments_fix_probe_checks_the_library_against_its_stamp(monkeypatch, tmp_path):
    import hashlib

    from vllm.device_allocator import alloc_conf

    (tmp_path / "lib").mkdir()
    lib = tmp_path / "lib" / "libc10_cuda.so"
    lib.write_bytes(b"patched")
    monkeypatch.setattr(alloc_conf.torch, "__file__", str(tmp_path / "__init__.py"))
    assert not alloc_conf.expandable_segments_fix_present()
    stamp = f"{lib}.fix-{alloc_conf.EXPANDABLE_SEGMENTS_FIX}"
    open(stamp, "w").write(hashlib.sha256(b"patched").hexdigest() + "  -\n")
    assert alloc_conf.expandable_segments_fix_present()
    lib.write_bytes(b"reinstalled")
    assert not alloc_conf.expandable_segments_fix_present(), "a stamp beside a reinstalled torch does not count"
