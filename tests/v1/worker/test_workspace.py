# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import pytest
import torch

import vllm.v1.worker.workspace as workspace
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import _num_workspace_lanes


class _SpecConfig:
    def __init__(self, dspark: bool) -> None:
        self._dspark = dspark

    def use_dspark(self) -> bool:
        return self._dspark


class _VllmConfig:
    def __init__(self, spec_config: _SpecConfig | None) -> None:
        self.speculative_config = spec_config


@pytest.mark.parametrize(
    ("use_v2_model_runner", "spec_config", "expected"),
    [
        (True, _SpecConfig(True), 2),
        (False, _SpecConfig(True), 1),
        (True, _SpecConfig(False), 1),
        (True, None, 1),
    ],
)
def test_workspace_lane_count_is_dspark_only(
    use_v2_model_runner: bool,
    spec_config: _SpecConfig | None,
    expected: int,
) -> None:
    config = cast(VllmConfig, _VllmConfig(spec_config))
    assert _num_workspace_lanes(config, use_v2_model_runner) == expected


def test_workspace_lanes_do_not_alias_and_restore_context(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    assert manager._current_workspaces == [None, None, None, None]

    (target,) = manager.get_simultaneous(((512,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft,) = manager.get_simultaneous(((256,), torch.uint8))
        (draft_reused,) = manager.get_simultaneous(((8,), torch.uint8))
    (target_reused,) = manager.get_simultaneous(((8,), torch.uint8))

    assert manager._current_workspaces[0].numel() == 512  # type: ignore[union-attr]
    assert manager._current_workspaces[1].numel() == 256  # type: ignore[union-attr]
    assert manager._current_workspaces[2:] == [None, None]
    assert target.data_ptr() != draft.data_ptr()
    assert draft.data_ptr() == draft_reused.data_ptr()
    assert target.data_ptr() == target_reused.data_ptr()


def test_workspace_lanes_compose_with_ubatches(monkeypatch) -> None:
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    pointers = set()
    for ubatch_id in range(2):
        active_ubatch[0] = ubatch_id
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                (buffer,) = manager.get_simultaneous(((16,), torch.uint8))
                pointers.add(buffer.data_ptr())

    assert len(pointers) == 4


def test_workspace_lane_validation(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    with (
        pytest.raises(ValueError, match="non-negative"),
        workspace.use_workspace_lane(-1),
    ):
        pass

    with (
        workspace.use_workspace_lane(1),
        pytest.raises(RuntimeError, match="is not configured"),
    ):
        manager.get_simultaneous(((1,), torch.uint8))

    with pytest.raises(ValueError, match="at least one"):
        workspace.WorkspaceManager(torch.device("cpu"), num_lanes=0)


def test_shrink_to_gives_back_only_growth_past_the_mark(monkeypatch) -> None:
    """The mark exists so a discarded phase does not set the floor.

    ``_ensure_workspace_size`` only grows, so a metadata builder constructed to
    profile CUDA graph capture -- sized from a ``max_model_len`` that auto-fit
    has not revised down yet -- would otherwise leave its reservation behind for
    the real builders to inherit.
    """
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=2)

    # Growth from warmup, which must survive: it is what `lock_workspace`
    # relies on having reached the runtime maximum.
    manager.get_simultaneous(((512,), torch.uint8))
    mark = manager.sizes()

    with workspace.use_workspace_lane(1):
        manager.get_simultaneous(((4096,), torch.uint8))
    manager.get_simultaneous(((8192,), torch.uint8))

    manager.shrink_to(mark)

    assert manager._current_workspaces[0].numel() == 512  # type: ignore[union-attr]
    # Lane 1 did not exist at the mark, so all of it is growth.
    assert manager._current_workspaces[1] is None
    # And the floor really did move back: the next request resizes rather than
    # free-riding on the discarded phase's buffer.
    (reused,) = manager.get_simultaneous(((1024,), torch.uint8))
    assert manager._current_workspaces[0].numel() == 1024  # type: ignore[union-attr]
    assert reused.numel() == 1024


def test_shrink_to_leaves_outstanding_views_valid(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    mark = manager.sizes()
    (held,) = manager.get_simultaneous(((256,), torch.uint8))
    held.fill_(7)

    manager.shrink_to(mark)

    assert manager._current_workspaces[0] is None
    assert held.numel() == 256
    assert int(held[0]) == 7
