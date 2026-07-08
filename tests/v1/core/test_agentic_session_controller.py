# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.core.agentic_session_controller import SessionController
from vllm.v1.engine import ContextManagementEditsParams, ContextManagementParams

pytestmark = pytest.mark.cpu_test


# ==================== Mock Helpers ====================


class _CallRecorder:
    """Helper to record callback invocations for testing."""

    def __init__(self, name: str):
        self.name = name
        self.calls: list[tuple[list[int], str]] = []

    def __call__(self, block_ids: list[int], session_id: str):
        self.calls.append((block_ids, session_id))

    def reset(self):
        self.calls.clear()


def _make_callbacks():
    offload = _CallRecorder("offload")
    prefetch = _CallRecorder("prefetch")
    evict = _CallRecorder("evict")
    callbacks = {
        "execute_offload": offload,
        "execute_prefetch": prefetch,
        "execute_evict": evict,
    }
    return callbacks, offload, prefetch, evict


@pytest.fixture
def controller():
    callbacks, offload, prefetch, evict = _make_callbacks()
    ctrl = SessionController(callbacks=callbacks)
    return ctrl, offload, prefetch, evict


def _make_cm(
    edit_type: str,
    block_start: int,
    block_end: int,
    manage_request: bool = True,
    session_id: str = "s1",
) -> ContextManagementParams:
    """Helper to build a ContextManagementParams with a single edit."""
    edit = ContextManagementEditsParams(
        type=edit_type, block_start=block_start, block_end=block_end)
    return ContextManagementParams(manage_request=manage_request, edits=[edit])


# ==================== Registry Tests ====================


def test_missing_required_callbacks_raises():
    with pytest.raises(ValueError, match="Missing required callbacks"):
        SessionController(callbacks={"execute_offload": lambda b, s: None})


def test_init_with_kw_callbacks():
    offload = _CallRecorder("offload")
    prefetch = _CallRecorder("prefetch")
    evict = _CallRecorder("evict")
    ctrl = SessionController(
        execute_offload=offload,
        execute_prefetch=prefetch,
        execute_evict=evict,
    )
    assert len(ctrl._registry) >= 3


def test_register_and_use_custom_callback():
    custom = _CallRecorder("custom")
    callbacks, _, _, _ = _make_callbacks()
    ctrl = SessionController(callbacks=callbacks)
    ctrl.register("execute_custom", custom)

    fn = ctrl._registry.get("execute_custom")
    assert fn is not None
    fn([1, 2], "s1")
    assert len(custom.calls) == 1


def test_unregister_optional_callback():
    callbacks, _, _, _ = _make_callbacks()
    ctrl = SessionController(callbacks=callbacks)
    ctrl.register("temp_fn", lambda: None)
    ctrl.unregister("temp_fn")
    assert "temp_fn" not in ctrl._registry


def test_unregister_required_callback_raises():
    callbacks, _, _, _ = _make_callbacks()
    ctrl = SessionController(callbacks=callbacks)
    with pytest.raises(ValueError, match="Cannot unregister"):
        ctrl.unregister("execute_offload")


# ==================== process_request_edits: edge cases ====================


def test_none_context_management_noop(controller):
    ctrl, offload, prefetch, evict = controller
    ctrl.process_request_edits("req-1", "s1", None)
    assert offload.calls == []
    assert prefetch.calls == []
    assert evict.calls == []


def test_none_edits_noop(controller):
    ctrl, offload, _, _ = controller
    cm = ContextManagementParams(edits=None)
    ctrl.process_request_edits("req-1", "s1", cm)
    assert offload.calls == []


def test_none_session_id_skips(controller):
    ctrl, offload, _, _ = controller
    cm = _make_cm("offload", 0, 2)
    ctrl.process_request_edits("req-1", None, cm)
    assert offload.calls == []


def test_no_block_range_skips(controller):
    """Edit without block_start/block_end should be skipped."""
    ctrl, offload, _, _ = controller
    edit = ContextManagementEditsParams(type="offload")  # block_start/end default None
    cm = ContextManagementParams(manage_request=True, edits=[edit])
    ctrl.process_request_edits("req-1", "s1", cm)
    assert offload.calls == []


# ==================== manage_request=True (immediate) ====================


def test_manage_request_offload(controller):
    ctrl, offload, prefetch, evict = controller
    cm = _make_cm("offload", 1, 3)

    ctrl.process_request_edits("req-1", "s1", cm)

    assert offload.calls == [([1, 2, 3], "s1")]
    assert prefetch.calls == []
    assert evict.calls == []


def test_manage_request_prefetch(controller):
    ctrl, offload, prefetch, evict = controller
    cm = _make_cm("prefetch", 0, 1)

    ctrl.process_request_edits("req-1", "s2", cm)

    assert prefetch.calls == [([0, 1], "s2")]
    assert offload.calls == []
    assert evict.calls == []


def test_manage_request_evict(controller):
    ctrl, offload, prefetch, evict = controller
    cm = _make_cm("evict", 5, 7)

    ctrl.process_request_edits("req-1", "s3", cm)

    assert evict.calls == [([5, 6, 7], "s3")]
    assert offload.calls == []
    assert prefetch.calls == []


def test_manage_request_multiple_edits(controller):
    """A single request can carry multiple edits of different types."""
    ctrl, offload, prefetch, evict = controller
    edits = [
        ContextManagementEditsParams(type="offload", block_start=0, block_end=1),
        ContextManagementEditsParams(type="prefetch", block_start=2, block_end=3),
        ContextManagementEditsParams(type="evict", block_start=4, block_end=5),
    ]
    cm = ContextManagementParams(manage_request=True, edits=edits)

    ctrl.process_request_edits("req-1", "s1", cm)

    assert offload.calls == [([0, 1], "s1")]
    assert prefetch.calls == [([2, 3], "s1")]
    assert evict.calls == [([4, 5], "s1")]


# ==================== manage_request=False (deferred) ====================


def test_deferred_not_executed_immediately(controller):
    ctrl, offload, _, _ = controller
    cm = _make_cm("offload", 0, 2, manage_request=False)

    ctrl.process_request_edits("req-1", "s1", cm)

    assert offload.calls == []
    assert "req-1" in ctrl._pending_edits


def test_deferred_executed_on_completion(controller):
    ctrl, offload, _, _ = controller
    cm = _make_cm("offload", 0, 2, manage_request=False)

    ctrl.process_request_edits("req-1", "s1", cm)
    assert offload.calls == []

    ctrl.on_request_completed("req-1")

    assert offload.calls == [([0, 1, 2], "s1")]
    assert "req-1" not in ctrl._pending_edits


def test_on_request_completed_no_pending_noop(controller):
    ctrl, _, _, _ = controller
    ctrl.on_request_completed("nonexistent")  # should not raise


def test_deferred_session_id_preserved(controller):
    """session_id registered at deferral time is used at execution time."""
    ctrl, offload, _, _ = controller
    cm = _make_cm("offload", 0, 0, manage_request=False)

    ctrl.process_request_edits("req-1", "session-abc", cm)
    ctrl.on_request_completed("req-1")

    assert offload.calls[0] == ([0], "session-abc")


def test_deferred_multiple_edits(controller):
    """Deferred request with multiple edits executes all on completion."""
    ctrl, offload, prefetch, evict = controller
    edits = [
        ContextManagementEditsParams(type="offload", block_start=0, block_end=1),
        ContextManagementEditsParams(type="prefetch", block_start=2, block_end=3),
    ]
    cm = ContextManagementParams(manage_request=False, edits=edits)

    ctrl.process_request_edits("req-1", "s1", cm)
    assert offload.calls == []
    assert prefetch.calls == []

    ctrl.on_request_completed("req-1")

    assert offload.calls == [([0, 1], "s1")]
    assert prefetch.calls == [([2, 3], "s1")]


# ==================== block_id range ====================


def test_block_range_single_block(controller):
    ctrl, offload, _, _ = controller
    cm = _make_cm("offload", 5, 5)

    ctrl.process_request_edits("req-1", "s1", cm)

    assert offload.calls[0] == ([5], "s1")


def test_block_range_multiple_blocks(controller):
    ctrl, offload, _, _ = controller
    cm = _make_cm("offload", 2, 5)

    ctrl.process_request_edits("req-1", "s1", cm)

    assert offload.calls[0] == ([2, 3, 4, 5], "s1")


# ==================== Integration ====================


def test_manage_then_deferred_complete():
    """Manage request (immediate) + deferred request, then complete."""
    callbacks, offload, prefetch, _ = _make_callbacks()
    ctrl = SessionController(callbacks=callbacks)

    # Manage request: immediate offload
    ctrl.process_request_edits(
        "req-manage", "s1",
        _make_cm("offload", 0, 2, manage_request=True))
    assert offload.calls == [([0, 1, 2], "s1")]

    # Normal request: deferred prefetch
    ctrl.process_request_edits(
        "req-normal", "s1",
        _make_cm("prefetch", 3, 4, manage_request=False))
    assert prefetch.calls == []

    # Complete normal request
    ctrl.on_request_completed("req-normal")
    assert prefetch.calls == [([3, 4], "s1")]


def test_multiple_deferred_requests_same_session():
    """Multiple deferred requests for same session complete in order."""
    callbacks, offload, _, _ = _make_callbacks()
    ctrl = SessionController(callbacks=callbacks)

    ctrl.process_request_edits(
        "req-1", "s1",
        _make_cm("offload", 0, 1, manage_request=False))
    ctrl.process_request_edits(
        "req-2", "s1",
        _make_cm("offload", 2, 3, manage_request=False))

    ctrl.on_request_completed("req-1")
    assert offload.calls == [([0, 1], "s1")]

    ctrl.on_request_completed("req-2")
    assert offload.calls == [([0, 1], "s1"), ([2, 3], "s1")]


def test_manage_then_deferred_different_session():
    """Manage request for session A, deferred for session B."""
    callbacks, offload, prefetch, evict = _make_callbacks()
    ctrl = SessionController(callbacks=callbacks)

    ctrl.process_request_edits(
        "req-manage", "sA",
        _make_cm("evict", 0, 1, manage_request=True))
    assert evict.calls == [([0, 1], "sA")]

    ctrl.process_request_edits(
        "req-normal", "sB",
        _make_cm("offload", 2, 3, manage_request=False))
    ctrl.on_request_completed("req-normal")
    assert offload.calls == [([2, 3], "sB")]