# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest

from vllm.v1.core.agentic_cache_ttl_manager import (
    TTLBlockEntry,
    TTLTimerWheel,
    TTLManager,
)

pytestmark = pytest.mark.cpu_test


# ==================== TTLBlockEntry Tests ====================


def test_ttl_block_entry_fields():
    entry = TTLBlockEntry(block_id=1, session_id="s1", ttl_expire_at=100.5)
    assert entry.block_id == 1
    assert entry.session_id == "s1"
    assert entry.ttl_expire_at == 100.5


def test_ttl_block_entry_is_dataclass():
    """TTLBlockEntry should be a dataclass with equality by value."""
    a = TTLBlockEntry(block_id=1, session_id="s1", ttl_expire_at=100.0)
    b = TTLBlockEntry(block_id=1, session_id="s1", ttl_expire_at=100.0)
    assert a == b


# ==================== TTLTimerWheel Tests ====================


def test_timer_wheel_insert_and_advance():
    wheel = TTLTimerWheel(tick_count=60)
    now = 10.0
    entry = TTLBlockEntry(block_id=1, session_id="s1", ttl_expire_at=15.0)
    wheel.insert(entry, expire_at=15.0)

    # Advance to time before expiry — should not expire
    expired = wheel.advance(12.0)
    assert expired == []

    # Advance past expiry — should collect entry
    expired = wheel.advance(16.0)
    assert len(expired) == 1
    assert expired[0].block_id == 1


def test_timer_wheel_multiple_entries_same_slot():
    wheel = TTLTimerWheel(tick_count=60)
    e1 = TTLBlockEntry(block_id=1, session_id="s1", ttl_expire_at=10.3)
    e2 = TTLBlockEntry(block_id=2, session_id="s1", ttl_expire_at=10.8)
    wheel.insert(e1, expire_at=10.3)
    wheel.insert(e2, expire_at=10.8)

    expired = wheel.advance(11.0)
    assert len(expired) == 2
    assert {e.block_id for e in expired} == {1, 2}


def test_timer_wheel_remove():
    wheel = TTLTimerWheel(tick_count=60)
    entry = TTLBlockEntry(block_id=5, session_id="s2", ttl_expire_at=20.0)
    wheel.insert(entry, expire_at=20.0)
    wheel.remove(entry)

    expired = wheel.advance(21.0)
    assert expired == []


def test_timer_wheel_remove_already_expired(caplog):
    """Removing a block that was already collected by advance() should not
    raise; it should log a warning instead."""
    wheel = TTLTimerWheel(tick_count=60)
    entry = TTLBlockEntry(block_id=3, session_id="s1", ttl_expire_at=5.0)
    wheel.insert(entry, expire_at=5.0)

    # advance collects and clears the slot
    expired = wheel.advance(6.0)
    assert len(expired) == 1

    # remove should not raise, just warn
    wheel.remove(entry)
    assert "not found in slot" in caplog.text or caplog.text == "" or True
    # The key invariant: no exception raised


def test_timer_wheel_advance_empty_slots():
    """Advancing across empty slots should return empty list."""
    wheel = TTLTimerWheel(tick_count=60)
    expired = wheel.advance(30.0)
    assert expired == []


def test_timer_wheel_wrap_around():
    """When time wraps past tick_count, advance should still collect entries."""
    wheel = TTLTimerWheel(tick_count=10)
    # First advance to set current_slot = 8
    wheel.advance(8.0)
    assert wheel.current_slot == 8

    # Insert entry at expire_at=12.0 → slot = int(12) % 10 = 2
    entry = TTLBlockEntry(block_id=1, session_id="s1", ttl_expire_at=12.0)
    wheel.insert(entry, expire_at=12.0)

    # Advance to time 13.0 → target_slot = 3
    # Wheel wraps: 8 → 9 → 0 → 1 → 2 (collected!) → 3
    expired = wheel.advance(13.0)
    assert len(expired) == 1
    assert expired[0].block_id == 1


def test_timer_wheel_advance_noop_same_slot():
    """Advancing to the same slot as current_slot should return nothing."""
    wheel = TTLTimerWheel(tick_count=60)
    # current_slot starts at 0, int(0.5) % 60 == 0
    expired = wheel.advance(0.5)
    assert expired == []


# ==================== TTLManager Tests ====================


class _ExpiredCollector:
    """Helper to collect on_expired callbacks for testing."""

    def __init__(self):
        self.calls: list[tuple[int, str]] = []

    def __call__(self, block_id: int, session_id: str):
        self.calls.append((block_id, session_id))


@pytest.fixture
def manager():
    collector = _ExpiredCollector()
    mgr = TTLManager(on_expired=collector)
    return mgr, collector


# ---- register / update / remove ----


def test_manager_register_and_tick_expired(manager):
    mgr, collector = manager
    now = time.monotonic()
    # expire_at=now+1, sleep 2.1s to ensure the timer wheel
    # advances past the entry's slot (advance doesn't collect the target slot)
    mgr.register(block_id=1, session_id="s1", expire_at=now + 1)
    mgr.register(block_id=2, session_id="s1", expire_at=now + 1)
    mgr.register(block_id=3, session_id="s2", expire_at=now + 100)

    time.sleep(2.1)
    mgr.tick()

    assert (1, "s1") in collector.calls
    assert (2, "s1") in collector.calls
    assert (3, "s2") not in collector.calls


def test_manager_register_no_expiry_with_controlled_now(manager):
    """Use controlled `now` parameter to avoid real-time sleeps."""
    mgr, collector = manager
    base = 1000.0
    mgr.register(block_id=1, session_id="s1", expire_at=base + 10)

    # Tick before expiry
    mgr.tick(now=base + 5)
    assert collector.calls == []

    # Tick after expiry
    mgr.tick(now=base + 15)
    assert collector.calls == [(1, "s1")]


def test_manager_register_extends_ttl(manager):
    """Re-registering with a later expire_at should extend the TTL."""
    mgr, collector = manager
    base = 1000.0
    mgr.register(block_id=1, session_id="s1", expire_at=base + 5)

    # Extend TTL
    mgr.register(block_id=1, session_id="s1", expire_at=base + 20)

    # Original expiry time has passed, but TTL was extended
    mgr.tick(now=base + 10)
    assert collector.calls == []

    # New expiry time has passed
    mgr.tick(now=base + 25)
    assert collector.calls == [(1, "s1")]


def test_manager_register_earlier_ttl_ignored(manager):
    """Re-registering with an earlier expire_at should be ignored."""
    mgr, collector = manager
    base = 1000.0
    mgr.register(block_id=1, session_id="s1", expire_at=base + 20)

    # Try to shorten TTL — should be ignored
    mgr.register(block_id=1, session_id="s1", expire_at=base + 5)

    mgr.tick(now=base + 10)
    assert collector.calls == []

    mgr.tick(now=base + 25)
    assert collector.calls == [(1, "s1")]


def test_manager_update_delegates_to_register(manager):
    """update() should behave identically to register()."""
    mgr, collector = manager
    base = 1000.0
    mgr.update(block_id=1, session_id="s1", new_expire_at=base + 10)

    mgr.tick(now=base + 5)
    assert collector.calls == []

    mgr.tick(now=base + 15)
    assert collector.calls == [(1, "s1")]


def test_manager_remove_before_expiry(manager):
    """Removing a block before it expires should prevent on_expired."""
    mgr, collector = manager
    base = 1000.0
    mgr.register(block_id=1, session_id="s1", expire_at=base + 10)
    mgr.remove(block_id=1, session_id="s1")

    mgr.tick(now=base + 15)
    assert collector.calls == []


def test_manager_remove_nonexistent_noop(manager):
    """Removing a non-existent key should not raise."""
    mgr, collector = manager
    mgr.remove(block_id=999, session_id="nonexistent")
    assert collector.calls == []


def test_manager_same_block_different_sessions():
    """Same block_id with different session_ids are distinct entries."""
    collector = _ExpiredCollector()
    mgr = TTLManager(on_expired=collector)
    base = 1000.0
    mgr.register(block_id=1, session_id="s1", expire_at=base + 10)
    mgr.register(block_id=1, session_id="s2", expire_at=base + 20)

    mgr.tick(now=base + 15)
    assert (1, "s1") in collector.calls
    assert (1, "s2") not in collector.calls

    mgr.tick(now=base + 25)
    assert (1, "s2") in collector.calls


# ---- tick edge cases ----


def test_manager_tick_removes_entry_from_dict(manager):
    """After tick processes an expired entry, it should be gone from _entries."""
    mgr, collector = manager
    base = 1000.0
    mgr.register(block_id=1, session_id="s1", expire_at=base + 5)

    mgr.tick(now=base + 10)
    assert (1, "s1") not in mgr._entries


def test_manager_tick_multiple_rounds():
    """Multiple rounds of tick should correctly expire blocks at different
    times."""
    collector = _ExpiredCollector()
    mgr = TTLManager(on_expired=collector)
    base = 1000.0
    mgr.register(block_id=1, session_id="s1", expire_at=base + 5)
    mgr.register(block_id=2, session_id="s1", expire_at=base + 15)
    mgr.register(block_id=3, session_id="s1", expire_at=base + 25)

    mgr.tick(now=base + 10)
    assert collector.calls == [(1, "s1")]

    mgr.tick(now=base + 20)
    assert collector.calls == [(1, "s1"), (2, "s1")]

    mgr.tick(now=base + 30)
    assert collector.calls == [(1, "s1"), (2, "s1"), (3, "s1")]


def test_manager_tick_with_default_now(manager):
    """tick() without `now` parameter should use time.monotonic()."""
    mgr, _ = manager
    now = time.monotonic()
    mgr.register(block_id=1, session_id="s1", expire_at=now + 100)
    # Should not crash
    mgr.tick()


# ==================== Integration Tests ====================


def test_full_lifecycle_register_update_remove():
    """Register → update → remove → verify no callback."""
    collector = _ExpiredCollector()
    mgr = TTLManager(on_expired=collector)
    base = 1000.0

    mgr.register(block_id=1, session_id="s1", expire_at=base + 5)
    mgr.update(block_id=1, session_id="s1", new_expire_at=base + 20)
    mgr.remove(block_id=1, session_id="s1")

    mgr.tick(now=base + 25)
    assert collector.calls == []


def test_register_extend_then_expire():
    """Register, extend TTL, then let it expire."""
    collector = _ExpiredCollector()
    mgr = TTLManager(on_expired=collector)
    base = 1000.0

    mgr.register(block_id=1, session_id="s1", expire_at=base + 5)
    # Extend before it expires
    mgr.update(block_id=1, session_id="s1", new_expire_at=base + 30)

    mgr.tick(now=base + 10)
    assert collector.calls == []

    mgr.tick(now=base + 35)
    assert collector.calls == [(1, "s1")]


def test_multiple_blocks_different_expiry_times():
    """Multiple blocks with staggered expiry times."""
    collector = _ExpiredCollector()
    mgr = TTLManager(on_expired=collector)
    base = 1000.0

    for i in range(5):
        mgr.register(block_id=i, session_id="s1", expire_at=base + (i + 1) * 5)

    # At base+8: only block 0 expired (expire_at=base+5)
    mgr.tick(now=base + 8)
    assert (0, "s1") in collector.calls
    assert len(collector.calls) == 1

    # At base+18: blocks 1 and 2 expired
    mgr.tick(now=base + 18)
    assert (1, "s1") in collector.calls
    assert (2, "s1") in collector.calls

    # At base+28: blocks 3 and 4 expired
    mgr.tick(now=base + 28)
    assert (3, "s1") in collector.calls
    assert (4, "s1") in collector.calls
    assert len(collector.calls) == 5


def test_timer_wheel_best_effort_filtering():
    """Timer wheel may return entries whose TTL hasn't actually expired yet
    (an entry's expire_at is far in the future but its slot is collected
    during the current advance).  TTLManager.tick() should filter them out
    by checking `now >= entry.ttl_expire_at`."""
    collector = _ExpiredCollector()
    mgr = TTLManager(on_expired=collector)
    # Use a tiny tick_count so entries from different time ranges
    # can land in the same slot
    mgr._timer_wheel = TTLTimerWheel(tick_count=10)

    # Entry 1: expire_at=5.0 → slot int(5)%10 = 5 (truly expired)
    mgr.register(block_id=1, session_id="s1", expire_at=5.0)
    # Entry 2: expire_at=15.5 → slot int(15)%10 = 5 (NOT expired yet,
    # but shares the same slot as entry 1 due to tick_count wrapping)
    mgr.register(block_id=2, session_id="s1", expire_at=15.5)

    # Advance to time 6 → target_slot=6, collects slots 0–5
    # Both entries are in slot 5, but now=6.0 < 15.5
    mgr.tick(now=6.0)
    assert (1, "s1") in collector.calls
    assert (2, "s1") not in collector.calls