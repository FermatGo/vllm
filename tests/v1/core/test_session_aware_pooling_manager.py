# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
import threading

import pytest

from vllm.v1.core.session_aware_pooling_manager import (
    SPMConfig,
    SessionKeyTracker,
)

pytestmark = pytest.mark.cpu_test


# ------------------ Mock Classes ------------------ #


class MockBackend:
    def __init__(self):
        self.set_device_called = False
        self.exists_called_with = []
        self.exists_return = True

    def set_device(self):
        self.set_device_called = True

    def exists(self, keys):
        self.exists_called_with = keys
        return self.exists_return


class MockConnectorWorker:
    def __init__(self):
        self.m_store = MockBackend()


class MockKVConnector:
    def __init__(self):
        self.connector_worker = MockConnectorWorker()


class MockFreeBlockQueue:
    def __init__(self, num_free_blocks=100):
        self.num_free_blocks = num_free_blocks


class MockBlockPool:
    def __init__(self):
        self.free_block_queue = MockFreeBlockQueue()


class MockKVCacheManager:
    def __init__(self):
        self.block_size = 16
        self.block_pool = MockBlockPool()


class MockSessionAwareManager:
    def __init__(self):
        self._sessions = {}
        self.listeners = []
        self.kv_cache_manager = MockKVCacheManager()

    def add_event_listener(self, listener):
        self.listeners.append(listener)


class MockScheduler:
    def __init__(self):
        self.submitted = []


# ------------------ TestSPMConfig ------------------ #


class TestSPMConfig:

    def test_default_values(self):
        config = SPMConfig()
        assert config.enable_keep_alive is False
        assert config.keep_alive_interval == 60
        assert config.max_keys_per_cycle == 1024
        assert config.enable_eviction is True
        assert config.eviction_grace_period == 30.0
        assert config.enable_prefetch is False
        assert config.prefetch_max_queue_size == 16
        assert config.prefetch_block_reserve == 8


# ------------------ TestSessionKeyTracker ------------------ #


class TestSessionKeyTracker:

    def test_add_keys_new_session(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1", "k2"], ["h1", "h2"])
        assert tracker.get_session_keys("s1") == ["k1", "k2"]
        assert tracker.get_session_block_hashes("s1") == ["h1", "h2"]

    def test_add_keys_duplicate_key(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1"], ["h1"])
        tracker.add_keys("s1", ["k1"], ["h1_new"])
        assert tracker.get_session_keys("s1") == ["k1"]
        assert tracker.get_session_block_hashes("s1") == ["h1_new"]

    def test_add_keys_shared_key(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1"], ["h1"])
        tracker.add_keys("s2", ["k1"], ["h1"])
        assert tracker.get_key_sessions("k1") == {"s1", "s2"}

    def test_remove_session_orphaned_keys(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1", "k2"], ["h1", "h2"])
        orphaned = tracker.remove_session("s1")
        assert sorted(orphaned) == ["k1", "k2"]
        assert tracker.get_session_keys("s1") == []
        assert tracker.get_key_sessions("k1") == set()

    def test_remove_session_no_orphan(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1"], ["h1"])
        tracker.add_keys("s2", ["k1"], ["h1"])
        orphaned = tracker.remove_session("s1")
        assert orphaned == []
        assert tracker.get_key_sessions("k1") == {"s2"}

    def test_remove_session_nonexistent(self):
        tracker = SessionKeyTracker()
        orphaned = tracker.remove_session("nonexistent")
        assert orphaned == []

    def test_remove_keys_partial(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1", "k2", "k3"], ["h1", "h2", "h3"])
        orphaned = tracker.remove_keys("s1", ["k1", "k3"])
        assert sorted(orphaned) == ["k1", "k3"]
        assert tracker.get_session_keys("s1") == ["k2"]

    def test_remove_keys_shared(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1"], ["h1"])
        tracker.add_keys("s2", ["k1"], ["h1"])
        orphaned = tracker.remove_keys("s1", ["k1"])
        assert orphaned == []
        assert tracker.get_key_sessions("k1") == {"s2"}

    def test_remove_keys_nonexistent(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1"], ["h1"])
        orphaned = tracker.remove_keys("s1", ["k_none"])
        assert orphaned == []

    def test_get_active_keys_all(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1"], ["h1"])
        tracker.add_keys("s2", ["k2"], ["h2"])
        active = tracker.get_active_keys()
        assert sorted(active) == ["k1", "k2"]

    def test_get_active_keys_empty(self):
        tracker = SessionKeyTracker()
        assert tracker.get_active_keys() == []

    def test_get_active_keys_filtered(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1"], ["h1"])
        tracker.add_keys("s2", ["k2"], ["h2"])
        tracker.add_keys("s3", ["k3"], ["h3"])
        active = tracker.get_active_keys(session_ids=["s1", "s3"])
        assert sorted(active) == ["k1", "k3"]

    def test_get_active_keys_max_limit(self):
        tracker = SessionKeyTracker()
        for i in range(5):
            tracker.add_keys("s{0}".format(i), ["k{0}".format(i)], ["h{0}".format(i)])
        active = tracker.get_active_keys(max_keys=3)
        assert len(active) == 3

    def test_get_session_block_hashes(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1", "k2"], ["h1", "h2"])
        assert tracker.get_session_block_hashes("s1") == ["h1", "h2"]

    def test_get_session_block_hashes_empty(self):
        tracker = SessionKeyTracker()
        assert tracker.get_session_block_hashes("nonexistent") == []

    def test_get_key_sessions(self):
        tracker = SessionKeyTracker()
        tracker.add_keys("s1", ["k1"], ["h1"])
        tracker.add_keys("s2", ["k1"], ["h1"])
        assert tracker.get_key_sessions("k1") == {"s1", "s2"}

    def test_get_key_sessions_nonexistent(self):
        tracker = SessionKeyTracker()
        assert tracker.get_key_sessions("nonexistent") == set()

    def test_concurrent_add_keys(self):
        tracker = SessionKeyTracker()
        errors = []

        def worker(sid, keys, hashes):
            try:
                tracker.add_keys(sid, keys, hashes)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker,
                args=("s{0}".format(i), ["k{0}".format(i)], ["h{0}".format(i)]))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(tracker.get_active_keys()) == 20

    def test_concurrent_remove_session(self):
        tracker = SessionKeyTracker()
        for i in range(10):
            tracker.add_keys("s{0}".format(i), ["k{0}".format(i)], ["h{0}".format(i)])
        errors = []

        def worker(sid):
            try:
                tracker.remove_session(sid)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("s{0}".format(i),))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert tracker.get_active_keys() == []
