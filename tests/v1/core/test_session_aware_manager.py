# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import time
import types
from dataclasses import dataclass

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


# TODO: Remove these shims after agent_hint request models and SPM land.
import vllm.entrypoints.openai.chat_completion.protocol as chat_protocol

if not hasattr(chat_protocol, "CacheControlParams"):

    @dataclass
    class CacheControlParams:
        ttl: float = 300.0
        block_offset: int | None = None

    chat_protocol.CacheControlParams = CacheControlParams

if "vllm.v1.core.session_aware_pooling_manager" not in sys.modules:
    spm_module = types.ModuleType("vllm.v1.core.session_aware_pooling_manager")

    class SessionEventListener:
        pass

    spm_module.SessionEventListener = SessionEventListener
    sys.modules["vllm.v1.core.session_aware_pooling_manager"] = spm_module

from vllm.v1.core import session_aware_manager as sam_module
from vllm.v1.core.session_aware_manager import EphemeralRange, SessionAwareManager


class FakeTTLManager:

    def __init__(self, on_expired):
        self.on_expired = on_expired
        self.registered: dict[tuple[int, str], float] = {}

    def register(self, block_id: int, session_id: str, expire_at: float) -> None:
        self.registered[(block_id, session_id)] = expire_at

    def update(self, block_id: int, session_id: str, expire_at: float) -> None:
        self.register(block_id, session_id, expire_at)


class FakeKVCacheManager:

    def __init__(self):
        self.calls: list[tuple[int, int, float | None]] = []

    def update_block_meta(
        self,
        block_id: int,
        delta_ref: int = 0,
        ttl_expire_at: float | None = None,
    ) -> None:
        self.calls.append((block_id, delta_ref, ttl_expire_at))


@pytest.fixture(autouse=True)
def patch_sam_dependencies(monkeypatch):
    monkeypatch.setattr(sam_module, "TTLManager", FakeTTLManager)


def test_get_new_blocks_resets_session_metadata_when_recycling_block():
    pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=False,
        hash_block_size=16,
        enable_kv_cache_events=False,
    )

    block = pool.get_new_blocks(1)[0]
    block._session_ref_cnt = 3
    block._ttl_expire_at = 0
    pool.free_blocks([block])

    reused = pool.get_new_blocks(1)[0]

    assert reused is block
    assert reused.ref_cnt == 1
    assert reused._session_ref_cnt == 0
    assert reused._ttl_expire_at == 0.0


def test_free_queue_prefers_zone_a_before_zone_b():
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(2)])
    zone_b_block = queue.popleft()
    zone_a_block = queue.popleft()

    zone_b_block._session_ref_cnt = 1
    queue.append(zone_b_block)
    queue.append(zone_a_block)

    assert queue.popleft() is zone_a_block
    assert queue.popleft() is zone_b_block


def test_free_queue_keeps_live_ephemeral_blocks_unallocatable():
    queue = FreeKVCacheBlockQueue([KVCacheBlock(i) for i in range(2)])
    block = queue.popleft()
    block._ttl_expire_at = time.monotonic() + 60
    queue.append(block)

    assert queue.popleft() is not block

    with pytest.raises(ValueError, match="No allocatable free blocks"):
        queue.popleft()


def test_free_queue_lazily_promotes_expired_ephemeral_block():
    queue = FreeKVCacheBlockQueue([KVCacheBlock(0)])
    block = queue.popleft()
    block._ttl_expire_at = time.monotonic() + 0.1
    queue.append(block)
    time.sleep(1)
    block = queue.popleft_n(1)[0]

    assert block._ttl_expire_at == 0.0


# def test_sam_allocates_ephemeral_range_from_block_offset():
#     kv_cache_manager = FakeKVCacheManager()
#     sam = SessionAwareManager(kv_cache_manager)

#     sam.on_blocks_allocated(
#         session_id="session-1",
#         parent_session_id=None,
#         block_ids=[10, 11, 12],
#         ephemeral_range=EphemeralRange(block_offset=1, ttl=30.0),
#     )

#     records = sam._session_blocks["session-1"]
#     assert records[10].is_ephemeral is False
#     assert records[11].is_ephemeral is True
#     assert records[12].is_ephemeral is True

#     assert kv_cache_manager.calls[0] == (10, 1, None)
#     assert kv_cache_manager.calls[1][0] == 11
#     assert kv_cache_manager.calls[1][1] == 1
#     assert kv_cache_manager.calls[1][2] is not None
#     assert kv_cache_manager.calls[2][0] == 12
#     assert kv_cache_manager.calls[2][1] == 1
#     assert kv_cache_manager.calls[2][2] is not None


# def test_sam_cache_hit_is_idempotent_for_same_session_and_block():
#     kv_cache_manager = FakeKVCacheManager()
#     sam = SessionAwareManager(kv_cache_manager)

#     sam.on_block_cache_hit("session-1", 7)
#     sam.on_block_cache_hit("session-1", 7)

#     assert kv_cache_manager.calls == [(7, 1, None)]
#     assert list(sam._session_blocks["session-1"]) == [7]


# def test_sam_free_session_decrements_each_registered_block_once():
#     kv_cache_manager = FakeKVCacheManager()
#     sam = SessionAwareManager(kv_cache_manager)
#     sam.on_block_cache_hit("session-1", 7)
#     sam.on_block_cache_hit("session-1", 8)
#     kv_cache_manager.calls.clear()

#     result = sam.free_session("session-1")

#     assert result["freed_blocks"] == 2
#     assert "session-1" not in sam._session_blocks
#     assert kv_cache_manager.calls == [(7, -1, None), (8, -1, None)]
