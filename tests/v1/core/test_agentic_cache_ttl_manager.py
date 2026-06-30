# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest

from vllm.v1.core.agentic_cache_ttl_manager import (
    AgenticCacheTTLManager,
    BlockOpType,
    CacheModifiedInfo,
)

pytestmark = pytest.mark.cpu_test


# ------------------ Mock Classes ------------------ #


class BlockZone(str):
    A = "a"
    B = "b"
    C = "c"


class MockBlock:
    def __init__(
        self,
        block_id: int,
        zone: str = BlockZone.C,
        session_id: str = "",
        ttl: float | None = None,
        ref_cnt: int = 0,
        is_offloaded: bool = False,
    ):
        self.block_id = block_id
        self.zone = zone
        self.session_id = session_id
        self.ttl = ttl
        self.ref_cnt = ref_cnt
        self.is_offloaded = is_offloaded

    @property
    def has_session(self) -> bool:
        return self.session_id != ""


class MockBlockPool:
    def __init__(self, num_blocks: int):
        self.blocks: dict[int, MockBlock] = {
            i: MockBlock(block_id=i) for i in range(num_blocks)
        }

    # ---- query callbacks ----

    def iter_blocks(self):
        return iter(self.blocks.values())

    def get_block_by_id(self, block_id: int):
        return self.blocks.get(block_id)

    def filter_by_ttl_expired(self) -> list[int]:
        now = time.monotonic()
        return [
            b.block_id for b in self.blocks.values()
            if b.zone == BlockZone.C
            and b.ttl is not None
            and b.ttl <= now
        ]

    def filter_by_no_session(self) -> list[int]:
        return [
            b.block_id for b in self.blocks.values()
            if b.zone == BlockZone.B
            and not b.has_session
        ]

    # ---- demote callbacks ----

    def demote_ttl_expired(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self.blocks[bid].zone = BlockZone.B

    def demote_no_session(self, block_ids: list[int]) -> None:
        for bid in block_ids:
            self.blocks[bid].zone = BlockZone.A

    # ---- modify callbacks ----

    def _get_session_blocks(self, session_id: str) -> list[MockBlock]:
        return sorted(
            [b for b in self.blocks.values() if b.session_id == session_id],
            key=lambda b: b.block_id,
        )

    def apply_modification(self, info: CacheModifiedInfo):
        session_blocks = self._get_session_blocks(info.session_id)
        if not session_blocks:
            return
        start = info.local_start_block_id
        stop = (info.local_stop_block_id
                if info.local_stop_block_id is not None
                else len(session_blocks))
        for local_idx in range(start, min(stop, len(session_blocks))):
            block = session_blocks[local_idx]
            if info.updated_ttl is not None:
                block.ttl = info.updated_ttl
            if info.cache_action == BlockOpType.OFFLOAD:
                block.is_offloaded = True
            elif info.cache_action == BlockOpType.PREFETCH:
                block.is_offloaded = False

    # ---- OOM prevention callback (optional) ----

    def demote_to_prevent_oom(self) -> list[int]:
        """Select C-zone blocks with the smallest TTL for forced demotion."""
        c_blocks = sorted(
            [b for b in self.blocks.values()
             if b.zone == BlockZone.C and b.ttl is not None],
            key=lambda b: b.ttl,
        )
        # Select up to 2 blocks with smallest TTL
        return [b.block_id for b in c_blocks[:2]]


# ------------------ Fixtures ------------------ #


@pytest.fixture
def pool_and_manager():
    pool = MockBlockPool(num_blocks=10)
    manager = AgenticCacheTTLManager(
        iter_blocks=pool.iter_blocks,
        get_block_by_id=pool.get_block_by_id,
        filter_by_ttl_expired=pool.filter_by_ttl_expired,
        filter_by_no_session=pool.filter_by_no_session,
        demote_ttl_expired=pool.demote_ttl_expired,
        demote_no_session=pool.demote_no_session,
        apply_modification=pool.apply_modification,
    )
    return pool, manager


# ------------------ Registry Tests ------------------ #


def test_missing_required_callbacks_raises():
    with pytest.raises(ValueError, match="Missing required callbacks"):
        AgenticCacheTTLManager(callbacks={"iter_blocks": lambda: []})


def test_register_and_get_callback(pool_and_manager):
    _, manager = pool_and_manager
    manager.register("custom_fn", lambda x: x * 2)
    assert manager.get_callback("custom_fn")(3) == 6


def test_unregister_optional_callback(pool_and_manager):
    _, manager = pool_and_manager
    manager.register("temp_fn", lambda: None)
    manager.unregister("temp_fn")
    with pytest.raises(KeyError, match="temp_fn"):
        manager.get_callback("temp_fn")


def test_unregister_required_callback_raises(pool_and_manager):
    _, manager = pool_and_manager
    with pytest.raises(ValueError, match="Cannot unregister"):
        manager.unregister("iter_blocks")


def test_get_nonexistent_callback_raises(pool_and_manager):
    _, manager = pool_and_manager
    with pytest.raises(KeyError, match="not registered"):
        manager.get_callback("nonexistent")


def test_init_with_dict_callbacks():
    pool = MockBlockPool(5)
    manager = AgenticCacheTTLManager(callbacks={
        "iter_blocks": pool.iter_blocks,
        "get_block_by_id": pool.get_block_by_id,
        "filter_by_ttl_expired": pool.filter_by_ttl_expired,
        "filter_by_no_session": pool.filter_by_no_session,
        "demote_ttl_expired": pool.demote_ttl_expired,
        "demote_no_session": pool.demote_no_session,
        "apply_modification": pool.apply_modification,
    })
    assert len(manager.get_all_cache_info()) == 5


# ------------------ Query Tests ------------------ #


def test_get_all_cache_info(pool_and_manager):
    pool, manager = pool_and_manager
    assert len(manager.get_all_cache_info()) == 10


def test_get_cache_info_by_id_found(pool_and_manager):
    _, manager = pool_and_manager
    b = manager.get_cache_info_by_id(2)
    assert b.block_id == 2


def test_get_cache_info_by_id_not_found(pool_and_manager):
    _, manager = pool_and_manager
    assert manager.get_cache_info_by_id(999) is None


# ------------------ check_cache_ttl Tests ------------------ #


def test_nothing_expired_initially(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    pool.blocks[0].session_id, pool.blocks[0].ttl = "s1", now + 5
    pool.blocks[1].session_id, pool.blocks[1].ttl = "s1", now + 5
    expired, no_session = manager.check_cache_ttl()
    assert expired == []
    assert no_session == []


def test_ttl_expired_after_sleep(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    pool.blocks[0].session_id, pool.blocks[0].ttl = "s1", now + 1
    pool.blocks[1].session_id, pool.blocks[1].ttl = "s1", now + 1
    pool.blocks[2].session_id, pool.blocks[2].ttl = "s2", now + 1
    pool.blocks[3].session_id, pool.blocks[3].ttl = "s2", now + 5

    time.sleep(1.1)
    expired, _ = manager.check_cache_ttl()

    assert sorted(expired) == [0, 1, 2]
    for bid in [0, 1, 2]:
        assert pool.blocks[bid].zone == BlockZone.B
    assert pool.blocks[3].zone == BlockZone.C


def test_demote_no_session_after_release(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    pool.blocks[0].session_id, pool.blocks[0].ttl = "s1", now + 1
    pool.blocks[1].session_id, pool.blocks[1].ttl = "s1", now + 1

    time.sleep(1.1)
    manager.check_cache_ttl()

    pool.blocks[0].session_id = ""
    pool.blocks[1].session_id = ""
    _, no_session = manager.check_cache_ttl()

    assert sorted(no_session) == [0, 1]
    assert pool.blocks[0].zone == BlockZone.A
    assert pool.blocks[1].zone == BlockZone.A


def test_partial_session_release(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    for i in range(4):
        pool.blocks[i].session_id, pool.blocks[i].ttl = "s1", now + 1

    time.sleep(1.1)
    manager.check_cache_ttl()

    pool.blocks[1].session_id = ""
    pool.blocks[2].session_id = ""
    _, no_session = manager.check_cache_ttl()

    assert sorted(no_session) == [1, 2]
    assert pool.blocks[0].zone == BlockZone.B
    assert pool.blocks[1].zone == BlockZone.A
    assert pool.blocks[2].zone == BlockZone.A
    assert pool.blocks[3].zone == BlockZone.B


def test_none_ttl_never_expires(pool_and_manager):
    pool, manager = pool_and_manager
    pool.blocks[0].session_id = "s1"
    pool.blocks[0].ttl = None

    expired, _ = manager.check_cache_ttl()
    assert 0 not in expired
    assert pool.blocks[0].zone == BlockZone.C


def test_two_rounds_of_check(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    pool.blocks[0].session_id, pool.blocks[0].ttl = "s1", now + 1

    time.sleep(1.1)
    expired1, _ = manager.check_cache_ttl()
    assert expired1 == [0]
    assert pool.blocks[0].zone == BlockZone.B

    pool.blocks[0].session_id = ""
    _, no_session2 = manager.check_cache_ttl()
    assert no_session2 == [0]
    assert pool.blocks[0].zone == BlockZone.A


# ------------------ modify_cache_info Tests ------------------ #


def test_modify_single_block_local_index(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    pool.blocks[0].session_id, pool.blocks[0].ttl = "s1", now + 10
    pool.blocks[1].session_id, pool.blocks[1].ttl = "s1", now + 10
    pool.blocks[2].session_id, pool.blocks[2].ttl = "s1", now + 10

    info = CacheModifiedInfo(
        session_id="s1",
        local_start_block_id=0,
        local_stop_block_id=1,
        updated_ttl=now,
    )
    manager.modify_cache_info(info)

    assert pool.blocks[0].ttl <= time.monotonic()
    assert pool.blocks[1].ttl > time.monotonic()
    assert pool.blocks[2].ttl > time.monotonic()


def test_modify_range_local_index(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    for i in range(4):
        pool.blocks[i].session_id, pool.blocks[i].ttl = "s1", now + 10

    info = CacheModifiedInfo(
        session_id="s1",
        local_start_block_id=1,
        local_stop_block_id=3,
        updated_ttl=now,
    )
    manager.modify_cache_info(info)

    assert pool.blocks[0].ttl > time.monotonic()
    assert pool.blocks[1].ttl <= time.monotonic()
    assert pool.blocks[2].ttl <= time.monotonic()
    assert pool.blocks[3].ttl > time.monotonic()


def test_modify_no_stop_modifies_all(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    pool.blocks[5].session_id, pool.blocks[5].ttl = "s1", now + 10
    pool.blocks[6].session_id, pool.blocks[6].ttl = "s1", now + 10

    info = CacheModifiedInfo(
        session_id="s1",
        local_start_block_id=0,
        updated_ttl=now,
    )
    manager.modify_cache_info(info)

    assert pool.blocks[5].ttl <= time.monotonic()
    assert pool.blocks[6].ttl <= time.monotonic()


def test_modify_offload_action(pool_and_manager):
    pool, manager = pool_and_manager
    pool.blocks[0].session_id, pool.blocks[0].ttl = "s1", 100
    pool.blocks[1].session_id, pool.blocks[1].ttl = "s1", 100

    info = CacheModifiedInfo(
        session_id="s1",
        local_start_block_id=0,
        cache_action=BlockOpType.OFFLOAD,
    )
    manager.modify_cache_info(info)

    assert pool.blocks[0].is_offloaded is True
    assert pool.blocks[1].is_offloaded is True


def test_modify_prefetch_action(pool_and_manager):
    pool, manager = pool_and_manager
    pool.blocks[0].session_id, pool.blocks[0].is_offloaded = "s1", True
    pool.blocks[1].session_id, pool.blocks[1].is_offloaded = "s1", True

    info = CacheModifiedInfo(
        session_id="s1",
        local_start_block_id=0,
        cache_action=BlockOpType.PREFETCH,
    )
    manager.modify_cache_info(info)

    assert pool.blocks[0].is_offloaded is False
    assert pool.blocks[1].is_offloaded is False


def test_modify_nonexistent_session_noop(pool_and_manager):
    _, manager = pool_and_manager
    info = CacheModifiedInfo(
        session_id="nonexistent",
        local_start_block_id=0,
        updated_ttl=time.monotonic(),
    )
    manager.modify_cache_info(info)  # should not raise


# ------------------ prevent_oom Tests ------------------ #


def test_prevent_oom_without_callback(pool_and_manager):
    """prevent_oom returns empty list when callback is not registered."""
    _, manager = pool_and_manager
    assert manager.prevent_oom() == []


def test_prevent_oom_with_callback():
    """prevent_oom selects blocks and demotes them via demote_ttl_expired."""
    pool = MockBlockPool(num_blocks=10)
    now = time.monotonic()
    # All blocks in C zone with long TTL — nothing would expire normally
    for i in range(5):
        pool.blocks[i].session_id = f"s{i}"
        pool.blocks[i].ttl = now + 100

    manager = AgenticCacheTTLManager(
        iter_blocks=pool.iter_blocks,
        get_block_by_id=pool.get_block_by_id,
        filter_by_ttl_expired=pool.filter_by_ttl_expired,
        filter_by_no_session=pool.filter_by_no_session,
        demote_ttl_expired=pool.demote_ttl_expired,
        demote_no_session=pool.demote_no_session,
        apply_modification=pool.apply_modification,
        demote_to_prevent_oom=pool.demote_to_prevent_oom,
    )

    demoted = manager.prevent_oom()
    # Smallest-TTL blocks get demoted
    assert len(demoted) == 2
    for bid in demoted:
        assert pool.blocks[bid].zone == BlockZone.B


def test_prevent_oom_returns_empty_when_no_c_blocks():
    """If no blocks are in C zone, callback returns nothing."""
    pool = MockBlockPool(num_blocks=5)
    # All blocks in A zone
    for i in range(5):
        pool.blocks[i].zone = BlockZone.A

    manager = AgenticCacheTTLManager(
        iter_blocks=pool.iter_blocks,
        get_block_by_id=pool.get_block_by_id,
        filter_by_ttl_expired=pool.filter_by_ttl_expired,
        filter_by_no_session=pool.filter_by_no_session,
        demote_ttl_expired=pool.demote_ttl_expired,
        demote_no_session=pool.demote_no_session,
        apply_modification=pool.apply_modification,
        demote_to_prevent_oom=pool.demote_to_prevent_oom,
    )

    demoted = manager.prevent_oom()
    assert demoted == []


def test_prevent_oom_then_check_releases_session():
    """Full flow: prevent_oom → demote to B → release session → demote to A."""
    pool = MockBlockPool(num_blocks=10)
    now = time.monotonic()
    for i in range(4):
        pool.blocks[i].session_id = "s1"
        pool.blocks[i].ttl = now + 100

    manager = AgenticCacheTTLManager(
        iter_blocks=pool.iter_blocks,
        get_block_by_id=pool.get_block_by_id,
        filter_by_ttl_expired=pool.filter_by_ttl_expired,
        filter_by_no_session=pool.filter_by_no_session,
        demote_ttl_expired=pool.demote_ttl_expired,
        demote_no_session=pool.demote_no_session,
        apply_modification=pool.apply_modification,
        demote_to_prevent_oom=pool.demote_to_prevent_oom,
    )

    # Step 1: OOM prevention forces demotion of 2 blocks
    demoted = manager.prevent_oom()
    assert len(demoted) == 2
    for bid in demoted:
        assert pool.blocks[bid].zone == BlockZone.B

    # Step 2: release their sessions
    for bid in demoted:
        pool.blocks[bid].session_id = ""

    # Step 3: normal check demotes them B → A
    _, no_session = manager.check_cache_ttl()
    assert sorted(no_session) == sorted(demoted)
    for bid in demoted:
        assert pool.blocks[bid].zone == BlockZone.A


# ------------------ Integration Tests ------------------ #


def test_modify_ttl_then_demote(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    pool.blocks[0].session_id, pool.blocks[0].ttl = "s1", now + 10
    pool.blocks[1].session_id, pool.blocks[1].ttl = "s1", now + 10

    info = CacheModifiedInfo(
        session_id="s1",
        local_start_block_id=0,
        local_stop_block_id=1,
        updated_ttl=now,
    )
    manager.modify_cache_info(info)

    expired, _ = manager.check_cache_ttl()
    assert expired == [0]
    assert pool.blocks[0].zone == BlockZone.B
    assert pool.blocks[1].zone == BlockZone.C


def test_full_lifecycle(pool_and_manager):
    pool, manager = pool_and_manager
    now = time.monotonic()
    pool.blocks[0].session_id, pool.blocks[0].ttl = "s1", now + 10
    pool.blocks[1].session_id, pool.blocks[1].ttl = "s1", now + 10

    # modify → expired
    manager.modify_cache_info(CacheModifiedInfo(
        session_id="s1",
        local_start_block_id=0,
        updated_ttl=now,
    ))
    expired, _ = manager.check_cache_ttl()
    assert sorted(expired) == [0, 1]
    assert pool.blocks[0].zone == BlockZone.B
    assert pool.blocks[1].zone == BlockZone.B

    # release session → reclaimable
    pool.blocks[0].session_id = ""
    pool.blocks[1].session_id = ""
    _, no_session = manager.check_cache_ttl()
    assert sorted(no_session) == [0, 1]
    assert pool.blocks[0].zone == BlockZone.A
    assert pool.blocks[1].zone == BlockZone.A


def test_non_contiguous_session_blocks():
    """Session blocks may be non-contiguous; local index still maps correctly."""
    pool = MockBlockPool(num_blocks=10)
    manager = AgenticCacheTTLManager(
        iter_blocks=pool.iter_blocks,
        get_block_by_id=pool.get_block_by_id,
        filter_by_ttl_expired=pool.filter_by_ttl_expired,
        filter_by_no_session=pool.filter_by_no_session,
        demote_ttl_expired=pool.demote_ttl_expired,
        demote_no_session=pool.demote_no_session,
        apply_modification=pool.apply_modification,
    )
    now = time.monotonic()
    # s1 occupies block 2, 5, 8 (non-contiguous)
    pool.blocks[2].session_id, pool.blocks[2].ttl = "s1", now + 10
    pool.blocks[5].session_id, pool.blocks[5].ttl = "s1", now + 10
    pool.blocks[8].session_id, pool.blocks[8].ttl = "s1", now + 10

    # local[0]=block2, local[1]=block5, local[2]=block8
    info = CacheModifiedInfo(
        session_id="s1",
        local_start_block_id=1,
        local_stop_block_id=3,
        updated_ttl=now,
    )
    manager.modify_cache_info(info)

    assert pool.blocks[2].ttl > time.monotonic()
    assert pool.blocks[5].ttl <= time.monotonic()
    assert pool.blocks[8].ttl <= time.monotonic()

    expired, _ = manager.check_cache_ttl()
    assert sorted(expired) == [5, 8]
