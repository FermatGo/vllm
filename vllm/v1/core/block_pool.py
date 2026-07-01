# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    TTLTimerWheel,
    generate_block_hash_extra_keys,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(
        self, 
        key: BlockHashWithGroupId, 
        session_id: str | None = None,
    ) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                if session_id is not None:
                    # 优先返回同session的block
                    for block in blocks.values():
                        if session_id in block.session_ref:
                            return block
                # 没有session匹配或没有session_id，返回第一个可用block
                # 优先返回session_ref为空的block（减少session关联污染）
                for block in blocks.values():
                    if not block.session_ref:
                        return block
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        self.ttl_timer_wheel = TTLTimerWheel()

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

        # session_id -> set[block_id], Used to quickly find all the blocks of a session
        self.session_to_blocks: dict[str, set[int]] = {}

        self._session_parent: dict[str, str | None] = {}
        self._session_children: dict[str, set[str]] = {}

    def get_cached_block(
        self, 
        block_hash: BlockHash, 
        kv_cache_group_ids: list[int],
        session_id: str | None = None,
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.
            session_id: The id of the session to which the blocks belong.
        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id, session_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null blocks when enabling sparse attention like
            # sliding window attention, or Mamba models with prefix-caching in
            # align mode. We skip null blocks here.
            if blk.is_null:
                continue
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            # Calculate token range for the blocks being cached
            start_token_idx = num_cached_blocks * block_size
            end_token_idx = num_full_blocks * block_size

            # Generate extra keys for each block individually.
            # Each block may have different extra_keys (e.g., different MM
            # features, or cache_salt only for the first block).
            # Skip null blocks to match the length of new_hashes.
            extra_keys_list: list[tuple[Any, ...] | None] = []
            curr_mm_idx = 0
            for i in range(num_cached_blocks, num_full_blocks):
                if blocks[i].is_null:
                    continue
                block_start = i * block_size
                block_end = block_start + block_size
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, block_start, block_end, curr_mm_idx
                )
                extra_keys_list.append(extra_keys)

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[start_token_idx:end_token_idx],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                    lora_name=request.lora_request.name
                    if request.lora_request
                    else None,
                    extra_keys=extra_keys_list if extra_keys_list else None,
                    group_idx=kv_cache_group_id,
                )
            )

    def get_new_blocks(
        self, 
        num_blocks: int, 
        session_id: str | None = None,
    ) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.
            session_id: The ID of the session to which the blocks belong.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                # 清理前一个session的关联和TTL
                self._clear_block_session_refs(block)
                block._ttl_expire_at = 0.0  # 清除TTL
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                # 设置新session引用
                if session_id is not None:
                    block._session_ref.add(session_id)
                    self._record_block_session(block.block_id, session_id)
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                if self.metrics_collector:
                    self.metrics_collector.on_block_allocated(block)
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        # Clean up metrics tracking first to prevent leaks
        if self.metrics_collector:
            self.metrics_collector.on_block_evicted(block)

        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                    group_idx=get_group_id(block_hash),
                )
            )
        return True

    def touch(
        self, 
        blocks: Sequence[KVCacheBlock], 
        session_id: str | None = None,
    ) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
            session_id: The ID of the session to which the blocks belong.
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if session_id is not None:
                # 记录session引用(幂等：同一session只记录一次)
                if session_id not in block._session_ref:
                    block._session_ref.add(session_id)
                    self._record_block_session(block.block_id, session_id)
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(
        self, 
        ordered_blocks: Iterable[KVCacheBlock], 
        ttl: float | None = None,
    ) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
            ttl: The time-to-live for the blocks being freed.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        now = time.monotonic() if ttl else 0
        for block in blocks_list:
            block.ref_cnt -= 1
        
        free_blocks = []
        for block in blocks_list:
            if block.ref_cnt == 0 and not block.is_null:
                # 设置 TTL 过期时间
                if ttl and ttl > 0:
                    new_expire = now + ttl
                    # TTL 刷新：以最晚的为准
                    block._ttl_expire_at = max(block._ttl_expire_at, new_expire)
                    self.ttl_timer_wheel.insert(block, block._ttl_expire_at)
                free_blocks.append(block)

        self.free_block_queue.append_n(free_blocks)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        only evicts blocks that are currently cached (have a hash). blocks
        with ref_cnt > 0 are not freed from the block pool, only evicted
        from the prefix cache hash table.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            block = self.blocks[block_id]
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events

    def register_session(
        self,
        session_id: str,
        parent_session_id: str | None = None,
    ) -> None:
        """注册session及其层级关系(幂等操作)"""
        if session_id in self._session_parent:
            return
        self._session_parent[session_id] = parent_session_id
        if parent_session_id is not None:
            if parent_session_id not in self._session_children:
                self._session_children[parent_session_id] = set()
            self._session_children[parent_session_id].add(session_id)
    
    def _record_block_session(self, block_id: int, session_id: str):
        """记录block属于哪个session(幂等操作)"""
        if session_id not in self.session_to_blocks:
            self.session_to_blocks[session_id] = set()
        self.session_to_blocks[session_id].add(block_id)

    def _remove_block_session(self, block_id: int, session_id: str):
        """移除block的session记录"""
        if session_id in self.session_to_blocks:
            self.session_to_blocks[session_id].discard(block_id)
            if not self.session_to_blocks[session_id]:
                del self.session_to_blocks[session_id]
    
    def get_session_blocks(self, session_id: str) -> set[int]:
        """Obtain all the block IDs of the session"""
        blocks = self.session_to_blocks.get(session_id)
        return blocks.copy() if blocks is not None else set()
    
    def _clear_block_session_refs(self, block: KVCacheBlock) -> None:
        """清理block的session引用"""
        for session_id in list(block._session_ref):
            self._remove_block_session(block.block_id, session_id)
        block._session_ref.clear()
    
    def free_session(self, session_id: str) -> dict:
        """按 session 清理 KV cache block。

        清理策略：
        1. 遍历该session的所有block (通过session_to_blocks映射)
        2. 对每个block, 移除该session的session_ref
        3. 如果block.ref_cnt == 0(在free queue中):
        - 清除session_ref后, 如果session_ref为空且TTL已过:
            将block从B区移到A区, 使其优先被重新分配
        - 如果session_ref非空: block仍在B区, 其他session仍可prefix cache命中
        - 如果TTL未过: block仍在C区, 只清除session_ref, 等TTL过后再移动
        4. 如果block.ref_cnt > 0(被活跃request使用)
        - 只清除session_ref, 不影响block的使用

        Returns:
            dict: 清理统计信息
        """
        block_ids = self.get_session_blocks(session_id)
        freed_blocks = 0
        orphaned_blocks = 0

        for block_id in block_ids:
            block = self.blocks[block_id]
            if session_id in block._session_ref:
                block._session_ref.discard(session_id)
                self._remove_block_session(block_id, session_id)

                if block.ref_cnt == 0 and not block.is_null:
                    if not block._session_ref and not block.is_ttl_protected:
                        # session_ref全空且TTL已过，移到A区
                        self.free_block_queue.promote_to_zone_a(block)
                        orphaned_blocks += 1
                    freed_blocks += 1

        # 清理session层级树
        parent = self._session_parent.pop(session_id, None)
        if parent and parent in self._session_children:
            self._session_children[parent].discard(session_id)
        self._session_children.pop(session_id, None)

        return {
            "session_id": session_id,
            "freed_blocks": freed_blocks,
            "orphaned_blocks": orphaned_blocks,
        }
    
    def free_session_tree(self, session_id: str) -> dict:
        """递归清理session及其所有子session"""
        total_result = {"session_id": session_id, "sessions": []}
        for child_sid in list(self._session_children.get(session_id, set())):
            child_result = self.free_session_tree(child_sid)
            total_result["sessions"].append(child_result)
        result = self.free_session(session_id)
        total_result["freed_blocks"] = result["freed_blocks"]
        total_result["orphaned_blocks"] = result["orphaned_blocks"]
        return total_result
    
    def advance_ttl_timer(self) -> None:
        """Promote expired TTL-protected free blocks from zone C to A/B."""
        now = time.monotonic()

        for block in self.ttl_timer_wheel.advance(now):
            # The block may have been allocated after it was inserted into the
            # timer wheel, or it may be the null block.
            if block.is_null or block.ref_cnt != 0:
                continue

            # Only blocks currently in the free queue can be promoted.
            if block.prev_free_block is None or block.next_free_block is None:
                continue

            # Timer wheel entries are candidates. Re-check the real expire time.
            if block._ttl_expire_at <= 0:
                continue

            if now < block._ttl_expire_at:
                # With a coarse wheel or refreshed TTL, this entry is not actually
                # expired yet. Put it back.
                self.ttl_timer_wheel.insert(block, block._ttl_expire_at)
                continue

            block._ttl_expire_at = 0.0

            if block.num_session_refs > 0:
                self.free_block_queue.promote_to_zone_b(block)
            else:
                self.free_block_queue.promote_to_zone_a(block)