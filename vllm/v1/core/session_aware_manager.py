# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations
import time
from typing import Callable, Any
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.request import Request
from vllm.entrypoints.openai.chat_completion.protocol import CacheControlParams
from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
from vllm.v1.core.session_event_listener import SessionEventListener
from vllm.v1.engine import ContextManagementEditsParams, ContextManagementParams
from vllm.v1.core.kv_cache_utils import BlockHash, get_block_hash


logger = init_logger(__name__)


@dataclass
class SessionBlockRecord:
    """单个 session 对单个 block 的引用记录（仅在 SAM 内部维护）"""
    session_id: str
    block_id: int
    is_ephemeral: bool = False          # 是否受 cache_control ephemeral 保护
    ttl_expire_at: float = 0.0         # ephemeral block 的 TTL 过期时间，0 表示无限制
    created_at: float = 0.0                  # 记录创建时间


@dataclass
class SessionInfo:
    """Session 注册信息"""
    session_id: str
    parent_session_id: str | None = None
    children: set[str] = field(default_factory=set)
    request_ids: set[str] = field(default_factory=set)  # 关联的请求 ID
    created_at: float = 0.0

@dataclass
class EphemeralRange:
    """Ephemeral 保护范围"""
    block_offset: int      # ephemeral 保护的起始 block index
    ttl: float             # TTL 时长（秒）

@dataclass
class EditResponse:
    """session controller针对单个edit返回结构体"""
    session_id: str # session id
    type: str # op类型
    op_staus: bool # 执行状态
    expected_op_block_num: int = 0 # 预期编辑block数
    actual_op_block_num: int = 0 # 实际编辑block数
    fail_reason: str = '' # （可选）执行失败原因

class SessionAwareManager:
    """Session-Aware Manager — 管理 session 生命周期与 block 映射"""

    # TODO：是否已实现分配的时候有ttl才保护
    # TODO：没空间是C区可开放，远程保持访问、保护
    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager

        # Session 注册表
        self._sessions: dict[str, SessionInfo] = {}

        # session_id → {block_id → SessionBlockRecord}
        # 反向索引：block_id → {session_id → SessionBlockRecord}（可以进一步增加包装类BlockInfo）
        self._session_blocks: dict[str, dict[int, SessionBlockRecord]] = {}
        self._block_sessions: dict[int, dict[str, SessionBlockRecord]] = {}
        self._session_block_hash: dict[str, list[BlockHash]] = {}

        # TTL 管理器（定时轮）
        self._ttl_manager = TTLManager(on_expired=self._on_ttl_expired)

        # Session 控制器
        self._session_controller = SessionController(
            execute_offload=self._execute_offload,
            execute_prefetch=self._execute_prefetch,
            execute_evict=self._execute_evict,
            get_global_block_id_by_session=self._get_session_global_block_ids,
            get_block_hashes_by_session=self._get_session_block_hash
        )

        # 新增：事件监听器列表
        self._event_listeners: list[SessionEventListener] = []

    def on_block_cache_hit_for_request(
        self,
        request: Request,
        blocks: KVCacheBlocks
    ) -> None:
        for group in blocks.get_block_ids():
            for block_id in group:
                self.on_block_cache_hit(request.session_id, block_id)

    def on_blocks_allocated_for_request(
        self,
        request: Request,
        blocks: KVCacheBlocks
    ) -> None:
        block_ids = [
            block_id
            for group in blocks.get_block_ids()
            for block_id in group
        ]
        self.on_blocks_allocated(
            session_id=request.session_id,
            parent_session_id=request.parent_session_id,
            block_ids=block_ids,
            ephemeral_range=compute_ephemeral_range(request.cache_control),
        )

    def on_blocks_allocated(
        self,
        session_id: str | None,
        parent_session_id: str,
        block_ids: list[int],
        ephemeral_range: EphemeralRange | None = None,
    ) -> None:
        """KVCacheManager 分配 block 后通知 SAM"""

        if session_id is None:
            return  # 无 session_id，不记录

        # 注册 session（如果首次出现）
        self._ensure_session_registered(session_id, parent_session_id)

        ephemeral_start = ephemeral_range.block_offset if ephemeral_range else None

        for idx, block_id in enumerate(block_ids):
            block_hash = self.kv_cache_manager.block_pool.blocks[block_id].block_hash
            if block_hash is None:
                break

            self._session_block_hash.setdefault(session_id, []).append(get_block_hash(block_hash))

            is_ephemeral = (
                ephemeral_start is not None and idx <= ephemeral_start
            )

            # 清零旧的 session 引用（SAM 内部）
            self._clear_block_session_refs(block_id)

            # 添加新的 session 引用（SAM 内部）
            ttl_expire_at = 0.0
            if is_ephemeral and ephemeral_range and ephemeral_range.ttl > 0:
                ttl_expire_at = time.monotonic() + ephemeral_range.ttl

            record = SessionBlockRecord(
                session_id=session_id,
                block_id=block_id,
                is_ephemeral=is_ephemeral,
                ttl_expire_at=ttl_expire_at,
                created_at=time.monotonic(),
            )

            # 更新 SAM 内部双向索引
            self._add_session_block_ref(record)

            # 注册 TTL（如果是 ephemeral block）
            if is_ephemeral and ttl_expire_at > 0:
                self._ttl_manager.register(block_id, session_id, ttl_expire_at)

            # 统一接口：通知 KVCacheManager 更新 block metadata
            self.kv_cache_manager.update_block_meta(
                block_id,
                delta_ref=+1,
                ttl_expire_at=ttl_expire_at if ttl_expire_at > 0 else None,
            )

    def on_block_cache_hit(
        self,
        session_id: str | None,
        block_id: int,
    ) -> None:
        """prefix cache 命中时通知 SAM"""

        if session_id is None:
            return

        # 幂等检查：如果该 session 已经引用了该 block，不重复添加
        if self._is_session_block_registered(session_id, block_id):
            # 已存在引用，刷新 TTL（如果是 ephemeral block）
            record = self._session_blocks[session_id][block_id]
            if record.is_ephemeral:
                new_expire = 0
                if record.ttl_expire_at > 0:
                    new_expire = max(record.ttl_expire_at,
                                     time.monotonic() + (record.ttl_expire_at - record.created_at))
                record.ttl_expire_at = new_expire
                self._ttl_manager.update(block_id, session_id, new_expire)
                # 刷新 block 上的 TTL
                self.kv_cache_manager.update_block_meta(
                    block_id, ttl_expire_at=new_expire
                )
        else:

            # 新增 session 引用
            record = SessionBlockRecord(
                session_id=session_id,
                block_id=block_id,
                is_ephemeral=False,      # cache hit 的 block 不新增 ephemeral 保护
                ttl_expire_at=0.0,
                created_at=time.monotonic(),
            )
            self._add_session_block_ref(record)

            # 统一接口
            self.kv_cache_manager.update_block_meta(block_id, delta_ref=+1)

        # cache hit 的 block 应当已经是完整且具有 hash 的 cached block。
        block = self.kv_cache_manager.block_pool.blocks[block_id]
        block_hash = get_block_hash(block.block_hash)
        self._session_block_hash[session_id].append(block_hash)

        if block_hash is None:
            logger.warning(
                "Cache-hit block %s has no block hash; "
                "skip session_cache_hit notification",
                block_id,
            )
            return

        self._notify_event(
            "session_cache_hit",
            session_id=session_id,
            block_id=block_id,
            block_hash=block_hash,
        )

    def _on_ttl_expired(self, block_id: int, session_id: str) -> None:
        """ephemeral block TTL 到期回调"""
        logger.info(f"working on _on_ttl_expired in SAM for block {block_id} and session id {session_id}")
        record = self._block_sessions.get(block_id, {}).get(session_id)
        if record is None or not record.is_ephemeral:
            return

        # 移除 SAM 内部记录
        self._remove_session_block_ref(session_id, block_id)

        # 统一接口：清除 TTL + 减少 session 引用
        # 这会导致 block 的 is_ephemeral() 返回 False
        # 且 _session_ref_cnt 减 1
        # FreeKVCacheBlockQueue.on_block_meta_changed 自动重评估分区
        self.kv_cache_manager.update_block_meta(
            block_id,
            delta_ref=-1,
            ttl_expire_at=0.0,
        )

        block = self.kv_cache_manager.block_pool.blocks[block_id]
        if not block.block_hash:
            return
        else:
            block_hash = get_block_hash(block.block_hash)

            # SAM 状态修改完成后再通知 SPM。
            self._notify_event(
                "session_ttl_expired",
                session_id=[session_id],
                block_ids=[block_id],
                block_hashs=[block_hash],
            )

    def _ensure_session_registered(self, session_id: str, parent_session_id: str) -> None:
        """确保 session 已注册（SAM 内部）"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionInfo(
                session_id=session_id,
                parent_session_id=parent_session_id,
                created_at=time.monotonic(),
            )
            # 如果有父 session，更新父 session 的 children
            if parent_session_id and parent_session_id in self._sessions:
                self._sessions[parent_session_id].children.add(session_id)

    def _clear_block_session_refs(self, block_id: int) -> None:
        """清除指定 block 的所有 session 引用（SAM 内部）"""
        if block_id in self._block_sessions:
            for session_id in list(self._block_sessions[block_id].keys()):
                self._remove_session_block_ref(session_id, block_id)

    def _add_session_block_ref(self, record: SessionBlockRecord) -> None:
        """添加 session 对 block 的引用（SAM 内部）"""
        self._session_blocks.setdefault(record.session_id, {})[record.block_id] = record
        self._block_sessions.setdefault(record.block_id, {})[record.session_id] = record

    def _is_session_block_registered(self, session_id: str, block_id: int) -> bool:
        """检查 session 是否已注册对 block 的引用（SAM 内部）"""
        return (
            session_id in self._session_blocks and
            block_id in self._session_blocks[session_id]
        )

    def _remove_session_block_ref(self, session_id: str, block_id: int) -> None:
        """移除 session 对 block 的引用（SAM 内部）"""
        if session_id in self._session_blocks:
            self._session_blocks[session_id].pop(block_id, None)
            if not self._session_blocks[session_id]:
                del self._session_blocks[session_id]
        if block_id in self._block_sessions:
            self._block_sessions[block_id].pop(session_id, None)
            if not self._block_sessions[block_id]:
                del self._block_sessions[block_id]

    def free_session(self, session_id: str) -> list:
        """清理指定 session 的所有 block 引用"""
        freed_blocks = 0
        orphaned_blocks = 0

        block_hashes = []

        if session_id in self._session_blocks:
            for block_id in list(self._session_blocks[session_id].keys()):
                self._remove_session_block_ref(session_id, block_id)
                freed_blocks += 1

                # 检查 block 是否已无任何 session 引用
                if block_id not in self._block_sessions:
                    orphaned_blocks += 1

                self.kv_cache_manager.update_block_meta(block_id, delta_ref=-1)

                block = self.kv_cache_manager.block_pool.blocks[block_id]
                block_hashes.append(get_block_hash(block.block_hash))
                
            if self._session_block_hash[session_id]:
                del self._session_block_hash[session_id]

        # 移除 session 注册信息
        if session_id in self._sessions:
            parent_session_id = self._sessions[session_id].parent_session_id
            if parent_session_id and parent_session_id in self._sessions:
                self._sessions[parent_session_id].children.discard(session_id)
            del self._sessions[session_id]

        return block_hashes

    def free_session_tree(self, session_id: str) -> list:
        """递归清理session及其所有子session"""
        session_info = self._sessions.get(session_id)
        if session_info is None:
            return []

        block_hashes_all = []
        for child_sid in list(self._sessions[session_id].children):
            block_hashes_all.extend(self.free_session_tree(child_sid))

        block_hashes_all.extend(self.free_session(session_id))

        logger.info(
            f"Free session tree for session {session_id}: "
            f"current session_to_blocks: {self._session_blocks}, "
            f"session_info: {self._sessions}, "
        )

        return block_hashes_all


    def _execute_offload(self, session_id: str) -> None:
        """卸载指定范围的 block — 减少 session 引用 + 清除当前session的TTL（通知TTLManager）"""
        return

    def _execute_prefetch(
            self,
            session_id: str,
            block_hashes: list[BlockHash],
        ) -> int:
        """预取: 分配新的blcok，添加session信息，加载cache（hash）"""
        """通知 SPM 创建远端预取任务。"""

        self._notify_event(
            "context_management_prefetch",
            session_id=session_id,
            block_hashes=block_hashes,
        )
        #TODO: 后续返回当前session及其子session的block hash
        return len(block_hashes)

    def _get_session_global_block_ids(self, session_id: str) -> list[int]:
        blocks_result = []
        if session_id in self._session_blocks:
            blocks_result = list(self._session_blocks[session_id].keys())
        return blocks_result


    def _get_session_block_hash(self, session_id: str) -> list[BlockHash]:
        block_hash_list = []
        block_ids = self._get_session_global_block_ids(session_id)
        if block_ids:
            for blk in block_ids:
                block = self.kv_cache_manager.block_pool.blocks[blk]
                block_hash_list.append(get_block_hash(block))
        return block_hash_list


    def _execute_evict(self, session_id: str, block_ids: list[int], is_session: bool) -> int:
        """驱逐指定范围的 block — 减少引用 + 清除当前session的TTL + 标记清除（session引用归0）
        清除本地引用，并通知 SPM 停止对应远端 PoolKey 的 Keep-Alive
        """
        res = 0
        if not is_session:
            affected_block_ids: list[int] = []
            affected_block_hashes: list[BlockHash] = []

            for block_id in block_ids:
                cur_block_session = self._block_sessions.get(block_id, {})
                record = cur_block_session.get(session_id)
                if len(cur_block_session)==0 or record is None:
                    continue

                if record.is_ephemeral:
                    self._ttl_manager.remove(block_id, session_id)

                self._remove_session_block_ref(session_id, block_id)
                
                remaining_records = self._block_sessions.get(block_id, {}).values()
                latest_ttl_expire_at = max(
                    (
                        remaining_record.ttl_expire_at
                        for remaining_record in remaining_records
                    ),
                    default=0.0,
                )
                self.kv_cache_manager.update_block_meta(
                    block_id,
                    delta_ref=-1,
                    ttl_expire_at=latest_ttl_expire_at,
                )

                # 只有当前session引用移除后，block无任何session引用时，才将其加入通知列表
                block = self.kv_cache_manager.block_pool.blocks[block_id]
                if block.num_session_refs == 0:
                    block_hash = get_block_hash(block.block_hash)
                    affected_block_hashes.append(block_hash)
                    affected_block_ids.append(block_id)

            #TODO: session引用是否清零
            if affected_block_ids:
                logger.warning(
                    f"sending param to context_management_evict session_id {session_id} block_ids {affected_block_ids} block_hashes {affected_block_hashes}")
                res = self._notify_event(
                    "context_management_evict",
                    session_id=session_id,
                    block_ids=affected_block_ids,
                    block_hashes=affected_block_hashes,
                )

                for blk_hash in affected_block_hashes:
                    if blk_hash in self._session_block_hash:
                        self._session_block_hash[session_id].remove(blk_hash, None)
                if not self._session_block_hash[session_id]:
                    del self._session_block_hash[session_id]

        else:
            block_hashes = self.free_session_tree(session_id)
            res = self._notify_event(
                "context_management_evict",
                block_hashes=block_hashes,
            )

        return res

    def add_event_listener(self, listener: SessionEventListener):
        """注册事件监听器（SPM 调用）"""
        self._event_listeners.append(listener)

    def _notify_event(self, event_type: str, **kwargs):
        """通知所有监听器"""
        for listener in tuple(self._event_listeners):
            handler = getattr(listener, f"on_{event_type}", None)
            if handler is None:
                continue

            try:
                handler(**kwargs)
            except Exception:
                logger.exception(
                    "Session event listener %r failed while handling %s",
                    listener,
                    event_type,
                )


@dataclass
class TTLBlockEntry:
    block_id: int
    session_id: str
    ttl_expire_at: float

class TTLTimerWheel:
    """Best-effort timer wheel for TTL-protected free KV cache blocks.

    This class does not mutate the free-block queue. It only tracks when a
    block may become eligible for promotion out of zone C. Callers should
    validate that returned blocks are still in the free queue, then call
    FreeKVCacheBlockQueue.promote_to_zone_a/b() as appropriate.
    """

    def __init__(self, tick_count: int = 60):
        """Initialize the timer wheel.

        Parameters
        ----------
        tick_count : int
            Number of slots in the wheel. Each slot corresponds to one
            time unit (second). A block whose TTL expires at time T is
            placed in slot ``int(T) % tick_count``. The wheel wraps
            around, so tick_count also determines the maximum TTL
            range that can be uniquely tracked.
        """
        self.slots: list[list[TTLBlockEntry]] = [[] for _ in range(tick_count)]
        self.current_slot = 0
        self.tick_count = tick_count

        logger.info(
            f"Initialized TTLTimerWheel with {tick_count} slots."
        )

    def insert(self, block: TTLBlockEntry, expire_at: float) -> None:
        """Insert a block into the wheel at the slot corresponding to
        its expiration time.

        Parameters
        ----------
        block : TTLBlockEntry
            The block entry to track.
        expire_at : float
            Absolute timestamp (e.g. time.monotonic()) at which the
            block's TTL expires. The block is placed in slot
            ``int(expire_at) % tick_count``.
        """
        tick_index = int(expire_at) % self.tick_count
        self.slots[tick_index].append(block)

        logger.info(
            f"Inserting block {block.block_id} into TTLTimerWheel, "
            f"block.session_id={block.session_id},"
            f"block.ttl_expire_at={block.ttl_expire_at:.2f}, "
        )

    def remove(self, block: TTLBlockEntry) -> None:
        """Remove a block from the wheel.

        Uses the block's ``ttl_expire_at`` to locate its slot.
        If the block has already been collected by ``advance()``
        (i.e. its TTL already expired), it won't be found in the
        wheel and a warning is logged instead of raising an error.

        Parameters
        ----------
        block : TTLBlockEntry
            The block entry to remove. Must be the same object that
            was passed to ``insert()`` (uses list.remove identity
            check).
        """
        tick_index = int(block.ttl_expire_at) % self.tick_count
        slot = self.slots[tick_index]
        try:
            slot.remove(block)
        except ValueError:
            logger.warning(
                "Block %d not found in slot %d (expire_at=%.2f), "
                "may have already expired.",
                block.block_id, tick_index, block.ttl_expire_at,
            )
            return

        logger.info(
            "Removed block %d from TTLTimerWheel slot %d, "
            "session_id=%s, expire_at=%.2f.",
            block.block_id, tick_index,
            block.session_id, block.ttl_expire_at,
        )

    def advance(self, now: float) -> list[TTLBlockEntry]:
        """Advance the wheel to the current time and return all blocks
        whose TTL has expired.

        Moves ``current_slot`` forward to ``int(now) % tick_count``,
        collecting and clearing every slot along the way. The returned
        blocks are those whose TTL deadline falls in the time range
        between the previous position and the current time.

        Important: this method only **identifies** expired blocks; it
        does not mutate the free-block queue or change block zones.
        Callers should validate that returned blocks are still in the
        free queue, then apply the appropriate zone transition.

        Parameters
        ----------
        now : float
            Current time, typically ``time.monotonic()``.

        Returns
        -------
        list[TTLBlockEntry]
            All blocks that expired between the previous wheel
            position and ``now``.
        """
        expired = []
        target_slot = int(now) % self.tick_count
        while self.current_slot != target_slot:
            expired.extend(self.slots[self.current_slot])
            self.slots[self.current_slot].clear()
            self.current_slot = (self.current_slot + 1) % self.tick_count
        if len(expired) > 0:
            logger.info(
                f"Advancing TTLTimerWheel to time {now:.2f}, "
                f"(current_slot={self.current_slot})."
                f" Expired blocks: {[block.block_id for block in expired]}."
            )

        return expired

class TTLManager:
    """Block 级 TTL 管理器.

    使用 ``(block_id, session_id)`` 作为键跟踪每个 block 的 TTL，
    内部维护一个 TTLTimerWheel 用于高效发现过期 entry。当 entry
    过期时调用初始化时传入的 ``on_expired`` 回调通知调用方。
    """

    def __init__(self, on_expired: Callable[[int, str], None]):
        """
        Parameters
        ----------
        on_expired : Callable[[int, str], None]
            Block 过期时的回调，签名为 ``on_expired(block_id, session_id)``。
            调用方在此回调里执行实际的 zone 转换 / 资源回收。
        """
        self._on_expired = on_expired
        self._entries: dict[tuple[int, str], TTLBlockEntry] = {}
        self._timer_wheel = TTLTimerWheel(tick_count=3600)

        logger.info("TTLManager initialized with on_expired=%s",
                    getattr(on_expired, "__name__", repr(on_expired)))

    def register(self, block_id: int, session_id: str,
                 expire_at: float) -> None:
        """注册或更新一个 block 的 TTL。

        如果 ``(block_id, session_id)`` 已存在：
          - 当新的 ``expire_at`` 比旧的更晚时，先从 timer wheel 移除旧
            entry，更新 expire_at 后重新插入；
          - 否则忽略（保留更晚的过期时间）。
        如果不存在：创建新的 TTLBlockEntry 并插入 timer wheel。
        """
        key = (block_id, session_id)
        if key in self._entries:
            old_entry = self._entries[key]
            if expire_at > old_entry.ttl_expire_at:
                self._timer_wheel.remove(old_entry)
                old_entry.ttl_expire_at = expire_at
                self._timer_wheel.insert(old_entry, expire_at)
        else:
            entry = TTLBlockEntry(
                block_id=block_id,
                session_id=session_id,
                ttl_expire_at=expire_at,
            )
            self._entries[key] = entry
            self._timer_wheel.insert(entry, expire_at)
        logger.info(f"Register block_id {block_id} and session id {session_id} with ttl {expire_at} in TTL Manager")

    def update(self, block_id: int, session_id: str,
               new_expire_at: float) -> None:
        """更新 block 的 TTL，语义等同于 ``register``。"""
        self.register(block_id, session_id, new_expire_at)

    def remove(self, block_id: int, session_id: str) -> None:
        """显式移除一个 block 的 TTL 跟踪。

        同时从 ``_entries`` 和 timer wheel 中删除。如果不存在则静默忽略。
        """
        key = (block_id, session_id)
        if key in self._entries:
            entry = self._entries.pop(key)
            self._timer_wheel.remove(entry)
        else:
            logger.info(f"Could not find block_id {block_id} and session id {session_id} in TTL Manager")

    def tick(self, now: float | None = None) -> None:
        """推进 timer wheel，处理所有已过期的 entry。

        对每个由 timer wheel 收集到的过期 entry，先校验 ``now >= expire_at``
        （timer wheel 是 best-effort，slot 里可能含尚未真正过期的 entry），
        然后从 ``_entries`` 移除并调用 ``on_expired`` 回调。

        Parameters
        ----------
        now : float, optional
            当前时间戳，默认 ``time.monotonic()``。
        """
        now = now if now is not None else time.monotonic()
        expired_entries = self._timer_wheel.advance(now)
        for entry in expired_entries:
            if now >= entry.ttl_expire_at:
                self._entries.pop(
                    (entry.block_id, entry.session_id), None)
                self._on_expired(entry.block_id, entry.session_id)

def example_expired_callback(block_id: int, session_id: str) -> None:
    logger.info(f"block_id {block_id} and session_id {session_id} is processing on TTL expiration")

class SessionController:
    #TODO: 考虑预取请求有content / session已被清理，重新计算hash
    """Context management edits 执行器。

    处理请求携带的 context_management.edits，根据
    manage_request 标志决定是立即执行还是在请求完成后执行。
    通过注册回调执行实际的 block 操作，不感知具体的
    offload/prefetch/evict 实现。

    Required callbacks
    ------------------
    execute_offload  (block_ids: list[int], session_id: str) -> None
        执行 offload 操作。
    execute_prefetch (block_ids: list[int], session_id: str) -> None
        执行 prefetch 操作。
    execute_evict   (block_ids: list[int], session_id: str) -> None
        执行 evict 操作。
    """

    _REQUIRED_CALLBACKS: list[str] = [
        "execute_offload",
        "execute_prefetch",
        "execute_evict",
        "get_global_block_id_by_session"
        "get_block_hashes_by_session"
    ]

    def __init__(
        self,
        callbacks: dict[str, Callable[..., Any]] | None = None,
        **kw_callbacks: Callable[..., Any],
    ):
        """
        Parameters
        ----------
        callbacks : dict[str, Callable], optional
            A mapping of {name: fn} to register in bulk.
        **kw_callbacks
            Same as *callbacks* but passed as keyword arguments.
        """
        self._registry: dict[str, Callable[..., Any]] = {}

        if callbacks:
            self._registry.update(callbacks)
        if kw_callbacks:
            self._registry.update(kw_callbacks)

        missing = [
            name for name in self._REQUIRED_CALLBACKS
            if name not in self._registry
        ]
        if missing:
            raise ValueError(
                f"Missing required callbacks: {missing}. "
                f"Provide them via `callbacks` dict or keyword arguments."
            )

        # request_id → [(edit, session_id), ...]
        self._pending_edits: dict[str, list[tuple[ContextManagementEditsParams, str]]] = {}

        logger.info("SessionController initialized with callbacks: %s",
                    list(self._registry))

    # ------------------------------------------------------------------
    #  Registry management
    # ------------------------------------------------------------------

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register (or overwrite) a callback by *name*."""
        self._registry[name] = fn
        logger.debug("Registered callback: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a callback.  Required callbacks cannot be removed."""
        if name in self._REQUIRED_CALLBACKS:
            raise ValueError(
                f"Cannot unregister required callback '{name}'."
            )
        self._registry.pop(name, None)
        logger.debug("Unregistered callback: %s", name)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def process_request_edits(
        self,
        request_id: str,
        session_id: str | None,
        context_management: "ContextManagementParams | None",
    ) -> list[EditResponse] | None:
        """处理请求携带的 context_management edits。

        Parameters
        ----------
        request_id : str
            请求唯一标识。
        session_id : str | None
            请求所属的 session。为 None 时直接忽略 edits。
        context_management : ContextManagementParams | None
            请求中的 context_management 字段。
        """
        if context_management is None or context_management.edits is None:
            return

        if session_id is None:
            logger.warning(
                "request %s has context_management.edits but no session_id, "
                "skipping.", request_id)
            return

        if context_management.manage_request:
            # 管理请求：不执行请求本身，直接执行所有 edits
            logger.info(
                "Processing manage_request edits for request %s, "
                "session %s, %d edits.",
                request_id, session_id, len(context_management.edits))
            edit_results = []
            for edit in context_management.edits:
                edit_results.append(self._execute_single_edit(edit, session_id))
            return edit_results
        else:
            # 普通请求：记录 edits，在请求完成后执行
            logger.info(
                "Deferring %d edits for request %s, session %s. Prefetch will not be performed after request finish",
                len(context_management.edits), request_id, session_id)
            self._pending_edits[request_id] = [
                (edit, session_id) for edit in context_management.edits if edit.type != "prefetch"
            ]

    def on_request_completed(self, request_id: str) -> None:
        """请求完成后执行其挂起的 edits。"""
        pending = self._pending_edits.pop(request_id, None)
        if pending is None:
            return

        logger.info(
            "Executing %d deferred edits for completed request %s.",
            len(pending), request_id)
        for edit, session_id in pending:
            self._execute_single_edit(edit, session_id)

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------
    def _generate_global_ids(self, session_id: str) -> list[int]:
        callback_name = f"get_global_block_id_by_session"
        fn = self._registry.get(callback_name)
        global_block_ids = []
        if fn is None:
            logger.warning("No callback registered for get_global_block_id_by_session")
            return global_block_ids
        global_block_ids = fn(session_id)

        if len(global_block_ids) == 0:
            logger.warning(
                f"Could not find session {session_id} with block ref record in SAM, failed to perform context management edit")
            return global_block_ids

        return global_block_ids

    def _execute_single_edit(
        self,
        edit: ContextManagementEditsParams,
        session_id: str,
    ) -> EditResponse:
        """执行单个 edit，通过注册的回调执行实际操作。"""
        # block_start / block_end 由 pymotor 从 message index 转换而来
        # 返回格式：EditResponse,记录op操作，session id，状态及详细信息
        if edit.target != "session" and (edit.block_start is None or edit.block_end is None):
            logger.info(
                "Edit type=%s has no block_start/block_end, skipping. edit=%s",
                edit.type, edit)
            return EditResponse(
                session_id=session_id,
                type=edit.type,
                op_staus=False,
                fail_reason=f"invalid block start {edit.block_start} or block end {edit.block_end}"
            )

        logger.info(
            "Executing edit type=%s for session %s"
            "(block_start=%d, block_end=%d).",
            edit.type, session_id, edit.block_start, edit.block_end)

        callback_name = f"execute_{edit.type}"
        fn = self._registry.get(callback_name)
        if fn is None:
            logger.warning("No callback registered for edit type: %s, skipping.", edit.type)
            return EditResponse(
                session_id=session_id,
                type=edit.type,
                op_staus=False,
                fail_reason=f"invalid edit type {edit.type}"
            )



        actual_process_blocks = 0
        op_result = True
        fail_reason = ''
        is_session_op = edit.type == "session"

        if edit.type == "evict":
            global_block_ids = self._generate_global_ids(session_id)
            logger.info(
                f"edit processing info: global_block_ids {global_block_ids} and process block num {len(global_block_ids)}")
            if len(global_block_ids) == 0:
                return EditResponse(
                    session_id=session_id,
                    type=edit.type,
                    op_staus=False,
                    fail_reason=f"No block record for session {session_id}"
                )

            op_result, fail_reason, result_target = self.process_edit_index(edit, global_block_ids)
            actual_process_blocks = fn(session_id, result_target, is_session_op)
        elif edit.type == "prefetch":
            callback_name = f"get_block_hashes_by_session"
            fn = self._registry.get(callback_name)
            session_hashes = fn(session_id)
            op_result, fail_reason, result_target = self.process_edit_index(edit, session_hashes)
            actual_process_blocks = fn(session_id, result_target)
        else:
            fn(session_id)
            actual_process_blocks = edit.block_end - edit.block_start

        return EditResponse(
            session_id=session_id,
            type=edit.type,
            op_staus=op_result,
            expected_op_block_num=edit.block_end - edit.block_start,
            actual_op_block_num=actual_process_blocks,
            fail_reason=fail_reason
        )

    def process_edit_index(self, edit:ContextManagementEditsParams, candidate_list: list[Any]) -> tuple[bool, str, list[Any]]:
        if edit.target == "session" and (edit.block_start is None or edit.block_end is None):
            edit.block_start = 0
            edit.block_end = len(candidate_list)

        if edit.block_end > len(candidate_list) or edit.block_start > len(candidate_list):
            logger.warning(f"edit index out of range: block end {edit.block_end} or block start {edit.block_start} "
                           f"is out of index, the total kv length is {len(candidate_list)}, fail to perform edit {edit.type}")
            fail_reason = f"block start {edit.block_start} or block end {edit.block_end}"
            edit.block_start = edit.block_end = 0
            return (False, fail_reason, [])

        return (True, '', candidate_list[edit.block_start:edit.block_end])


def compute_ephemeral_range(
    cache_control: CacheControlParams,
) -> EphemeralRange | None:
    if cache_control is None:
        return None
    block_offset = cache_control.block_offset or 0  # 默认所有 block 都受保护
    return EphemeralRange(
        block_offset=block_offset,
        ttl=cache_control.ttl,
    )