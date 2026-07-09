# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import itertools
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.request import Request
from vllm.entrypoints.openai.chat_completion.protocol import CacheControlParams
from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.session_aware_pooling_manager import SessionEventListener


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

class SessionAwareManager:
    """Session-Aware Manager — 管理 session 生命周期与 block 映射"""

    def __init__(self, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager

        # Session 注册表
        self._sessions: dict[str, SessionInfo] = {}

        # session_id → {block_id → SessionBlockRecord}
        # 反向索引：block_id → {session_id → SessionBlockRecord}（可以进一步增加包装类BlockInfo）
        self._session_blocks: dict[str, dict[int, SessionBlockRecord]] = {}
        self._block_sessions: dict[int, dict[str, SessionBlockRecord]] = {}

        # TTL 管理器（定时轮）
        self._ttl_manager = TTLManager(on_expired=self._on_ttl_expired)

        # Session 控制器
        self._session_controller = SessionController(self)

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
            is_ephemeral = (
                ephemeral_start is not None and idx >= ephemeral_start
            )

            # 清零旧的 session 引用（SAM 内部）
            self._clear_block_session_refs(block_id)

            # 添加新的 session 引用（SAM 内部）
            ttl_expire_at = 0.0
            if is_ephemeral and ephemeral_range:
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
                new_expire = max(
                    record.ttl_expire_at,
                    time.monotonic() + (record.ttl_expire_at - record.created_at)
                )
                record.ttl_expire_at = new_expire
                self._ttl_manager.update(block_id, session_id, new_expire)
                # 刷新 block 上的 TTL
                self.kv_cache_manager.update_block_meta(
                    block_id, ttl_expire_at=new_expire
                )
            return

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
    
    def _on_ttl_expired(self, block_id: int, session_id: str) -> None:
        """ephemeral block TTL 到期回调"""

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

    def free_session(self, session_id: str) -> dict:
        """清理指定 session 的所有 block 引用"""
        freed_blocks = 0
        orphaned_blocks = 0

        if session_id in self._session_blocks:
            for block_id in list(self._session_blocks[session_id].keys()):
                self._remove_session_block_ref(session_id, block_id)
                freed_blocks += 1

                # 检查 block 是否已无任何 session 引用
                if block_id not in self._block_sessions:
                    orphaned_blocks += 1

                self.kv_cache_manager.update_block_meta(block_id, delta_ref=-1)

        # 移除 session 注册信息
        if session_id in self._sessions:
            parent_session_id = self._sessions[session_id].parent_session_id
            if parent_session_id and parent_session_id in self._sessions:
                self._sessions[parent_session_id].children.discard(session_id)
            del self._sessions[session_id]

        return {
            "session_id": session_id,
            "freed_blocks": freed_blocks,
            "orphaned_blocks": orphaned_blocks,
        }

    def free_session_tree(self, session_id: str) -> dict:
        """递归清理session及其所有子session"""
        session_info = self._sessions.get(session_id)
        if session_info is None:
            return {
                "session_id": session_id,
                "freed_blocks": 0,
                "orphaned_blocks": 0,
                "children_freed": [],
            }

        children_freed = []

        for child_sid in list(self._sessions[session_id].children):
            child_result = self.free_session_tree(child_sid)
            children_freed.append(child_result)

        result = self.free_session(session_id)

        logger.info(
            f"Free session tree for session {session_id}: "
            f"current session_to_blocks: {self._session_blocks}, "
            f"session_info: {self._sessions}, "
        )

        return {
            "session_id": session_id,
            "freed_blocks": result["freed_blocks"],
            "orphaned_blocks": result["orphaned_blocks"],
            "children_freed": children_freed,
        }
    
    # def _mark_block_hash_evictable(self, block_id: int) -> None:
        # """标记 block hash 可被惰性清除（SAM 内部）"""
    
    def _execute_offload(self, block_ids: list[int], session_id: str) -> None:
        """卸载指定范围的 block — 减少 session 引用 + 清除当前session的TTL（通知TTLManager）"""
        for block_id in block_ids:
            self._remove_session_block_ref(session_id, block_id)
            self.kv_cache_manager.update_block_meta(block_id, delta_ref=-1)

    def _execute_prefetch(self, block_ids: list[int], session_id: str) -> None:
        """预取: 分配新的blcok，添加session信息，加载cache（hash）"""
        for block_id in block_ids:
            self._clear_block_session_refs(block_id)

    def _execute_evict(self, block_ids: list[int], session_id: str) -> None:
        """驱逐指定范围的 block — 减少引用 + 清除当前session的TTL + 标记清除（session引用归0）"""
        for block_id in block_ids:
            self._remove_session_block_ref(session_id, block_id)
            # session_ref_cnt -1 + 清除 TTL
            self.kv_cache_manager.update_block_meta(
                block_id, delta_ref=-1, ttl_expire_at=0.0
            )
            # 标记 block hash 可被惰性清除(待定)
            # self._mark_block_hash_evictable(block_id)
    
    def add_event_listener(self, listener: SessionEventListener):
        """注册事件监听器（SPM 调用）"""
        self._event_listeners.append(listener)

    def _notify_event(self, event_type: str, **kwargs):
        """通知所有监听器"""
        for listener in self._event_listeners:
            handler = getattr(listener, f"on_{event_type}", None)
            if handler is not None:
                handler(**kwargs)
    





class TTLManager:
    """Block 级 TTL 管理器"""


class SessionController:
    """Context management edits 执行行器"""


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