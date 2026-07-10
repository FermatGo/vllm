from dataclasses import dataclass
import logging
import time
import threading
from typing import Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.backend import Backend
from vllm.v1.core.session_aware_manager import SessionAwareManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.kv_cache_utils import BlockHash

logger = logging.getLogger(__name__)

_SESSION_KEY_TRACKER : "SessionKeyTracker" | None = None

def get_session_key_tracker() -> "SessionKeyTracker":
    global _SESSION_KEY_TRACKER
    if _SESSION_KEY_TRACKER is None:
        _SESSION_KEY_TRACKER = SessionKeyTracker()
    return _SESSION_KEY_TRACKER

@dataclass
class SPMConfig:
    """SPM 配置"""
    # Keep-Alive 配置
    enable_keep_alive: bool = False         # 是否启用 Keep-Alive
    keep_alive_interval: int = 60          # Keep-Alive 刷新间隔（秒）
    max_keys_per_cycle: int = 1024        # 每轮最多刷新的 key 数

    # 驱逐配置
    enable_eviction: bool = True           # 是否启用主动驱逐
    eviction_grace_period: float = 30.0   # 驱逐宽限期（秒）

    # 预取配置
    enable_prefetch: bool = False          # 是否启用主动预取
    prefetch_max_queue_size: int = 16     # 预取队列最大长度
    prefetch_block_reserve: int = 8        # 为预取保留的空闲 block 数量


@dataclass
class PrefetchRequest:
    """预取请求描述"""
    session_id: str
    request_id: str              # 关联的请求 ID
    block_hashes: list[str]      # 需要预取的 block hash 列表
    pool_keys: list[str]        # 对应的 PoolKey 列表
    token_len: int               # 需要预取的 token 数量
    created_at: float            # 创建时间
    priority: int = 0            # 优先级（0=最高，由 manage_request 触发）


@dataclass
class EvictionMark:
    """驱逐标记 — 标记 session 的远端 KV cache 为可驱逐"""
    session_id: str
    pool_keys: list[str]         # 被标记的 PoolKey
    evict_at: float             # 预期驱逐时间（给 Keep-Alive 一点缓冲）
    is_partial: bool = False   # 是否部分驱逐


class SessionEventListener(Protocol):
    """SPM 实现此协议，监听 SAM 的 session 生命周期事件"""

    def on_session_registered(
        self, session_id: str, parent_session_id: str | None
    ) -> None: ...

    def on_session_blocks_allocated(
        self,
        session_id: str,
        block_ids: list[int],
        pool_keys: list[str],       # 远端 PoolKey 列表（来自 AscendStoreConnector）
        block_hashes: list[str],    # 对应的 block hash
    ) -> None: ...

    def on_session_cache_hit(
        self,
        session_id: str,
        block_id: int,
        pool_key: str | None,      # cache hit 的 block 对应的远端 PoolKey（可能无）
        block_hash: str,
    ) -> None: ...

    def on_session_ttl_expired(
        self, session_id: str, block_ids: list[int]
    ) -> None: ...

    def on_session_freed(self, session_id: str) -> None: ...

    def on_context_management_evict(
        self,
        session_id: str,
        block_ids: list[int],
        pool_keys: list[str],      # 被驱逐 block 对应的远端 PoolKey
    ) -> None: ...

    def on_context_management_offload(
        self, session_id: str, block_ids: list[int]
    ) -> None: ...

    def on_context_management_prefetch(
        self,
        session_id: str,
        block_hashes: list[str],
        token_len: int,
    ) -> None: ...


class SessionKeyTracker:
    """跟踪每个 session 写入远端存储的 PoolKey"""

    def __init__(self):
        # session_id → {PoolKey string → block_hash}
        self._session_keys: dict[str, dict[str, str]] = {}
        # 反向索引：PoolKey string → {session_id}
        self._key_sessions: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def add_keys(
        self,
        session_id: str,
        keys: list[str],
        block_hashes: list[str],
    ) -> None:
        """Session 的 KV cache 被 put 到远端时记录。
        幂等调用：如果 key 已存在，仅更新反向索引。
        """
        with self._lock:
            if session_id not in self._session_keys:
                self._session_keys[session_id] = {}
            for key, bh in zip(keys, block_hashes):
                self._session_keys[session_id][key] = bh
                if key not in self._key_sessions:
                    self._key_sessions[key] = set()
                self._key_sessions[key].add(session_id)

    def remove_session(self, session_id: str) -> list[str]:
        """Session 被清理时移除所有 key 关联。
        返回该 session 独有的、不再被其他 session 引用的 PoolKey 列表。
        """
        with self._lock:
            session_keys = self._session_keys.pop(session_id, {})
            orphaned_keys = []
            for key in session_keys:
                if session_id in self._key_sessions.get(key, set()):
                    self._key_sessions[key].discard(session_id)
                    if not self._key_sessions[key]:
                        # 无其他 session 引用，该 key 不再需要 Keep-Alive
                        self._key_sessions.pop(key, None)
                        orphaned_keys.append(key)
            return orphaned_keys

    def remove_keys(self, session_id: str, keys: list[str]) -> list[str]:
        """移除 session 对特定 key 的关联（用于部分驱逐）。
        返回不再被任何 session 引用的 PoolKey 列表。
        """
        with self._lock:
            orphaned_keys = []
            for key in keys:
                if session_id in self._session_keys:
                    self._session_keys[session_id].pop(key, None)
                if key in self._key_sessions:
                    self._key_sessions[key].discard(session_id)
                    if not self._key_sessions[key]:
                        self._key_sessions.pop(key, None)
                        orphaned_keys.append(key)
            return orphaned_keys

    def get_active_keys(
        self,
        session_ids: list[str] | None = None,
        max_keys: int = 0,
    ) -> list[str]:
        """获取活跃 session 的 PoolKey（用于 Keep-Alive）。
        如果指定 session_ids，仅返回这些 session 的 key；
        否则返回所有活跃 session 的 key。
        """
        with self._lock:
            if session_ids is not None:
                all_keys = set()
                for sid in session_ids:
                    all_keys.update(self._session_keys.get(sid, {}).keys())
            else:
                all_keys = set(self._key_sessions.keys())
            result = list(all_keys)
            return result[:max_keys] if max_keys > 0 else result

    def get_session_keys(self, session_id: str) -> list[str]:
        """获取指定 session 的所有 PoolKey"""
        with self._lock:
            return list(self._session_keys.get(session_id, {}).keys())

    def get_session_block_hashes(self, session_id: str) -> list[str]:
        """获取指定 session 的所有 block hash（用于预取时构建 lookup 参数）"""
        with self._lock:
            return list(self._session_keys.get(session_id, {}).values())

    def get_key_sessions(self, key: str) -> set[str]:
        """获取指定 PoolKey 关联的所有 session（用于共享 key 判断）"""
        with self._lock:
            return self._key_sessions.get(key, set()).copy()


class KVCacheKeepAliveThread(threading.Thread):
    """周期性刷新活跃 session 的远端 KV cache LRU 位置"""

    def __init__(
        self,
        m_store: Backend,
        session_key_tracker: SessionKeyTracker,
        interval: int = 60,           # 刷新间隔（秒）
        max_keys_per_cycle: int = 0,  # 每轮最多刷新的 key 数（0=不限）
    ):
        super().__init__(daemon=True, name="KVCacheKeepAliveThread")
        self.m_store = m_store
        self.tracker = session_key_tracker
        self.interval = interval
        self.max_keys = max_keys_per_cycle
        self._stopped = threading.Event()

    def run(self):
        self.m_store.set_device()
        while not self._stopped.wait(self.interval):
            try:
                keys = self.tracker.get_active_keys(max_keys=self.max_keys)
                if keys:
                    self.m_store.exists(keys)  # exists() 更新 LRU
            except Exception as e:
                logger.error("Keep-alive thread error: %s", e)

    def stop(self):
        self._stopped.set()


class SessionAwarePoolingManager(SessionEventListener):
    """Session-Aware Pooling Manager — 池化场景下的 session 感知管理

    实现 SessionEventListener 协议，注册为 SAM 的事件监听器。
    持有 SessionKeyTracker 和 KVCacheKeepAliveThread。
    """

    def __init__(
        self,
        sam: SessionAwareManager,
        connector: KVConnectorBase_V1 | None = None,
        config: SPMConfig | None = None,
    ):
        self.sam = sam
        self.connector = connector
        self.config = config or SPMConfig()

        # SessionKeyTracker — PoolKey 跟踪
        self.key_tracker = get_session_key_tracker()

        # Keep-Alive 线程
        self.keep_alive_thread: KVCacheKeepAliveThread | None = None

        # 预取队列
        self._prefetch_queue: list[PrefetchRequest] = []

        # 驱逐标记队列
        self._eviction_marks: dict[str, EvictionMark] = {}

        # 注册为 SAM 事件监听器
        sam.add_event_listener(self)

    def start(self) -> None:
        """启动 Keep-Alive 线程"""
        if self.config.enable_keep_alive and self.connector is not None:
            self.keep_alive_thread = KVCacheKeepAliveThread(
                m_store=self.connector.connector_worker.m_store,
                session_key_tracker=self.key_tracker,
                interval=self.config.keep_alive_interval,
                max_keys_per_cycle=self.config.max_keys_per_cycle,
            )
            self.keep_alive_thread.start()

    def stop(self) -> None:
        """停止 Keep-Alive 线程"""
        if self.keep_alive_thread is not None:
            self.keep_alive_thread.stop()

    # --- SessionEventListener 实现 ---

    def on_session_registered(self, session_id: str, parent_session_id: str | None) -> None:
        """Session 注册时初始化 tracker 记录"""
        # key_tracker 按需初始化，无需预分配
        return

    def on_session_blocks_allocated(
        self,
        session_id: str,
        block_ids: list[int],
        pool_keys: list[str],
        block_hashes: list[str],
    ) -> None:
        """block 被分配且 KV cache 被写入远端后记录 PoolKey"""
        if pool_keys:
            self.key_tracker.add_keys(session_id, pool_keys, block_hashes)

    def on_session_cache_hit(
        self,
        session_id: str,
        block_id: int,
        pool_key: str | None,
        block_hash: str,
    ) -> None:
        """prefix cache 命中时记录 PoolKey（幂等）"""
        if pool_key is not None:
            self.key_tracker.add_keys(session_id, [pool_key], [block_hash])

    def on_session_ttl_expired(self, session_id: str, block_ids: list[int]) -> None:
        """TTL 到期时检查远端 KV cache 是否需驱逐"""
        if not self.config.enable_eviction:
            return
        # TTL 到期的 block 可能对应的 PoolKey 仍有其他 session 引用
        # 需检查每个 block 对应的 PoolKey
        for block_id in block_ids:
            pool_keys = self._get_block_pool_keys(block_id)
            for key in pool_keys:
                remaining = self.key_tracker.get_key_sessions(key)
                if not remaining:
                    self._mark_for_eviction(session_id, [key], is_partial=True)

    def on_session_freed(self, session_id: str) -> None:
        """Session 被清理时移除所有 PoolKey 关联并标记驱逐"""
        orphaned_keys = self.key_tracker.remove_session(session_id)
        if orphaned_keys and self.config.enable_eviction:
            self._eviction_marks[session_id] = EvictionMark(
                session_id=session_id,
                pool_keys=orphaned_keys,
                evict_at=time.monotonic() + self.config.eviction_grace_period,
            )

    def on_context_management_evict(
        self,
        session_id: str,
        block_ids: list[int],
        pool_keys: list[str],
    ) -> None:
        """evict 操作时移除部分 PoolKey 关联"""
        if pool_keys:
            orphaned_keys = self.key_tracker.remove_keys(session_id, pool_keys)
            if orphaned_keys and self.config.enable_eviction:
                self._mark_for_eviction(
                    session_id, orphaned_keys, is_partial=True
                )

    def on_context_management_offload(self, session_id: str, block_ids: list[int]) -> None:
        """offload 操作时仅移除本地 block 引用，远端 KV cache 保留"""
        # offload 仅影响本地 block，远端 KV cache 的 Keep-Alive 保护不变
        return

    def on_context_management_prefetch(
        self,
        session_id: str,
        block_hashes: list[str],
        token_len: int,
    ) -> None:
        """prefetch 操作时创建预取请求"""
        if not self.config.enable_prefetch:
            return
        pool_keys = self.key_tracker.get_session_keys(session_id)
        request = PrefetchRequest(
            session_id=session_id,
            request_id=f"__prefetch_{session_id}_{time.monotonic():.0f}",
            block_hashes=block_hashes,
            pool_keys=pool_keys[:len(block_hashes)],
            token_len=token_len,
            priority=0,
            created_at=time.monotonic(),
        )
        if len(self._prefetch_queue) < self.config.prefetch_max_queue_size:
            self._prefetch_queue.append(request)

    def _lookup_remote_cache(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,):
        return self.connector.connector_scheduler.client.lookup(token_len, block_hashes, kv_cache_group_ids)

    # --- 调度循环集成 ---

    def process_prefetch_queue(self, scheduler: Scheduler) -> list[PrefetchRequest]:
        """在 Scheduler 调度循环中处理预取请求"""

        if not self._prefetch_queue:
            return []

        # 按 priority 排序（0=最高优先）
        self._prefetch_queue.sort(key=lambda r: r.priority)

        completed = []
        remaining = []

        for prefetch_req in self._prefetch_queue:
            # 1. 检查预取请求是否仍然有效
            if prefetch_req.session_id not in self.sam._sessions:
                continue  # session 已不存在，跳过

            # 2. 检查本地 BlockPool 是否有足够的空闲 block
            # 保留 prefetch_block_reserve 个 block 用于正常请求
            available = self.sam.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
            available -= self.config.prefetch_block_reserve
            required = (prefetch_req.token_len + self.sam.kv_cache_manager.block_size - 1) \
                    // self.sam.kv_cache_manager.block_size
            if available < required:
                remaining.append(prefetch_req)  # 资源不足，延后
                continue

            # 3. 检查远端 KV cache 是否存在
            try:
                matched_tokens = self._lookup_remote_cache(
                    prefetch_req.block_hashes,
                    prefetch_req.token_len,
                )
                if matched_tokens > 0:
                    # 4. 创建预取请求到 Scheduler
                    # Scheduler 在下次调度时分配 block 并触发 load
                    self._submit_prefetch_to_scheduler(
                        prefetch_req, matched_tokens, scheduler
                    )
                completed.append(prefetch_req)
            except Exception as e:
                logger.error("Prefetch failed for session %s: %s",
                            prefetch_req.session_id, e)
                remaining.append(prefetch_req)

        self._prefetch_queue = remaining
        return completed

    def process_eviction_marks(self, now: float | None = None) -> None:
        """处理驱逐标记 — 停止 Keep-Alive 保护"""

        now = now or time.monotonic()
        expired_marks = []

        for session_id, mark in self._eviction_marks.items():
            if now >= mark.evict_at:
                # PoolKey 已从 SessionKeyTracker 中移除
                # Keep-Alive 线程不再刷新这些 key
                # 远端 LRU 自然淘汰
                expired_marks.append(session_id)

        for session_id in expired_marks:
            del self._eviction_marks[session_id]

    def _mark_for_eviction(
        self,
        session_id: str,
        pool_keys: list[str],
        is_partial: bool = False,
    ) -> None:
        mark_key = f"{session_id}_{'partial' if is_partial else 'full'}"
        existing = self._eviction_marks.get(mark_key)
        if existing is None:
            self._eviction_marks[mark_key] = EvictionMark(
                session_id=session_id,
                pool_keys=pool_keys,
                evict_at=time.monotonic() + self.config.eviction_grace_period,
                is_partial=is_partial,
            )
        else:
            # 合并 PoolKey
            existing.pool_keys.extend(pool_keys)
            existing.evict_at = max(
                existing.evict_at,
                time.monotonic() + self.config.eviction_grace_period
            )

    def _get_block_pool_keys(self, block_id: int) -> list[str]:
        """通过 block_id 查找对应的 PoolKey（需要从 KVCacheManager 获取 block_hash）"""
        # block_id → block_hash → PoolKey
        # 这需要在 block 分配时建立映射，或通过 block_pool.blocks[block_id]._block_hash 间接获取
        # 实现时需要与 KVCacheManager/AscendStoreConnector 协调
        return []