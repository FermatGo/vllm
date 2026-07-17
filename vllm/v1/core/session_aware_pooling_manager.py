from dataclasses import dataclass
from vllm.logger import init_logger
import time
import threading

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.v1.core.session_aware_manager import SessionAwareManager
from vllm.v1.core.session_event_listener import SessionEventListener
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.request import Request

logger = init_logger(__name__)

@dataclass
class SPMConfig:
    """SPM 配置"""
    # Keep-Alive 配置
    enable_keep_alive: bool = True         # 是否启用 Keep-Alive
    keep_alive_interval: int = 60          # Keep-Alive 刷新间隔（秒）
    max_keys_per_cycle: int = 1024        # 每轮最多刷新的 key 数

    # 驱逐配置
    enable_eviction: bool = True           # 是否启用主动驱逐
    eviction_grace_period: float = 30.0   # 驱逐宽限期（秒）

    # 预取配置
    enable_prefetch: bool = True          # 是否启用主动预取
    prefetch_max_queue_size: int = 16     # 预取队列最大长度
    prefetch_block_reserve: int = 8        # 为预取保留的空闲 block 数量


@dataclass
class PrefetchRequest:
    """预取请求描述"""
    session_id: str
    request_id: str              # 关联的请求 ID
    block_hashes: list[BlockHash]      # 需要预取的 block hash 列表
    pool_keys: list[str]        # 对应的 PoolKey 列表
    token_len: int               # 需要预取的 token 数量
    created_at: float            # 创建时间
    dest_block_ids: list[int]    # 待搬入block ids
    priority: int = 0            # 优先级（0=最高，由 manage_request 触发）


@dataclass
class EvictionMark:
    """驱逐标记 — 标记 session 的远端 KV cache 为可驱逐"""
    session_id: str
    pool_keys: list[str]         # 被标记的 PoolKey
    evict_at: float             # 预期驱逐时间（给 Keep-Alive 一点缓冲）
    is_partial: bool = False   # 是否部分驱逐


class SessionKeyTracker:
    """跟踪每个 session 写入远端存储的 PoolKey"""
    _instance_lock = threading.Lock()

    def __init__(self):
        pass

    def __new__(cls):
        if not hasattr(SessionKeyTracker, "_instance"):
            with SessionKeyTracker._instance_lock:
                if not hasattr(SessionKeyTracker, "_instance"):
                    SessionKeyTracker._instance = object.__new__(cls)
                    # session_id → {PoolKey → block_hash}
                    SessionKeyTracker._instance._session_keys: dict[str, dict[str, BlockHash]] = {}
                    # 反向索引session_id → {block_hash → PoolKey}
                    SessionKeyTracker._instance._session_hashes: dict[str, dict[BlockHash, str]] = {}
                    # 反向索引：PoolKey string → {session_id}
                    SessionKeyTracker._instance._key_sessions: dict[str, set[str]] = {}
                    # {block_hash →PoolKey}
                    SessionKeyTracker._instance._hash_keys: dict[BlockHash, str] = {}
                    SessionKeyTracker._instance._lock = threading.Lock()
        return SessionKeyTracker._instance

    def add_keys(
        self,
        session_id: str,
        keys: list[str],
        block_hashes: list[BlockHash],
    ) -> None:
        """Session 的 KV cache 被 put 到远端时记录。
        幂等调用：如果 key 已存在，仅更新反向索引。
        """
        with self._lock:
            if session_id not in self._session_keys:
                self._session_keys[session_id] = {}
                self._session_hashes[session_id] = {}
            for key, bh in zip(keys, block_hashes):
                self._session_keys[session_id][key] = bh
                self._session_hashes[session_id][bh] = key
                self._hash_keys[bh] = key
                if key not in self._key_sessions:
                    self._key_sessions[key] = set()
                self._key_sessions[key].add(session_id)
                logger.info(f"SessionKeyTracker: add blocks: session_id: {session_id},  keys: {keys}, block_hashes: {block_hashes}")

    def remove_session(self, session_id: str) -> list[str]:
        """Session 被清理时移除所有 key 关联。
        返回该 session 独有的、不再被其他 session 引用的 PoolKey 列表。
        """
        with self._lock:
            session_keys = self._session_keys.pop(session_id, {})
            self._session_hashes.pop(session_id, {})
            orphaned_keys = []
            for key in session_keys:
                if session_id in self._key_sessions.get(key, set()):
                    self._key_sessions[key].discard(session_id)
                    if not self._key_sessions[key]:
                        # 无其他 session 引用，该 key 不再需要 Keep-Alive
                        self._key_sessions.pop(key, None)
                        orphaned_keys.append(key)
            logger.info(f"SessionKeyTracker: remove session: session_id: {session_id},  remove keys: {orphaned_keys}")
            return orphaned_keys

    def get_key_by_block_hash(self, block_hash: BlockHash) -> str:
        """根据session_id从block_hash反查pool_key。
        """
        with self._lock:
            pool_key = self._hash_keys.get(block_hash, None)
            return pool_key


    def remove_keys(self, session_id: str, keys: list[str]) -> list[str]:
        """移除 session 对特定 key 的关联（用于部分驱逐）。
        返回不再被任何 session 引用的 PoolKey 列表。
        """
        with self._lock:
            orphaned_keys = []
            for key in keys:
                if session_id in self._session_keys:
                    bh = self._session_keys[session_id].pop(key, None)
                    self._session_hashes[session_id].pop(bh, None)
                if key in self._key_sessions:
                    self._key_sessions[key].discard(session_id)
                    if not self._key_sessions[key]:
                        self._key_sessions.pop(key, None)
                        orphaned_keys.append(key)
            logger.info(f"SessionKeyTracker: remove keys: {orphaned_keys}")
            return orphaned_keys

    def get_active_keys(
        self,
        session_ids: list[str] | None = None
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
            logger.info(f"SessionKeyTracker: active keys: {result}")
            return result

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
        connector: KVConnectorBase_V1,
        session_key_tracker: SessionKeyTracker,
        interval: int = 60,           # 刷新间隔（秒）
        max_keys_per_cycle: int = 0,  # 每轮最多刷新的 key 数（0=不限）
    ):
        super().__init__(daemon=True, name="KVCacheKeepAliveThread")
        self.connector = connector
        self.tracker = session_key_tracker
        self.interval = interval
        self.max_keys = max_keys_per_cycle
        self._stopped = threading.Event()

    def run(self):
        #TODO: max keys改为chunk发送
        while not self._stopped.wait(self.interval):
            try:
                keys = self.tracker.get_active_keys()
                if not keys:
                    continue
                for i in range(0, len(keys), self.max_keys):
                    batch = keys[i:i+self.max_keys]
                    res = self.connector.look_up_keys(batch)
                    logger.info(f"Keep-alive thread return: {res}")
            except Exception as e:
                logger.error("Keep-alive thread error: %s", e)

    def stop(self):
        self._stopped.set()


class SessionAwarePoolingManager(SessionEventListener):
    """Session-Aware Pooling Manager — 池化场景下的 session 感知管理

    实现 SessionEventListener 协议，注册为 SAM 的事件监听器。
    持有 SessionKeyTracker 和 KVCacheKeepAliveThread。
    """
    # TODO：获取block hash，及其他meta信息计算 pool key
    # TODO：SAM定期向SPM写入需要保护的hash，暴露增加、删除
    # TODO：检查ttl接口是否正确
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
        self.key_tracker = SessionKeyTracker()

        # Keep-Alive 线程
        self.keep_alive_thread: KVCacheKeepAliveThread | None = None

        # 预取队列
        self.prefetch_waiting_queue: list[PrefetchRequest] = []

        # 进行预取中的队列
        self.prefetch_running_queue: list[Request] = []

        # 驱逐标记队列
        self._eviction_marks: dict[str, EvictionMark] = {}

        # 注册为 SAM 事件监听器
        sam.add_event_listener(self)

        self.block_size = 0

    def start(self) -> None:
        """启动 Keep-Alive 线程"""
        if self.config.enable_keep_alive and self.connector is not None:
            #TODO: 修改线程入口
            self.keep_alive_thread = KVCacheKeepAliveThread(
                connector=self.connector,
                session_key_tracker=self.key_tracker,
                interval=self.config.keep_alive_interval,
                max_keys_per_cycle=self.config.max_keys_per_cycle,
            )
            self.keep_alive_thread.start()
            logger.info("Keep-alive thread start.")

    def stop(self) -> None:
        """停止 Keep-Alive 线程"""
        if self.keep_alive_thread is not None:
            self.keep_alive_thread.stop()
            logger.info("Keep-alive thread stop.")

    # --- SessionEventListener 实现 ---

    def on_session_registered(self, session_id: str, parent_session_id: str | None) -> None:
        """Session 注册时初始化 tracker 记录"""
        # key_tracker 按需初始化，无需预分配
        return

    # 可以通过ascend的pool_worker的回调函数来调用key_tracker.add_keys
    def on_session_blocks_allocated(
        self,
        session_id: str,
        block_ids: list[int],
        pool_keys: list[str],
        block_hashes: list[BlockHash],
    ) -> None:
        """block 被分配且 KV cache 被写入远端后记录 PoolKey"""
        pass

    def on_session_cache_hit(
        self,
        session_id: str,
        block_id: int,
        # pool_key: str | None,
        block_hash: BlockHash | None,
    ) -> None:
        """prefix cache 命中时记录 PoolKey（幂等）"""
        if block_hash is not None:
            pool_key = self.key_tracker.get_key_by_block_hash(block_hash)
            block_key = self.key_tracker._session_hashes.get(session_id, None)
            if block_key is not None:
                key = block_key.get(block_hash, None)
                if key is not None:
                    return
            self.key_tracker.add_keys(session_id, [pool_key], [block_hash])

    def on_session_ttl_expired(self, session_id: str, block_ids: list[int], block_hashs: list[BlockHash]) -> None:
        """TTL 到期时检查远端 KV cache 是否需驱逐"""
        if not self.config.enable_eviction:
            return
        # TTL 到期的 block 可能对应的 PoolKey 仍有其他 session 引用
        # 需检查每个 block 对应的 PoolKey
        # for block_id in block_ids:
        #     pool_keys = self._get_block_pool_keys(block_id)
        #     for key in pool_keys:
        #         remaining = self.key_tracker.get_key_sessions(key)
        #         if not remaining:
        #             self._mark_for_eviction(session_id, [key], is_partial=True)
        for block_hash in block_hashs:
            pool_key = self.key_tracker.get_key_by_block_hash(block_hash)
            remaining = self.key_tracker.get_key_sessions(pool_key)
            if not remaining:
                self._mark_for_eviction(session_id, [pool_key], is_partial=True)

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
        # pool_keys: list[str],
        block_hashes: list[BlockHash],
    ) -> None:
        """evict 操作时移除部分 PoolKey 关联"""
        logger.info(f"SessionAwarePoolingManager.on_context_management_evict: required session_id: {session_id}, required block_hashes: {block_hashes}"
                           f"SessionKeyTracker saved session_id's block_hashes: {self.key_tracker._session_keys[session_id].values()}")
        if block_hashes:
            pool_keys = []
            for block_hash in block_hashes:
                pool_key = self.key_tracker.get_key_by_block_hash(block_hash)
                if pool_key is not None:
                    pool_keys.append(pool_key)
            orphaned_keys = self.key_tracker.remove_keys(session_id, pool_keys)
            if orphaned_keys and self.config.enable_eviction:
                self._mark_for_eviction(
                    session_id, orphaned_keys, is_partial=True
                )

    def on_context_management_offload(self, session_id: str, block_ids: list[int]) -> None:
        """offload 操作时仅移除本地 block 引用，远端 KV cache 保留"""
        return

    def on_context_management_prefetch(
        self,
        session_id: str,
        logical_block_start: int,
        logical_block_end: int,
        block_ids: list[int] | None = None,
    ) -> bool:
        """prefetch 操作时创建预取请求"""
        if not self.config.enable_prefetch:
            return
        token_len = len(logical_block_end - logical_block_start) * self.block_size
        block_hashes = self.key_tracker.get_session_block_hashes(session_id)[logical_block_start:logical_block_end]
        logger.info(f"calling cb func on_context_management_prefetch with session {session_id} block_hashes {block_hashes} "
                    f"token_len {token_len} block_ids {block_ids}")
        #预取需要的参数：Pool keys, block hash以及HBM上的block id
        #TODO: 预取请求分配block ids
        #TODO: 传入参数对齐，需要block hash
        #TODO: 计算token len？如何获取 1. blocksize * block数 2. pymotor传入解析
        pool_keys = self.key_tracker.get_session_keys(session_id)[logical_block_start:logical_block_end]
        request = PrefetchRequest(
            session_id=session_id,
            request_id=f"__prefetch_{session_id}_{time.monotonic():.0f}",
            block_hashes=block_hashes,
            pool_keys=pool_keys[:len(block_hashes)],
            token_len=token_len,
            priority=0,
            created_at=time.monotonic(),
            dest_block_ids=block_ids
        )
        if len(self.prefetch_waiting_queue) < self.config.prefetch_max_queue_size:
            self.prefetch_waiting_queue.append(request)
            logger.info(f"SessionAwarePoolingManager on_context_management_prefetch: prefetch_waiting_queue added PrefetchRequest: {PrefetchRequest}")
            return True
        else:
            logger.warning(f"prefetch queue is reaching the max queue size {self.config.prefetch_max_queue_size} "
                           f"and failed to add to the queue")
            return False

    # --- 调度循环集成 ---
    def on_check_matched_token(
        self,
        session_id: str,
        check_matched_start: int,
        check_matched_end: int,
    ) -> int:
        matched_token = 0
        if check_matched_end < len(self.key_tracker.get_session_block_hashes(session_id)):
            block_hashes = self.key_tracker.get_session_block_hashes(session_id)[check_matched_start:check_matched_end]
            matched_token = self._lookup_remote_cache(block_hashes, len(block_hashes)*self.block_size)
        return matched_token

    def _lookup_remote_cache(self, block_hashes: list[BlockHash], token_len: int) -> int:
        res = self.connector.connector_scheduler.client.lookup(
            token_len=token_len,
            block_hashes=block_hashes,
        )
        return res

    def _submit_prefetch_to_scheduler(self, prefetch_req, matched_tokens):
        # 从 SAM 获取 session 已有的 block 和对应的 Request 对象
        # request = self.sam.get_request_by_session(prefetch_req.session_id)
        # if request is None:
        #     logger.warning("Session %s has no active request, skipping prefetch", prefetch_req.session_id)
        #     return
        
        # 获取 session 已有的 block_ids（通过 KVCacheManager）
        # block_ids = self.sam.get_session_block_ids(prefetch_req.session_id)
        block_ids = prefetch_req.dest_block_ids
        if not block_ids:
            logger.warning("Session %s has no allocated blocks, skipping prefetch", prefetch_req.session_id)
            return
        
        # 通过 KVPoolScheduler 注入 prefetch metadata
        self.connector.connector_scheduler.add_prefetch_request(
            prefetch_req, matched_tokens
        )

    def process_prefetch_queue(self) -> list[PrefetchRequest]:
        """在 Scheduler 调度循环中处理预取请求"""

        if not self.prefetch_waiting_queue:
            return []

        # 按 priority 排序（0=最高优先）
        # self.prefetch_waiting_queue.sort(key=lambda r: r.priority)

        completed = []
        remaining = []

        for prefetch_req in self.prefetch_waiting_queue:
            # 1. 检查预取请求是否仍然有效
            if prefetch_req.session_id not in self.sam._sessions:
                continue  # session 已不存在，跳过

            try:
                matched_tokens = self._lookup_remote_cache(
                    block_hashes=prefetch_req.block_hashes,
                    token_len=prefetch_req.token_len,
                )
                logger.info(f"lookup_remote_cache: prefetch_req with session_id {prefetch_req.session_id} "
                            f"gets matched_tokens {matched_tokens}")

                if matched_tokens > 0:
                    # 4. 创建预取请求到 Scheduler
                    # Scheduler 在下次调度时分配 block 并触发 load
                    self._submit_prefetch_to_scheduler(
                        prefetch_req, matched_tokens
                    )
                completed.append(prefetch_req)
            except Exception as e:
                logger.error("Prefetch failed for session %s: %s",
                            prefetch_req.session_id, e)
                remaining.append(prefetch_req)

        self.prefetch_waiting_queue = remaining
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