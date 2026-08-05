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
    token_len: int               # 需要预取的 token 数量
    created_at: float            # 创建时间
    dest_block_ids: tuple[list[int], ...] | list[int] | list[list[int]] | None   # 待搬入block ids
    priority: int = 0            # 优先级（0=最高，由 manage_request 触发）


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
                    SessionKeyTracker._instance._session_hashes: dict[str, list[BlockHash]] = {}
                    SessionKeyTracker._instance._lock = threading.Lock()
        return SessionKeyTracker._instance

    def get_dicts(self) -> dict[str, list[BlockHash]]:
        with self._lock:
            return dict(self._session_hashes)

    def add_hashes(
        self,
        session_id: str,
        block_hashes: list[BlockHash],
    ) -> None:
        with self._lock:
            self._session_hashes[session_id] = block_hashes
            logger.debug(f"SessionKeyTracker.add_hashes: session_id: {session_id}, block_hashes saved: {block_hashes}")

    def remove_hashes(
        self,
        session_id,
        block_hashes: list[BlockHash],
    ) -> int:
        removed_nums = 0
        with self._lock:
            # 异常处理
            if session_id is None:
                logger.warning("SessionKeyTracker.remove_hash: session_id cannot be None")
                return removed_nums
            if block_hashes is None:
                logger.warning(f"SessionKeyTracker.remove_hash: invalid block_hashes {block_hashes}")
                return removed_nums
            list_hashes = self._session_hashes.get(session_id, [])
            if not set(block_hashes).issubset(set(list_hashes)):
                logger.debug(f"SessionKeyTracker.remove_hash: invalid block_hashes {block_hashes} or invalid session_id: {session_id}")
                return removed_nums

            list_hashes = self._session_hashes.pop(session_id, [])
            removed_nums = len(list_hashes)
            logger.debug(f"SessionKeyTracker.remove_hash: session_id: {session_id}, block_hashes removed: {list_hashes}")
        return removed_nums


class KVCacheKeepAliveThread(threading.Thread):
    """周期性刷新活跃 session 的远端 KV cache LRU 位置"""

    def __init__(
        self,
        connector: KVConnectorBase_V1,
        session_key_tracker: SessionKeyTracker,
        interval: int = 60,           # 刷新间隔（秒）
        max_keys_per_cycle: int = 0,  # 每轮最多刷新的 key 数（0=不限）
        block_size: int = 0,
    ):
        super().__init__(daemon=True, name="KVCacheKeepAliveThread")
        self.connector = connector
        self.tracker = session_key_tracker
        self.interval = interval
        self.max_keys = max_keys_per_cycle
        self._stopped = threading.Event()
        self.block_size = block_size

    def run(self):
        #TODO: max keys改为chunk发送
        while not self._stopped.wait(self.interval):
            try:
                dict_hashes = self.tracker.get_dicts()
                if not dict_hashes:
                    continue
                hints_block_nums = 0
                all_block_nums = 0
                for session_id, hashes in dict_hashes.items():
                    nums = len(hashes)
                    all_block_nums = all_block_nums + nums
                    token_len = self.block_size * nums
                    res = self.connector.look_up_keys(token_len, hashes)
                    hints_block_nums = hints_block_nums + res // self.block_size
                logger.debug(f"KVCacheKeepAliveThread: all blockes numbers: {all_block_nums}, hints block numbers: {hints_block_nums}")
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
                block_size=self.block_size,
            )
            self.keep_alive_thread.start()
            logger.info("Keep-alive thread start.")

    def stop(self) -> None:
        """停止 Keep-Alive 线程"""
        if self.keep_alive_thread is not None:
            self.keep_alive_thread.stop()
            logger.info("Keep-alive thread stop.")

    # --- SessionEventListener 实现 ---

    # 可以通过ascend的pool_worker的回调函数来调用key_tracker.add_keys
    def on_session_blocks_protected(
        self,
        session_id,
        block_hashes: list[BlockHash],
    ) -> None:
        """block 保护"""
        self.key_tracker.add_hashes(session_id, block_hashes)

    def on_session_blocks_removed(
        self,
        session_id,
        block_hashes: list[BlockHash],
    ) -> int:
        """block 移除"""
        return self.key_tracker.remove_hashes(session_id, block_hashes)

    # --- 调度循环集成 ---
    def _lookup_remote_cache(self, block_hashes: list[BlockHash], token_len: int) -> int:
        res = self.connector.connector_scheduler.client.lookup(
            token_len=token_len,
            block_hashes=block_hashes,
            kv_cache_group_ids=self.connector.connector_scheduler.kv_cache_group_ids
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

    def _get_block_pool_keys(self, block_id: int) -> list[str]:
        """通过 block_id 查找对应的 PoolKey（需要从 KVCacheManager 获取 block_hash）"""
        # block_id → block_hash → PoolKey
        # 这需要在 block 分配时建立映射，或通过 block_pool.blocks[block_id]._block_hash 间接获取
        # 实现时需要与 KVCacheManager/AscendStoreConnector 协调
        return []

    def on_session_blocks_allocated(
        self,
        session_id: str | None = None,
        # block_ids: list[int] = None,
        # pool_keys: list[str] = None,       # 远端 PoolKey 列表（来自 AscendStoreConnector）
        block_hashes: list[BlockHash] = None,    # 对应的 block hash
    ) -> None:
        return

    def on_session_cache_hit(
        self,
        session_id: str | None = None,
        block_id: int | None = None,
        block_hash: BlockHash | None = None,
    ) -> None:
        return

    def on_session_ttl_expired(
        self,
        session_id: str | None = None,
        block_hashes: list[BlockHash] | None = None,
    ) -> int:
        return self.on_session_blocks_removed(session_id, block_hashes)

    def on_context_management_prefetch(
        self,
        session_id: str | None = None,
        logical_block_start: int | None = None,
        logical_block_end: int = None,
        block_ids: list[int] | None = None,
        block_hashes: list[BlockHash] = None,
    ) -> bool:
        """prefetch 操作时创建预取请求"""
        if not self.config.enable_prefetch:
            return True
        if block_hashes is None:
            logger.warning("block_hashes can not be None")
            return True
        token_len = len(block_hashes) * self.block_size
        logger.debug(f"calling cb func on_context_management_prefetch with session {session_id} block_hashes {block_hashes} "
                    f"token_len {token_len}")
        logger.debug(f"SessionAwarePoolingManager.on_context_management_prefetch: token_len: {token_len}")
        logger.debug(f"SessionAwarePoolingManager.on_context_management_prefetch: block_hashes: {block_hashes}")
        logger.debug(f"SessionAwarePoolingManager.on_context_management_prefetch: block_hashes_len: {len(block_hashes)}")
        request = PrefetchRequest(
            session_id=session_id,
            request_id=f"__prefetch_{session_id}_{time.monotonic():.0f}",
            block_hashes=block_hashes,
            token_len=token_len,
            priority=0,
            created_at=time.monotonic(),
            dest_block_ids=None,
        )
        if len(self.prefetch_waiting_queue) < self.config.prefetch_max_queue_size:
            self.prefetch_waiting_queue.append(request)
            logger.debug(f"SessionAwarePoolingManager.on_context_management_prefetch: prefetch_waiting_queue added PrefetchRequest: {PrefetchRequest}")
            return True
        else:
            logger.warning(f"prefetch queue is reaching the max queue size {self.config.prefetch_max_queue_size} "
                           f"and failed to add to the queue")
            return False

    def on_context_management_evict(
        self,
        session_id: str | None = None,
        block_hashes: list[BlockHash] = None,
    ) -> int:
        return self.on_session_blocks_removed(session_id, block_hashes)