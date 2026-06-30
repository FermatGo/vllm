#TODO: 某session对应的部分block，由于同时被其他block占用，无法把完整的全部block加入free queue当中，如何进行block粒度的保护

from typing import Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BlockOpType(str, Enum):
    OFFLOAD = "offload"
    PREFETCH = "prefetch"
    EVICT = "evict"


@dataclass
class CacheModifiedInfo:
    session_id: str
    local_start_block_id: Optional[int] = 0
    local_stop_block_id: Optional[int] = None
    updated_ttl: Optional[int] = None
    cache_action: Optional[BlockOpType] = None


class AgenticCacheTTLManager():
    """Generic TTL manager that delegates all data-access and state-transition
    logic to a callable registry.  The manager itself only makes *decisions*
    (which blocks need attention); registered callbacks carry out the actual
    data-structure changes.

    Manager 不感知具体的区域划分（A/B/C 等），只通过两个回调表达
    语义上的"降级"操作：
      - demote_ttl_expired : TTL 过期的 block 需要降级
      - demote_no_session  : 没有 session 的 block 需要降级

    Optional callbacks
    ------------------
    demote_to_prevent_oom  () -> list[int]
        Select block IDs to demote for OOM prevention (e.g. when all blocks
        are in active zone and none can be reclaimed through normal TTL
        or session checks).  The manager will pass the returned IDs to
        demote_ttl_expired.

    Built-in callback names
    -----------------------
    iter_blocks            () -> Iterable[Any]
        Iterate over all managed blocks.
    get_block_by_id        (block_id: int) -> Any | None
        Retrieve a single block by its ID.
    filter_by_ttl_expired  () -> list[int]
        Return block IDs whose TTL deadline <= time.monotonic().
    filter_by_no_session   () -> list[int]
        Return block IDs that have no session bound.
    demote_ttl_expired     (block_ids: list[int]) -> None
        Carry out the state transition for TTL-expired blocks.
    demote_no_session      (block_ids: list[int]) -> None
        Carry out the state transition for session-free blocks.
    apply_modification     (info: CacheModifiedInfo) -> None
        Apply a modification described by *info* to the corresponding block(s).
        *local_start_block_id* / *local_stop_block_id* are **local indices**
        within the session's block list (0-based, ordered by block_id), NOT
        global block IDs.  The callback maps local indices to actual blocks.
        When *local_stop_block_id* is None, the callback should apply to
        all blocks from *start* to the session's last block.
    """

    _REQUIRED_CALLBACKS: list[str] = [
        "iter_blocks",
        "get_block_by_id",
        "filter_by_ttl_expired",
        "filter_by_no_session",
        "demote_ttl_expired",
        "demote_no_session",
        "apply_modification",
    ]

    def __init__(
        self,
        callbacks: Optional[dict[str, Callable[..., Any]]] = None,
        **kw_callbacks: Callable[..., Any],
    ):
        """
        Parameters
        ----------
        callbacks : dict[str, Callable], optional
            A mapping of ``{name: fn}`` to register in bulk.
        **kw_callbacks
            Same as *callbacks* but passed as keyword arguments.

        Both sources are merged; keyword arguments take precedence
        when a name appears in both.
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

        logger.info("AgenticCacheTTLManager initialized with "
                     "callbacks: %s", list(self._registry))

    # ------------------------------------------------------------------
    #  Registry management
    # ------------------------------------------------------------------

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Register (or overwrite) a callback by *name*."""
        self._registry[name] = fn
        logger.debug("Registered callback: %s", name)

    def unregister(self, name: str) -> None:
        """Remove a callback.  Built-in required names cannot be removed."""
        if name in self._REQUIRED_CALLBACKS:
            raise ValueError(
                f"Cannot unregister required callback '{name}'."
            )
        self._registry.pop(name, None)
        logger.debug("Unregistered callback: %s", name)

    def get_callback(self, name: str) -> Callable[..., Any]:
        """Retrieve a registered callback by *name*."""
        if name not in self._registry:
            raise KeyError(
                f"Callback '{name}' is not registered. "
                f"Available: {list(self._registry)}"
            )
        return self._registry[name]

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def get_all_cache_info(self):
        """Return information for every managed block."""
        return list(self._registry["iter_blocks"]())

    def get_cache_info_by_id(self, block_id: int):
        """Return information for a single block identified by *block_id*."""
        return self._registry["get_block_by_id"](block_id)

    def check_cache_ttl(self):
        """Check TTL status and execute demotion via registered callbacks.

        Decision logic:
          1. filter_by_ttl_expired → 找出 TTL 过期的 block
             → demote_ttl_expired  执行降级
          2. filter_by_no_session  → 找出无 session 的 block
             → demote_no_session   执行降级

        Manager 不感知具体区域，降级语义由注册方定义。

        Returns
        -------
        (expired_ids, no_session_ids) : tuple[list[int], list[int]]
            IDs of blocks processed in each step.
        """
        # Step 1: TTL 过期 → 降级
        expired_ids: list[int] = self._registry["filter_by_ttl_expired"]()
        if expired_ids:
            logger.info("TTL expired blocks: %s, demoting", expired_ids)
            self._registry["demote_ttl_expired"](expired_ids)
        else:
            logger.debug("No TTL-expired blocks found")

        # Step 2: 无 session → 降级
        no_session_ids: list[int] = self._registry["filter_by_no_session"]()
        if no_session_ids:
            logger.info("Session-free blocks: %s, demoting", no_session_ids)
            self._registry["demote_no_session"](no_session_ids)
        else:
            logger.debug("No session-free blocks found")

        return expired_ids, no_session_ids

    def prevent_oom(self):
        """Force-demote blocks to prevent OOM.

        Called when normal TTL / session checks cannot free enough blocks
        (e.g. all blocks are in active zone).  If the optional callback
        ``demote_to_prevent_oom`` is registered, it selects block IDs;
        the manager then passes them to ``demote_ttl_expired``.

        Returns
        -------
        demoted_ids : list[int]
            IDs of blocks that were demoted, or empty list if the
            callback is not registered or returned nothing.
        """
        fn = self._registry.get("demote_to_prevent_oom")
        if fn is None:
            logger.debug("demote_to_prevent_oom not registered, skipping")
            return []
        demoted_ids: list[int] = fn()
        if demoted_ids:
            logger.warning("OOM prevention: force-demoting blocks %s",
                           demoted_ids)
            self._registry["demote_ttl_expired"](demoted_ids)
        else:
            logger.debug("OOM prevention: no blocks selected")
        return demoted_ids

    def run(self):
        """
        Main function to run ttl-manager. Required to check ttl and optional to prevent OOM
        """
        self.check_cache_ttl()
        self.prevent_oom()

    def modify_cache_info(self, cache_todo_info: CacheModifiedInfo):
        """Apply the modification described in *cache_todo_info*.

        Manager 只负责调用，具体 local→global 映射和遍历逻辑由注册的
        apply_modification 回调处理。local_start/stop 是 session 内的
        local index，与全局 block_id 无关。
        """
        logger.info("modify_cache_info: session=%s, local_start=%s, "
                     "local_stop=%s, action=%s",
                     cache_todo_info.session_id,
                     cache_todo_info.local_start_block_id,
                     cache_todo_info.local_stop_block_id,
                     cache_todo_info.cache_action)
        self._registry["apply_modification"](cache_todo_info)

