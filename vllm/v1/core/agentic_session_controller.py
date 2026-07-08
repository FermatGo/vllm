from __future__ import annotations

from typing import Callable, Any

from vllm.v1.engine import ContextManagementEditsParams

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class SessionController:
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
    ) -> None:
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
            for edit in context_management.edits:
                self._execute_single_edit(edit, session_id)
        else:
            # 普通请求：记录 edits，在请求完成后执行
            logger.info(
                "Deferring %d edits for request %s, session %s.",
                len(context_management.edits), request_id, session_id)
            self._pending_edits[request_id] = [
                (edit, session_id) for edit in context_management.edits
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

    def _execute_single_edit(
        self,
        edit: ContextManagementEditsParams,
        session_id: str,
    ) -> None:
        """执行单个 edit，通过注册的回调执行实际操作。"""
        # block_start / block_end 由 pymotor 从 message index 转换而来
        if edit.block_start is None or edit.block_end is None:
            logger.info(
                "Edit type=%s has no block_start/block_end, skipping. edit=%s",
                edit.type, edit)
            return

        block_ids = list(range(edit.block_start, edit.block_end + 1))
        if not block_ids:
            return

        logger.info(
            "Executing edit type=%s for session %s on blocks %s "
            "(block_start=%d, block_end=%d).",
            edit.type, session_id, block_ids, edit.block_start, edit.block_end)

        callback_name = f"execute_{edit.type}"
        fn = self._registry.get(callback_name)
        if fn is None:
            logger.warning("No callback registered for edit type: %s, skipping.", edit.type)
            return

        fn(block_ids, session_id)