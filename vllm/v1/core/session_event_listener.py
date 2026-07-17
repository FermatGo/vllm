from typing import Protocol
from vllm.v1.core.kv_cache_utils import BlockHash


class SessionEventListener(Protocol):
    """SPM 实现此协议，监听 SAM 的 session 生命周期事件"""

    def on_session_registered(
        self, session_id: str, parent_session_id: str | None
    ) -> None: ...

    def on_session_blocks_protected(
        self,
        block_hashes: list[BlockHash],
    ) -> None: ...

    def on_session_blocks_removed(
        self,
        block_hashes: list[BlockHash],
    ) -> None: ...

    def on_session_cache_hit(
        self,
        block_hashes: list[BlockHash],
    ) -> None: ...

    def on_session_ttl_expired(self, block_hashs: list[BlockHash]) -> None: ...

    def on_session_freed(self, session_id: str) -> None: ...

    def on_context_management_evict(
        self,
        block_hashes: list[BlockHash],
    ) -> None: ...

    def on_context_management_offload(
        self, session_id: str, block_ids: list[int]
    ) -> None: ...

    def on_context_management_prefetch(
        self,
        session_id: str,
        block_hashes: list[BlockHash],
    ) -> bool: ...

    def on_check_matched_token(
        self,
        session_id: str,
        check_matched_start: int,
        check_matched_end: int,
    ) -> int: ...