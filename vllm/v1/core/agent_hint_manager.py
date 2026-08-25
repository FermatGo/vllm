# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extension point for agent-provided KV-cache lifecycle hints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        FreeKVCacheBlockQueue,
        KVCacheBlock,
    )
    from vllm.v1.engine import AgentHintResponse
    from vllm.v1.request import Request


@dataclass(frozen=True)
class AgentHintManagerContext:
    """Dependencies exposed to an out-of-tree agent-hint implementation."""

    vllm_config: VllmConfig
    kv_cache_manager: KVCacheManager
    connector: KVConnectorBase_V1 | None
    add_request: Callable[[Request], None]
    block_size: int
    hash_block_size: int


class AgentHintManager:
    """No-op base class for scheduler-side agent-hint implementations.

    Hardware plugins may subclass this interface and provide it through an
    `AgentHintBackend`. vLLM owns request transport and calls
    these lifecycle hooks, while plugins own all session and cache policy.
    """

    def is_kvc_management_request(self, request: Request) -> bool:
        return False

    def register_kvc_management_request(self, request: Request) -> AgentHintResponse | None:
        return None

    def on_request_added(self, request: Request) -> None:
        pass

    def on_request_scheduled(self, request: Request) -> None:
        pass

    def on_step(self, num_unfinished_requests: int) -> None:
        pass

    def on_request_finished(self, request: Request) -> None:
        pass

    def has_pending_work(self) -> bool:
        return False

    def shutdown(self) -> None:
        pass


class AgentHintBackend(Protocol):
    """Plugin backend providing all Agent Hint runtime components."""

    def is_supported(self) -> bool: ...

    def create_manager(self, context: AgentHintManagerContext) -> AgentHintManager: ...

    def create_free_kv_cache_block_queue(
        self,
        blocks: list[KVCacheBlock],
    ) -> FreeKVCacheBlockQueue: ...


_BACKENDS: dict[str, AgentHintBackend] = {}


def register_agent_hint_backend(name: str, backend: AgentHintBackend) -> None:
    """Register an out-of-tree Agent Hint backend."""
    existing = _BACKENDS.get(name)
    if existing is not None and existing is not backend:
        raise ValueError(f"Agent Hint backend {name!r} is already registered")
    _BACKENDS[name] = backend


def get_agent_hint_backend() -> AgentHintBackend | None:
    """Return the single backend supporting the active platform."""
    backends = [backend for backend in _BACKENDS.values() if backend.is_supported()]
    if not backends:
        return None
    if len(backends) > 1:
        raise RuntimeError("Multiple Agent Hint backends support the active platform")
    return backends[0]


def create_agent_hint_manager(
    context: AgentHintManagerContext,
) -> AgentHintManager:
    """Create the Agent Hint manager for the active backend."""
    backend = get_agent_hint_backend()
    if backend is None:
        return AgentHintManager()
    return backend.create_manager(context)
