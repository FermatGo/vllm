from typing import Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
import time

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


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
        self._on_expired = on_expired #TODO：SAM传入对应回调函数
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
        #TODO: 联调时确保，对于某个block_id的不同操作，记录的key值是一致的
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