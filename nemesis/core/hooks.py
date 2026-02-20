"""Hook System — event-driven lifecycle hooks for Nemesis operations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HookEvent(Enum):
    """Events that can trigger hooks."""

    PRE_INDEX = "pre_index"
    POST_INDEX = "post_index"
    PRE_REINDEX = "pre_reindex"
    POST_REINDEX = "post_reindex"
    ON_FILE_CHANGE = "on_file_change"
    ON_ERROR = "on_error"
    PRE_QUERY = "pre_query"
    POST_QUERY = "post_query"


@dataclass
class HookContext:
    """Context passed to hook callbacks.

    Attributes:
        event: The event that triggered the hook.
        data: Event-specific data (file path, result, error, etc.).
    """

    event: HookEvent
    data: dict[str, Any] = field(default_factory=dict)


# Type alias for hook callbacks
HookCallback = Callable[[HookContext], None]


class HookManager:
    """Manages lifecycle hooks for Nemesis operations.

    Allows registering callbacks for specific events. Multiple callbacks
    can be registered for the same event. Callbacks are invoked in
    registration order. Errors in callbacks are logged but don't
    propagate (fail-safe).
    """

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookCallback]] = {
            event: [] for event in HookEvent
        }

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        """Register a callback for an event.

        Args:
            event: The event to listen for.
            callback: Function to call when event fires.
        """
        self._hooks[event].append(callback)

    def unregister(self, event: HookEvent, callback: HookCallback) -> None:
        """Remove a previously registered callback.

        Args:
            event: The event the callback was registered for.
            callback: The callback to remove.

        Raises:
            ValueError: If callback was not registered for this event.
        """
        self._hooks[event].remove(callback)

    def emit(self, event: HookEvent, data: dict[str, Any] | None = None) -> None:
        """Emit an event, invoking all registered callbacks.

        Callbacks are invoked in registration order. Errors are logged
        but do not propagate — this ensures one failing hook cannot
        break the indexing pipeline.

        Args:
            event: The event to emit.
            data: Optional event-specific data.
        """
        context = HookContext(event=event, data=data or {})
        for callback in self._hooks[event]:
            try:
                callback(context)
            except Exception:
                logger.exception(
                    "Hook callback %s failed for event %s",
                    getattr(callback, "__name__", repr(callback)),
                    event.value,
                )

    def clear(self, event: HookEvent | None = None) -> None:
        """Remove all hooks for an event, or all hooks if event is None.

        Args:
            event: Specific event to clear, or None for all.
        """
        if event is None:
            for evt in HookEvent:
                self._hooks[evt].clear()
        else:
            self._hooks[event].clear()

    def get_hooks(self, event: HookEvent) -> list[HookCallback]:
        """Get all registered callbacks for an event.

        Args:
            event: The event to query.

        Returns:
            List of registered callbacks (copy).
        """
        return list(self._hooks[event])

    @property
    def hook_count(self) -> int:
        """Total number of registered hooks across all events."""
        return sum(len(cbs) for cbs in self._hooks.values())
