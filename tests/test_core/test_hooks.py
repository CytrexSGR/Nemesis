"""Tests for the Hook System."""

from __future__ import annotations

import logging

import pytest

from nemesis.core.hooks import HookContext, HookEvent, HookManager

# ---------------------------------------------------------------------------
# TestHookEvent
# ---------------------------------------------------------------------------


class TestHookEvent:
    """Tests for HookEvent enum."""

    def test_all_events_defined(self) -> None:
        """All 8 lifecycle events must be defined."""
        assert len(HookEvent) == 8

    def test_event_values(self) -> None:
        """Each event's .value matches the expected snake_case string."""
        expected = {
            HookEvent.PRE_INDEX: "pre_index",
            HookEvent.POST_INDEX: "post_index",
            HookEvent.PRE_REINDEX: "pre_reindex",
            HookEvent.POST_REINDEX: "post_reindex",
            HookEvent.ON_FILE_CHANGE: "on_file_change",
            HookEvent.ON_ERROR: "on_error",
            HookEvent.PRE_QUERY: "pre_query",
            HookEvent.POST_QUERY: "post_query",
        }
        for event, value in expected.items():
            assert event.value == value


# ---------------------------------------------------------------------------
# TestHookContext
# ---------------------------------------------------------------------------


class TestHookContext:
    """Tests for HookContext dataclass."""

    def test_context_creation(self) -> None:
        """HookContext stores event and data correctly."""
        data = {"path": "/tmp/test.py"}
        ctx = HookContext(event=HookEvent.PRE_INDEX, data=data)
        assert ctx.event is HookEvent.PRE_INDEX
        assert ctx.data == {"path": "/tmp/test.py"}

    def test_context_default_data(self) -> None:
        """Default data is an empty dict, not shared across instances."""
        ctx1 = HookContext(event=HookEvent.ON_ERROR)
        ctx2 = HookContext(event=HookEvent.ON_ERROR)
        assert ctx1.data == {}
        assert ctx2.data == {}
        # Ensure distinct instances (not a shared mutable default)
        ctx1.data["x"] = 1
        assert "x" not in ctx2.data


# ---------------------------------------------------------------------------
# TestHookManager
# ---------------------------------------------------------------------------


class TestHookManager:
    """Tests for HookManager."""

    def test_register_and_emit(self) -> None:
        """A registered callback is invoked when the event is emitted."""
        manager = HookManager()
        calls: list[HookContext] = []
        manager.register(HookEvent.PRE_INDEX, calls.append)
        manager.emit(HookEvent.PRE_INDEX)
        assert len(calls) == 1

    def test_multiple_callbacks(self) -> None:
        """Multiple callbacks on the same event are all invoked."""
        manager = HookManager()
        results: list[str] = []
        manager.register(HookEvent.POST_INDEX, lambda ctx: results.append("a"))
        manager.register(HookEvent.POST_INDEX, lambda ctx: results.append("b"))
        manager.emit(HookEvent.POST_INDEX)
        assert results == ["a", "b"]

    def test_callback_order(self) -> None:
        """Callbacks are invoked in the order they were registered."""
        manager = HookManager()
        order: list[int] = []
        for i in range(5):
            manager.register(
                HookEvent.ON_FILE_CHANGE, lambda ctx, n=i: order.append(n)
            )
        manager.emit(HookEvent.ON_FILE_CHANGE)
        assert order == [0, 1, 2, 3, 4]

    def test_unregister(self) -> None:
        """After unregister, the callback is no longer invoked."""
        manager = HookManager()
        calls: list[HookContext] = []
        manager.register(HookEvent.PRE_QUERY, calls.append)
        manager.unregister(HookEvent.PRE_QUERY, calls.append)
        manager.emit(HookEvent.PRE_QUERY)
        assert calls == []

    def test_unregister_unknown_raises(self) -> None:
        """Unregistering a callback that was never registered raises ValueError."""
        manager = HookManager()
        with pytest.raises(ValueError):
            manager.unregister(HookEvent.ON_ERROR, lambda ctx: None)

    def test_emit_passes_context(self) -> None:
        """The callback receives a HookContext with correct event and data."""
        manager = HookManager()
        received: list[HookContext] = []
        manager.register(HookEvent.POST_QUERY, received.append)
        manager.emit(HookEvent.POST_QUERY, data={"query": "test", "score": 0.95})
        assert len(received) == 1
        ctx = received[0]
        assert ctx.event is HookEvent.POST_QUERY
        assert ctx.data == {"query": "test", "score": 0.95}

    def test_emit_error_does_not_propagate(self, caplog: pytest.LogCaptureFixture) -> None:
        """An exception in one callback is logged, and subsequent callbacks still run."""
        manager = HookManager()
        results: list[str] = []

        def failing_hook(ctx: HookContext) -> None:
            raise RuntimeError("boom")

        manager.register(HookEvent.ON_ERROR, failing_hook)
        manager.register(HookEvent.ON_ERROR, lambda ctx: results.append("ok"))

        with caplog.at_level(logging.ERROR, logger="nemesis.core.hooks"):
            manager.emit(HookEvent.ON_ERROR)

        # The second callback must have run despite the first one failing
        assert results == ["ok"]
        # The error must have been logged
        assert "failing_hook" in caplog.text
        assert "on_error" in caplog.text

    def test_clear_single_event(self) -> None:
        """clear(event) removes hooks only for that event."""
        manager = HookManager()
        manager.register(HookEvent.PRE_INDEX, lambda ctx: None)
        manager.register(HookEvent.POST_INDEX, lambda ctx: None)
        manager.clear(HookEvent.PRE_INDEX)
        assert manager.get_hooks(HookEvent.PRE_INDEX) == []
        assert len(manager.get_hooks(HookEvent.POST_INDEX)) == 1

    def test_clear_all(self) -> None:
        """clear() with no argument removes all hooks across all events."""
        manager = HookManager()
        for event in HookEvent:
            manager.register(event, lambda ctx: None)
        assert manager.hook_count == len(HookEvent)
        manager.clear()
        assert manager.hook_count == 0

    def test_get_hooks(self) -> None:
        """get_hooks returns a copy of registered callbacks."""
        manager = HookManager()
        cb = lambda ctx: None  # noqa: E731
        manager.register(HookEvent.PRE_REINDEX, cb)
        hooks = manager.get_hooks(HookEvent.PRE_REINDEX)
        assert hooks == [cb]
        # Mutating the returned list must not affect internal state
        hooks.clear()
        assert len(manager.get_hooks(HookEvent.PRE_REINDEX)) == 1

    def test_hook_count(self) -> None:
        """hook_count reflects the total number of registered hooks."""
        manager = HookManager()
        assert manager.hook_count == 0
        manager.register(HookEvent.PRE_INDEX, lambda ctx: None)
        manager.register(HookEvent.POST_INDEX, lambda ctx: None)
        manager.register(HookEvent.PRE_INDEX, lambda ctx: None)
        assert manager.hook_count == 3

    def test_emit_without_data(self) -> None:
        """Emitting without data passes an empty dict in the context."""
        manager = HookManager()
        received: list[HookContext] = []
        manager.register(HookEvent.PRE_INDEX, received.append)
        manager.emit(HookEvent.PRE_INDEX)
        assert received[0].data == {}
