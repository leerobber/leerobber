import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine


class EventBus:
    """Async event bus for decoupled inter-module communication."""

    def __init__(self):
        self._handlers: dict[str, list[Callable[..., Coroutine]]] = defaultdict(list)

    def on(self, event: str, handler: Callable[..., Coroutine]) -> None:
        self._handlers[event].append(handler)

    async def emit(self, event: str, data: Any = None) -> None:
        tasks = [handler(data) for handler in self._handlers.get(event, [])]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


event_bus = EventBus()
