from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol


Event = Dict[str, Any]
EventHandler = Callable[[Event], None]


class EventBus(Protocol):
    def publish(self, topic: str, event: Event) -> None:
        ...

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        ...

    def ack(self, event_id: str) -> None:
        ...

    def nack(self, event_id: str, reason: str) -> None:
        ...

    def health(self) -> Dict[str, Any]:
        ...


@dataclass
class InMemoryEventBus:
    subscribers: Dict[str, List[EventHandler]]

    def __init__(self) -> None:
        self.subscribers = {}

    def publish(self, topic: str, event: Event) -> None:
        for handler in self.subscribers.get(topic, []):
            handler(event)

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        self.subscribers.setdefault(topic, []).append(handler)

    def ack(self, event_id: str) -> None:
        return

    def nack(self, event_id: str, reason: str) -> None:
        return

    def health(self) -> Dict[str, Any]:
        return {"ok": True, "backend": "in_memory"}


class RedisEventBus:
    """Placeholder adapter. Wire with redis streams in next iteration."""

    def publish(self, topic: str, event: Event) -> None:
        raise NotImplementedError

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        raise NotImplementedError

    def ack(self, event_id: str) -> None:
        raise NotImplementedError

    def nack(self, event_id: str, reason: str) -> None:
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        return {"ok": False, "backend": "redis", "ready": False}
