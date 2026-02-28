from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class Event:
    name: str
    payload: Any = None


class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)

    def subscribe(self, name: str, callback: Callable[[Event], None]) -> None:
        self._subs[name].append(callback)

    def emit(self, name: str, payload: Any = None) -> None:
        event = Event(name=name, payload=payload)
        for cb in list(self._subs.get(name, [])):
            cb(event)
