"""Optional process-local hook for observing communication calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollectiveEvent:
    category: str
    operation: str
    numel: int
    dtype: str
    bytes: int

    def as_tuple(self):
        return (self.category, self.operation, self.numel, self.dtype, self.bytes)


class CollectiveObserver:
    def __init__(self): self.events = []
    def record(self, category, operation, numel, dtype, bytes):
        self.events.append(CollectiveEvent(category, operation, int(numel), str(dtype).replace("torch.", ""), int(bytes)))
    def signature(self): return [event.as_tuple() for event in self.events]


_active_observer = None


def set_active_observer(observer):
    global _active_observer
    _active_observer = observer


def observe_collective(category, operation, tensor, bytes=None):
    if _active_observer is not None:
        payload = int(tensor.numel() * tensor.element_size()) if bytes is None else int(bytes)
        _active_observer.record(category, operation, tensor.numel(), tensor.dtype, payload)


def signatures_agree(signatures):
    return bool(signatures) and all(signature == signatures[0] for signature in signatures[1:])
