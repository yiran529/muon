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


def get_active_observer():
    return _active_observer


def observe_collective(category, operation, tensor, bytes=None, numel=None):
    if _active_observer is not None:
        payload = int(tensor.numel() * tensor.element_size()) if bytes is None else int(bytes)
        logical_numel = tensor.numel() if numel is None else int(numel)
        _active_observer.record(category, operation, logical_numel, tensor.dtype, payload)


def signatures_agree(signatures):
    return bool(signatures) and all(signature == signatures[0] for signature in signatures[1:])


def aggregate_observed(observer):
    """Aggregate observed logical payloads without mixing them with durations."""
    totals = {}
    for event in observer.events:
        key = (event.category, event.operation)
        item = totals.setdefault(key, {"category": event.category, "operation": event.operation,
                                       "numel": 0, "dtype": event.dtype,
                                       "bytes": 0, "count": 0})
        item["numel"] += event.numel
        item["bytes"] += event.bytes
        item["count"] += 1
    return list(totals.values())
