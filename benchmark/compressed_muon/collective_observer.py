"""Public benchmark aliases for the optional collective observer."""

from dion.collective_observer import (CollectiveEvent, CollectiveObserver,
                                      observe_collective, set_active_observer,
                                      signatures_agree)

__all__ = ["CollectiveEvent", "CollectiveObserver", "observe_collective",
           "set_active_observer", "signatures_agree"]
