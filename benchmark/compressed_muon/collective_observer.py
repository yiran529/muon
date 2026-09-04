"""Public benchmark aliases for the optional collective observer."""

from dion.collective_observer import (CollectiveEvent, CollectiveObserver,
                                      aggregate_observed,
                                      get_active_observer,
                                      observe_collective, set_active_observer,
                                      signatures_agree)

__all__ = ["CollectiveEvent", "CollectiveObserver", "aggregate_observed", "get_active_observer", "observe_collective",
           "set_active_observer", "signatures_agree"]
