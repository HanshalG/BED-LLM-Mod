"""Compatibility facade for the location-finding environment.

The implementation now lives under :mod:`environments.location_finding`.  This
top-level module intentionally re-exports the legacy public surface so existing
imports such as ``from location_finding import run_location_finding`` keep
working while the internals are split into focused modules.
"""

from environments.location_finding import runner as _runner

globals().update(
    {
        name: value
        for name, value in vars(_runner).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__all__ = [
    name
    for name in globals()
    if not (name.startswith("__") and name.endswith("__")) and name != "_runner"
]
