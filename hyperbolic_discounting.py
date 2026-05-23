"""Compatibility facade for hyperbolic temporal discounting."""

from environments.hyperbolic_discounting import runner as _runner

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
