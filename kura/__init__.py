"""Kura - Conversation Analysis Library"""

import importlib
from typing import Any

# List of submodules that can be lazily loaded
_submodules = [
    "checkpoint",
    "checkpoints",
    "summarisation",
    "cluster",
    "dimensionality",
    "meta_cluster",
    "types",
    "k_means",
    "hdbscan",
    "visualization",
]


def __getattr__(name: str) -> Any:
    """Lazily import submodules and their contents."""
    if name in globals():
        return globals()[name]

    if name in _submodules:
        # Import the submodule
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
