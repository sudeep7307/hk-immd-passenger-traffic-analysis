"""
Data Science Project - HK Immigration Traffic Analysis
Author: Luk Ka Chun
Course: DIT 5412 Data Science, THEi FDE
Semester: 1 AY2025/26
"""

from __future__ import annotations
import logging
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _get_version
from typing import Any

logger = logging.getLogger(__name__)

# Try to get installed package version, fall back to hardcoded
try:
    __version__ = _get_version("hk-immd-passenger-traffic-analysis")
except PackageNotFoundError:
    __version__ = "1.0.0"

__author__ = "Luk Ka Chun"
__email__ = "solarlaziers_ultra@outlook.com"

# Public API names (strings so we can lazy-load on attribute access)
__all__ = (
    "DataPreprocessor",
    "TrafficModels",
    "TrafficVisualizer",
    "create_project_structure",
    "save_results",
    "load_config",
)

# Map public names to their modules for lazy loading
_import_map = {
    "DataPreprocessor": "data_preprocessing",
    "TrafficModels": "models",
    "TrafficVisualizer": "visualization",
    "create_project_structure": "utils",
    "save_results": "utils",
    "load_config": "utils",
}


def __getattr__(name: str) -> Any:
    """Lazy import of public attributes to avoid heavy imports/circular deps."""
    if name in _import_map:
        module = import_module(f".{_import_map[name]}", __package__)
        try:
            return getattr(module, name)
        except AttributeError as exc:
            raise AttributeError(f"module {module.__name__!r} has no attribute {name!r}") from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))


# Non-intrusive initialization log (no prints on import)
logger.debug("HK Immigration Traffic Analysis package initialized (v%s)", __version__)