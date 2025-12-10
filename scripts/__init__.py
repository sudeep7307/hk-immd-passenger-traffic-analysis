"""
Data Science Project - HK Immigration Traffic Analysis

A comprehensive package for analyzing Hong Kong Immigration Department 
passenger traffic data using machine learning and visualization techniques.

Modules:
--------
- data_preprocessing: Data cleaning, transformation, and feature engineering
- models: Machine learning models (regression, classification, clustering)
- visualization: Comprehensive plotting and visual analysis tools
- utils: Utility functions for project setup and data management

Author: Luk Ka Chun, Liew Wang Yui, Tang Chi To
Course: DIT 5412 Data Science, THEi FDE
Semester: 1 AY2025/26
Version: 1.0.0
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
    "initialize_project",
)

# Map public names to their modules for lazy loading
_import_map = {
    "DataPreprocessor": "data_preprocessing",
    "TrafficModels": "models",
    "TrafficVisualizer": "visualization",
    "create_project_structure": "utils",
    "save_results": "utils",
    "load_config": "utils",
    "initialize_project": "utils",
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


def get_version() -> str:
    """Get the current package version."""
    return __version__


def get_author_info() -> dict:
    """Get author and project information."""
    return {
        "author": __author__,
        "email": __email__,
        "version": __version__,
        "course": "DIT 5412 Data Science",
        "institution": "THEi FDE",
        "semester": "1 AY2025/26"
    }


def quick_start() -> None:
    """Print quick start instructions for the package."""
    info = get_author_info()
    print("=" * 60)
    print(f"HK Immigration Traffic Analysis v{info['version']}")
    print("=" * 60)
    print(f"Author: {info['author']} ({info['email']})")
    print(f"Course: {info['course']}, {info['institution']}")
    print(f"Semester: {info['semester']}")
    print("\nQuick Start:")
    print("1. First, create project structure:")
    print("   >>> from hk_immd_traffic import create_project_structure")
    print("   >>> create_project_structure()")
    print("\n2. Load and preprocess data:")
    print("   >>> from hk_immd_traffic import DataPreprocessor")
    print("   >>> preprocessor = DataPreprocessor('data/raw/passenger_traffic.csv')")
    print("   >>> df = preprocessor.process()")
    print("\n3. Analyze with machine learning:")
    print("   >>> from hk_immd_traffic import TrafficModels")
    print("   >>> models = TrafficModels(df, features=['day_of_week', 'month'], target='total')")
    print("   >>> results = models.run_all_models()")
    print("=" * 60)


# Non-intrusive initialization log (no prints on import)
logger.debug("HK Immigration Traffic Analysis package initialized (v%s)", __version__)