"""SESTRAV public package facade.

The implementation currently lives in the top-level ``src`` and ``functions``
packages (retained for internal import stability). This module re-exports the
stable public surface so that an installed wheel exposes ``import sestrav`` and
so downstream code is insulated from the internal layout.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("sestrav")
except _metadata.PackageNotFoundError:  # editable / source checkout
    __version__ = "0.0.0+local"

# Re-export the most commonly used public entry points. Import lazily-safe
# modules only; heavy optional deps (torch-geometric, streamlit) stay lazy.
from src import features as features  # noqa: F401
from src.cli import main as main  # noqa: F401  (console-script target reuse)

__all__ = ["features", "main", "__version__"]
