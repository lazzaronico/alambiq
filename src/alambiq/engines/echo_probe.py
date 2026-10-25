"""A probe for the engine boundary itself.

The module contains no science, so that a failure of the engine boundary is
distinguishable from a failure of an engine. It imports the standard library
only, like every module in this package.
"""

from __future__ import annotations

import shutil
from typing import Any


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Return the request with a marker, confirming both directions work."""
    return {**request, "seen": True}


def where(request: dict[str, Any]) -> dict[str, Any]:
    """Report where each named tool is found from inside the environment."""
    return {name: shutil.which(name) for name in request["tools"]}


def fail(request: dict[str, Any]) -> dict[str, Any]:
    """Raise, so the launcher's error reporting can be checked."""
    raise RuntimeError("deliberate engine failure")
