"""Alambiq: capabilities and campaigns for molecular investigations.

The execution stack is imported when a name in `alambiq.api` is first used, so a
numerical module can be imported and tested without Prefect or a scientific engine.
The science of a capability is therefore written in plain functions.

The entry points are `alambiq.api` and `alambiq.cli`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__version__ = "0.1.1"


def _prefect_home() -> None:
    """Separate the state of Prefect from the state of any other Prefect installation.

    Prefect stores its database under `PREFECT_HOME`, `~/.prefect` by default, and a
    database migrated by one version of Prefect cannot be opened by an older one. A
    researcher who uses Prefect for another purpose, or who has two Alambiq installations
    with different Prefect versions, would otherwise break one installation with the other.
    Unless `PREFECT_HOME` is set, the state is stored in `ALAMBIQ_PREFECT_HOME`, or in a
    directory under `~/.alambiq` named for the installed Prefect version. This function
    runs before Prefect is imported, because only then does it take effect. The directory
    is created with its parents, because Prefect does not create missing parents.
    """
    if "PREFECT_HOME" not in os.environ:
        from importlib import metadata

        try:
            version = metadata.version("prefect")
        except metadata.PackageNotFoundError:
            version = "unknown"
        os.environ["PREFECT_HOME"] = os.environ.get(
            "ALAMBIQ_PREFECT_HOME",
            str(Path.home() / ".alambiq" / f"prefect-{version}"),
        )
    try:
        Path(os.environ["PREFECT_HOME"]).expanduser().mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


_prefect_home()

_API = {"describe", "load_campaign", "run", "step", "validate"}


def __getattr__(name: str) -> Any:
    """Resolve a name re-exported from `alambiq.api` on first use (PEP 562)."""
    if name in _API:
        from alambiq import api

        value = getattr(api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'alambiq' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _API)
