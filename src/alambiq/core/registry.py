"""The registry of capabilities.

A capability is a decorated function. Its request and result types come from its
own signature, so the registry derives the metadata from the signature.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, get_type_hints

FAMILIES = ("characterize", "investigate", "engineer", "measure", "operate")


@dataclasses.dataclass(frozen=True, slots=True)
class Capability:
    """A registered capability and everything the core needs to call it."""

    id: str
    version: str
    family: str
    summary: str
    run: Callable[..., Any]
    request_type: type
    result_type: type
    environment: str | None = None
    environment_for: Callable[[Any], str] | None = None
    resources: tuple[str, ...] = ()
    resources_for: Callable[[Any], tuple[str, ...]] | None = None
    presets: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    render: Callable[[Any], str] | None = None

    def environment_used(self, request: Any = None) -> str | None:
        """The environment this request runs in: the declared one, unless it picks."""
        if self.environment_for is None or request is None:
            return self.environment
        return self.environment_for(request)

    def resources_used(self, request: Any) -> tuple[str, ...]:
        """The pinned resources this request reads: all of them, unless it narrows."""
        if self.resources_for is None:
            return self.resources
        return tuple(self.resources_for(request))

    def preset(self, name: str | None) -> dict[str, Any]:
        if name is None:
            return {}
        if name not in self.presets:
            known = ", ".join(sorted(self.presets)) or "none"
            raise KeyError(
                f"{self.id} has no preset {name!r}; the presets are: {known}"
            )
        return dict(self.presets[name])

    @property
    def doc(self) -> str:
        """This capability's documentation, read from its own code.

        Two docstrings, in the order a reader needs them: the module explains
        the method, its assumptions and the conditions under which its results hold;
        the function says what this call does. Both sit on the code.
        """
        import sys

        module = sys.modules.get(self.run.__module__)
        about = ((module.__doc__ or "") if module else "").strip()
        does = (self.run.__doc__ or "").strip()
        if does and does not in about:
            return f"{about}\n\n{does}".strip()
        return about


_REGISTRY: dict[str, Capability] = {}


def capability(
    id: str,
    *,
    version: str = "1.0",
    family: str,
    summary: str,
    environment: str | None = None,
    environment_for: Callable[[Any], str] | None = None,
    resources: tuple[str, ...] = (),
    resources_for: Callable[[Any], tuple[str, ...]] | None = None,
    presets: dict[str, dict[str, Any]] | None = None,
    render: Callable[[Any], str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as a capability.

    The function takes `(request, ctx)` and returns a result dataclass. Both
    types are read from its annotations.

    `environment` names the engine environment it runs work in, and
    `resources` the pinned models or databases it reads. Both become part of
    what a saved result depends on, so upgrading an engine or changing a model
    makes earlier results stale rather than silently reusable. A capability
    that offers several models lists them all in `resources` and names, in
    `resources_for`, the ones a given request reads, so that changing one model
    makes stale only the results that used it. A capability whose engine comes
    in variants (the same model built for different hardware) names the default
    in `environment` and, in `environment_for`, the one a given request runs in.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {', '.join(FAMILIES)}")

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        hints = get_type_hints(function)
        try:
            request_type = hints["request"]
        except KeyError:
            raise TypeError(
                f"{function.__name__} needs an annotated 'request' parameter"
            ) from None
        result_type = hints.get("return")
        if result_type is None:
            raise TypeError(f"{function.__name__} needs an annotated return type")
        for name, kind in (("request", request_type), ("result", result_type)):
            if not dataclasses.is_dataclass(kind):
                raise TypeError(f"{function.__name__}'s {name} must be a dataclass")
        for name in resources:
            from alambiq.core import resources as pinned

            pinned.spec(name)  # fail at import if a declared resource is not pinned
        if id in _REGISTRY:
            raise ValueError(
                f"Capability {id!r} is already registered by "
                f"{_REGISTRY[id].run.__module__}"
            )
        _REGISTRY[id] = Capability(
            id=id,
            version=version,
            family=family,
            summary=summary,
            run=function,
            request_type=request_type,
            result_type=result_type,
            environment=environment,
            environment_for=environment_for,
            resources=tuple(resources),
            resources_for=resources_for,
            presets=presets or {},
            render=render,
        )
        return function

    return decorate


def get(id: str) -> Capability:
    try:
        return _REGISTRY[id]
    except KeyError:
        raise KeyError(
            f"No capability {id!r}. Run `alambiq capability list` to see them."
        ) from None


def registered() -> tuple[Capability, ...]:
    return tuple(_REGISTRY[k] for k in sorted(_REGISTRY))


_LOADED: set[str] = set()


def mark_loaded(module: str) -> None:
    _LOADED.add(module)


def is_loaded(module: str) -> bool:
    return module in _LOADED


def clear() -> None:
    """Empty the registry, including the record of what has been imported.

    Clearing without forgetting the imports would leave the registry
    permanently empty, because importing an already-imported module does not
    run its decorators again.
    """
    _REGISTRY.clear()
    _LOADED.clear()


@contextmanager
def isolated() -> Iterator[None]:
    """Register throwaway capabilities without disturbing the real ones.

    Tests that need their own capabilities use this rather than clearing the
    registry, so one test file cannot empty the registry another relies on.
    """
    saved_registry = dict(_REGISTRY)
    saved_loaded = set(_LOADED)
    _REGISTRY.clear()
    _LOADED.clear()
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(saved_registry)
        _LOADED.clear()
        _LOADED.update(saved_loaded)
