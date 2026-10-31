"""Turn plain data into a capability's request dataclass.

This module handles all input. There is no per-capability validation: the
dataclass field types define what is accepted, and each error names the step
and the field at fault.
"""

from __future__ import annotations

import dataclasses
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


class RequestError(ValueError):
    """An input a capability cannot accept, named by where it came from."""


def _fail(where: str, message: str) -> None:
    raise RequestError(f"{where}: {message}")


def _found(value: Any) -> str:
    """The value as an error message states it, shortened when it is long."""
    text = repr(value)
    text = text if len(text) <= 60 else text[:57] + "..."
    kind = {list: "a list", tuple: "a list", dict: "a mapping"}.get(type(value))
    return f"{kind} {text}" if kind else text


def _single(expected: str, value: Any) -> str:
    """What a field wants, with a reminder that it is one value when a list was given."""
    if isinstance(value, (list, tuple)):
        return f"{expected} (one value, not a list)"
    return expected


def _optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap `X | None`, reporting whether None was allowed."""
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
        return annotation, type(None) in get_args(annotation)
    return annotation, False


def coerce(value: Any, annotation: Any, where: str) -> Any:
    """Convert one value to its annotated type, or explain why it cannot be."""
    annotation, allows_none = _optional(annotation)
    if value is None:
        if allows_none:
            return None
        _fail(where, "found null; a value is required")

    origin = get_origin(annotation)

    if origin is Literal:
        allowed = get_args(annotation)
        if value not in allowed:
            _fail(
                where,
                f"found {_found(value)}; expected one of {', '.join(map(repr, allowed))}",
            )
        return value

    if annotation is Path:
        if not isinstance(value, (str, Path)):
            _fail(where, f"found {_found(value)}; expected {_single('a path', value)}")
        return Path(value)

    if annotation is bool:
        if not isinstance(value, bool):
            _fail(where, f"found {_found(value)}; expected true or false")
        return value

    if annotation is int:
        # bool is an int in Python; a flag is not a count.
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(where, f"found {_found(value)}; expected a whole number")
        return value

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(where, f"found {_found(value)}; expected a number")
        return float(value)

    if annotation is str:
        if not isinstance(value, str):
            _fail(where, f"found {_found(value)}; expected {_single('text', value)}")
        return value

    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            _fail(where, f"found {_found(value)}; expected a list")
        (item_type,) = get_args(annotation) or (Any,)
        items = [
            coerce(item, item_type, f"{where}[{i}]") for i, item in enumerate(value)
        ]
        return tuple(items) if origin is tuple else items

    if origin is dict:
        if not isinstance(value, dict):
            _fail(where, f"found {_found(value)}; expected a mapping")
        key_type, value_type = get_args(annotation) or (Any, Any)
        return {
            coerce(k, key_type, f"{where} key"): coerce(v, value_type, f"{where}.{k}")
            for k, v in value.items()
        }

    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, dict):
            _fail(
                where,
                f"found {_found(value)}; expected a mapping of {annotation.__name__} fields",
            )
        return build(annotation, value, where)

    return value


def build(
    request_type: type,
    data: dict[str, Any],
    where: str,
    deferred: frozenset[str] = frozenset(),
) -> Any:
    """Build a request dataclass from a mapping, naming any bad field.

    `deferred` names fields that another step will supply: they are neither required nor
    checked here, and with any of them the request is checked but not built (None is
    returned), which is how a campaign is validated before anything has run.
    """
    if not dataclasses.is_dataclass(request_type):
        raise TypeError(f"{request_type!r} is not a request dataclass")
    if not isinstance(data, dict):
        _fail(where, f"found {_found(data)}; expected a mapping of fields")

    hints = get_type_hints(request_type)
    fields = {f.name: f for f in dataclasses.fields(request_type)}

    unknown = sorted(set(data) - set(fields))
    if unknown:
        known = ", ".join(sorted(fields))
        _fail(
            where,
            f"unknown field{'s' if len(unknown) > 1 else ''} "
            f"{', '.join(unknown)}; this capability accepts {known}",
        )

    values: dict[str, Any] = {}
    for name, spec in fields.items():
        if name in deferred:
            continue
        if name in data:
            values[name] = coerce(data[name], hints[name], f"{where}.{name}")
        elif spec.default is not dataclasses.MISSING:
            values[name] = spec.default
        elif spec.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            values[name] = spec.default_factory()  # type: ignore[misc]
        else:
            _fail(where, f"{name} is required")
    if deferred & set(fields):
        return None
    return request_type(**values)


def as_data(value: Any) -> Any:
    """Render a result as plain data for storage and reports."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: as_data(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {k: as_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_data(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
