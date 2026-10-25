"""What every capability must satisfy, enforced for all of them.

These checks exist because the same few mistakes recurred while capabilities
were being written: an option declared but never run, a limit left unstated, a
result that could not be rebuilt from what was saved. They walk the registry,
so a new capability is held to the rules the moment it is registered, and a
failure names the capability and the rule.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import sys
import types
import typing
from pathlib import Path

import pytest

from alambiq import api, capabilities
from alambiq.core import envs, registry

TESTS = Path(__file__).parent
SOURCE = Path(__file__).resolve().parents[2] / "src" / "alambiq" / "capabilities"

# A field may be exempted from the "a test runs it" rule only with a reason.
# This is meant to stay empty.
NOT_EXERCISED: dict[tuple[str, str], str] = {}

capabilities.load()
ALL = sorted(registry.registered(), key=lambda c: c.id)
IDS = [c.id for c in ALL]


def fields_of(cls: type) -> list[dataclasses.Field]:
    return list(dataclasses.fields(cls))


def nested_dataclasses(annotation, seen: set[type] | None = None) -> list[type]:
    """Dataclasses reachable through a type annotation, such as list[Part]."""
    seen = seen if seen is not None else set()
    found: list[type] = []
    for argument in [annotation, *typing.get_args(annotation)]:
        for inner in typing.get_args(argument) or ():
            found += nested_dataclasses(inner, seen)
        if dataclasses.is_dataclass(argument) and isinstance(argument, type):
            if argument not in seen:
                seen.add(argument)
                found.append(argument)
                hints = typing.get_type_hints(argument)
                for field in fields_of(argument):
                    found += nested_dataclasses(hints[field.name], seen)
    return found


def documented_names(cls: type) -> set[str]:
    """Field names a class docstring describes, as `name:` or `a, b:` lines."""
    names: set[str] = set()
    for line in (cls.__doc__ or "").splitlines():
        match = re.match(r"^\s*([A-Za-z_][\w, ]*):\s*$", line)
        if match:
            names |= {name.strip() for name in match.group(1).split(",")}
    return names


def has_own_docstring(cls: type) -> bool:
    """A real docstring, not the signature a dataclass generates for itself."""
    doc = (cls.__doc__ or "").strip()
    return bool(doc) and not doc.startswith(f"{cls.__name__}(")


SUPPORTED_SCALARS = {bool, int, float, str, Path, typing.Any}


def rebuildable(annotation) -> bool:
    """Whether the loader can rebuild a value of this type from plain data."""
    if annotation in SUPPORTED_SCALARS or annotation is type(None):
        return True
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin is typing.Literal:
        return True
    if origin in (typing.Union, types.UnionType):
        return all(rebuildable(a) for a in arguments)
    if origin in (list, tuple):
        return all(rebuildable(a) for a in arguments if a is not Ellipsis)
    if origin is dict:
        return all(rebuildable(a) for a in arguments)
    if dataclasses.is_dataclass(annotation):
        hints = typing.get_type_hints(annotation)
        return all(rebuildable(hints[f.name]) for f in fields_of(annotation))
    return False


def test_there_are_capabilities_to_check():
    assert len(ALL) >= 15


def test_every_module_that_defines_a_capability_is_registered():
    listed = set(capabilities.MODULES)
    defined = set()
    for path in SOURCE.rglob("*.py"):
        if "@capability(" in path.read_text():
            relative = path.with_suffix("").relative_to(SOURCE.parent.parent)
            defined.add(".".join(relative.parts))
    assert defined == listed, {
        "defined but not listed": sorted(defined - listed),
        "listed but not defined": sorted(listed - defined),
    }


@pytest.mark.parametrize("capability", ALL, ids=IDS)
class TestEveryCapability:
    def test_it_has_a_summary_that_fits_on_one_line(self, capability):
        assert capability.summary.strip() == capability.summary
        assert 10 <= len(capability.summary) <= 90, capability.summary
        assert "\n" not in capability.summary

    def test_its_version_looks_like_a_version(self, capability):
        assert re.fullmatch(r"\d+\.\d+(\.\d+)?", capability.version), capability.version

    def test_its_module_explains_the_method_and_states_its_limits(self, capability):
        module = sys.modules[capability.run.__module__]
        text = (module.__doc__ or "").strip()
        assert len(text.split()) >= 40, (
            "the module docstring is too short to explain a method"
        )
        assert "limit" in text.lower(), "the module docstring never states a limit"

    def test_its_function_says_what_a_call_does(self, capability):
        assert (capability.run.__doc__ or "").strip()

    def test_what_it_accepts_and_what_it_says_reach_the_command_line(self, capability):
        shown = api.describe(capability.id)
        assert shown["id"] == capability.id and shown["summary"] == capability.summary
        assert {f["name"] for f in shown["request"]} == {
            f.name for f in fields_of(capability.request_type)
        }
        assert {f["name"] for f in shown["result"]} == {
            f.name for f in fields_of(capability.result_type)
        }
        # The method and its limits are what `capability show` prints.
        assert "limit" in shown["documentation"].lower()
        assert shown["resources"] == list(capability.resources)

    def test_it_can_be_listed(self, capability):
        assert capability.id in {row["id"] for row in api.capability_list()}

    def test_it_can_present_its_own_result(self, capability):
        assert callable(capability.render)

    def test_what_its_report_reads_from_the_result_exists(self, capability):
        source = inspect.getsourcefile(capability.render)
        # A report module can serve several capabilities (a comparison and its ensemble).
        sharing = [
            c
            for c in registry.registered()
            if inspect.getsourcefile(c.render) == source
        ]
        known = {
            name
            for c in sharing
            for name in dir(c.result_type)
            if not name.startswith("_")
        } | {f.name for c in sharing for f in fields_of(c.result_type)}
        read = set(re.findall(r"\bresult\.([A-Za-z_]\w*)", Path(source).read_text()))
        assert read <= known, f"the report reads missing fields: {sorted(read - known)}"

    def test_every_request_field_is_described(self, capability):
        request = capability.request_type
        missing = [
            f.name
            for f in fields_of(request)
            if f.name not in documented_names(request)
        ]
        assert not missing, f"not described in the request docstring: {missing}"

    def test_every_type_inside_the_request_is_described(self, capability):
        hints = typing.get_type_hints(capability.request_type)
        for field in fields_of(capability.request_type):
            for nested in nested_dataclasses(hints[field.name]):
                assert has_own_docstring(nested), f"{nested.__name__} has no docstring"

    def test_its_result_is_described(self, capability):
        assert has_own_docstring(capability.result_type)

    def test_its_result_can_be_rebuilt_from_what_was_saved(self, capability):
        """Reuse rebuilds the result from saved outputs; every type must allow it."""
        assert rebuildable(capability.result_type), (
            f"{capability.result_type.__name__} uses a type the loader cannot rebuild"
        )

    def test_its_presets_name_real_fields(self, capability):
        names = {f.name for f in fields_of(capability.request_type)}
        for preset, values in capability.presets.items():
            assert set(values) <= names, (preset, sorted(set(values) - names))

    def test_a_declared_environment_exists(self, capability):
        if capability.environment:
            assert capability.environment in envs.ENVIRONMENTS

    def test_every_request_field_is_run_by_a_test(self, capability):
        """A field declared but never run can be wrong without anything noticing."""
        texts = [
            path.read_text()
            for path in TESTS.glob("test_*.py")
            if path.name != Path(__file__).name
        ]
        about = [t for t in texts if capability.id in t]
        unrun = []
        for field in fields_of(capability.request_type):
            if (capability.id, field.name) in NOT_EXERCISED:
                continue
            pattern = re.compile(rf"""["']{field.name}["']\s*:|\b{field.name}\s*=""")
            if not any(pattern.search(t) for t in about):
                unrun.append(field.name)
        assert not unrun, f"no test sets these fields: {unrun}"


def test_exemptions_from_the_rule_that_a_test_runs_every_field_are_justified():
    for key, reason in NOT_EXERCISED.items():
        assert len(reason.split()) >= 5, f"{key} is exempted without a reason"
