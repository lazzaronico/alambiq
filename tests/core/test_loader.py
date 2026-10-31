"""The loader is the whole of input handling, so it is tested directly."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from alambiq.core.loader import RequestError, as_data, build


@dataclass(frozen=True, slots=True)
class Inner:
    label: str
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class Example:
    name: str
    count: int
    ratio: float = 0.5
    mode: Literal["fast", "careful"] = "fast"
    flag: bool = False
    where: Path | None = None
    tags: list[str] = field(default_factory=list)
    limits: dict[str, float] = field(default_factory=dict)
    inner: Inner | None = None


def make(**data):
    return build(Example, {"name": "n", "count": 1, **data}, "step s")


class TestDefaults:
    def test_defaults_fill_in(self):
        result = make()
        assert (result.ratio, result.mode, result.flag) == (0.5, "fast", False)
        assert result.tags == [] and result.limits == {}

    def test_default_factories_are_not_shared(self):
        first, second = make(), make()
        first.tags.append("x")
        assert second.tags == []

    def test_missing_required_field_names_itself(self):
        with pytest.raises(RequestError, match="count is required"):
            build(Example, {"name": "n"}, "step s")


class TestUnknownFields:
    def test_unknown_field_is_named_with_the_step(self):
        with pytest.raises(RequestError) as caught:
            make(colour="red")
        assert "step s" in str(caught.value)
        assert "unknown field colour" in str(caught.value)

    def test_the_message_lists_what_is_accepted(self):
        with pytest.raises(RequestError, match="this capability accepts"):
            make(colour="red")

    def test_several_unknown_fields_are_all_named(self):
        with pytest.raises(RequestError) as caught:
            make(colour="red", size=2)
        assert "colour, size" in str(caught.value)


class TestCoercion:
    def test_int_accepts_whole_numbers_only(self):
        assert make(count=7).count == 7
        with pytest.raises(RequestError, match="expected a whole number"):
            make(count=1.5)

    def test_a_flag_is_not_a_count(self):
        # bool is a subclass of int in Python; that must not pass here.
        with pytest.raises(RequestError, match="expected a whole number"):
            make(count=True)

    def test_float_accepts_integers(self):
        assert make(ratio=2).ratio == pytest.approx(2.0)

    def test_a_list_given_for_text_or_a_path_is_named_and_one_value_is_asked_for(self):
        with pytest.raises(RequestError) as caught:
            make(name=["protein.fasta"], count=1)
        assert (
            "found a list ['protein.fasta']; expected text (one value, not a list)"
            in str(caught.value)
        )
        with pytest.raises(RequestError, match="expected a path .one value, not a list"):
            make(where=["a.txt"])

    def test_bool_requires_a_boolean(self):
        with pytest.raises(RequestError, match="expected true or false"):
            make(flag="yes")

    def test_literal_reports_the_allowed_values(self):
        assert make(mode="careful").mode == "careful"
        with pytest.raises(RequestError, match="expected one of"):
            make(mode="quick")

    def test_path_accepts_text(self):
        assert make(where="a/b.json").where == Path("a/b.json")

    def test_optional_accepts_none(self):
        assert make(where=None).where is None

    def test_list_items_are_coerced_and_indexed_on_failure(self):
        assert make(tags=["a", "b"]).tags == ["a", "b"]
        with pytest.raises(RequestError, match=r"tags\[1\]"):
            make(tags=["a", 2])

    def test_dict_values_are_coerced_and_named_on_failure(self):
        assert make(limits={"low": 1}).limits == {"low": pytest.approx(1.0)}
        with pytest.raises(RequestError, match="limits.low"):
            make(limits={"low": "x"})

    def test_nested_dataclass_is_built(self):
        result = make(inner={"label": "x", "weight": 2})
        assert result.inner == Inner(label="x", weight=2.0)

    def test_nested_errors_carry_the_full_path(self):
        with pytest.raises(RequestError, match="inner.weight"):
            make(inner={"label": "x", "weight": "heavy"})

    def test_nested_unknown_field_is_caught(self):
        with pytest.raises(RequestError, match="unknown field colour"):
            make(inner={"label": "x", "colour": "red"})


class TestAsData:
    def test_dataclasses_become_plain_data(self):
        value = as_data(make(inner={"label": "x"}, tags=["a"], where="p.json"))
        assert value["inner"] == {"label": "x", "weight": 1.0}
        assert value["tags"] == ["a"]
        assert value["where"] == "p.json"

    def test_nested_containers_are_converted(self):
        assert as_data({"a": [Inner("x")]}) == {"a": [{"label": "x", "weight": 1.0}]}
