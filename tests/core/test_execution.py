"""The core's behaviour: bindings, ordering, caching, failure and reports."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from alambiq.core import registry
from alambiq.core.campaigns import campaign_from_data
from alambiq.core.context import StepContext
from alambiq.core.execution import execute
from alambiq.core.loader import RequestError
from alambiq.core.model import Binding, Campaign, Step


@dataclass(frozen=True, slots=True)
class AddRequest:
    left: int
    right: int = 1


@dataclass(frozen=True, slots=True)
class AddResult:
    total: int
    note: str = ""


@dataclass(frozen=True, slots=True)
class ManyResult:
    values: list[int]


@dataclass(frozen=True, slots=True)
class Item:
    name: str = ""
    amount: int = 0


@dataclass(frozen=True, slots=True)
class CollectRequest:
    items: list[Item]


@dataclass(frozen=True, slots=True)
class CollectResult:
    total: int
    names: str


@pytest.fixture
def counted(monkeypatch):
    """Register a throwaway capability that records how often it ran."""
    calls: list[int] = []
    with registry.isolated():

        @registry.capability(
            "test.add", family="measure", summary="Add two numbers", version="1.0"
        )
        def add(request: AddRequest, ctx: StepContext) -> AddResult:
            """Add the two supplied numbers."""
            calls.append(1)
            if request.left < 0:
                ctx.note("A negative left operand is unusual here.")
            return AddResult(total=request.left + request.right)

        @registry.capability(
            "test.scratch",
            family="measure",
            summary="Needs an empty directory",
            version="1.0",
        )
        def scratch(request: AddRequest, ctx: StepContext) -> AddResult:
            """Create a directory that must not already exist."""
            (ctx.workdir / "out").mkdir()
            (ctx.workdir / "out" / "file").write_text(str(request.left))
            return AddResult(total=request.left)

        @registry.capability(
            "test.many", family="measure", summary="Returns a list", version="1.0"
        )
        def many(request: AddRequest, ctx: StepContext) -> ManyResult:
            """Return a short list."""
            return ManyResult(values=[10, 20, 30])

        @registry.capability(
            "test.collect",
            family="measure",
            summary="Sums the amounts of a list of items",
            version="1.0",
        )
        def collect(request: CollectRequest, ctx: StepContext) -> CollectResult:
            """Add up the items."""
            return CollectResult(
                total=sum(i.amount for i in request.items),
                names=",".join(i.name for i in request.items),
            )

        @registry.capability(
            "test.boom", family="measure", summary="Always fails", version="1.0"
        )
        def boom(request: AddRequest, ctx: StepContext) -> AddResult:
            """Raise, to exercise failure handling."""
            raise RuntimeError("deliberate")

        yield calls


def campaign(*steps: Step, **kwargs) -> Campaign:
    return Campaign(steps=tuple(steps), **kwargs)


class TestRunning:
    def test_a_single_step_runs_and_reports_its_result(self, counted, tmp_path):
        run = execute(
            campaign(Step(id="a", capability="test.add", params={"left": 2})),
            run_dir=tmp_path,
        )
        assert run.ok
        assert run.outputs("a")["total"] == 3
        assert len(counted) == 1

    def test_notes_reach_the_result(self, counted, tmp_path):
        run = execute(
            campaign(Step(id="a", capability="test.add", params={"left": -1})),
            run_dir=tmp_path,
        )
        assert "negative left operand" in run.steps["a"].notes[0]

    def test_a_failing_step_is_reported_not_raised(self, counted, tmp_path):
        run = execute(
            campaign(Step(id="a", capability="test.boom", params={"left": 1})),
            run_dir=tmp_path,
        )
        assert not run.ok
        assert run.steps["a"].status == "failed"
        assert "deliberate" in run.steps["a"].error

    def test_a_bad_request_fails_that_step_with_a_named_field(self, counted, tmp_path):
        run = execute(
            campaign(Step(id="a", capability="test.add", params={"left": "two"})),
            run_dir=tmp_path,
        )
        assert run.steps["a"].status == "failed"
        assert "left" in run.steps["a"].error


class TestScratch:
    def test_a_step_run_again_in_the_same_directory_starts_from_an_empty_workdir(
        self, counted, tmp_path
    ):
        for left in (1, 2):
            run = execute(
                campaign(
                    Step(id="a", capability="test.scratch", params={"left": left})
                ),
                run_dir=tmp_path,
            )
            assert run.ok, run.steps["a"].error
            assert run.outputs("a")["total"] == left


class TestBindings:
    def test_an_output_feeds_the_next_step(self, counted, tmp_path):
        run = execute(
            campaign(
                Step(id="a", capability="test.add", params={"left": 2}),
                Step(
                    id="b",
                    capability="test.add",
                    bind={"left": Binding("a", "total")},
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.outputs("b")["total"] == 4  # (2+1) + 1

    def test_steps_run_in_dependency_order_not_written_order(self, counted, tmp_path):
        run = execute(
            campaign(
                Step(
                    id="b", capability="test.add", bind={"left": Binding("a", "total")}
                ),
                Step(id="a", capability="test.add", params={"left": 5}),
            ),
            run_dir=tmp_path,
        )
        assert run.outputs("b")["total"] == 7

    def test_a_number_in_a_binding_path_picks_an_item_of_a_list(
        self, counted, tmp_path
    ):
        run = execute(
            campaign(
                Step(id="a", capability="test.many", params={"left": 0}),
                Step(
                    id="b",
                    capability="test.add",
                    bind={"left": Binding("a", "values.1")},
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.outputs("b")["total"] == 21

    def test_a_binding_can_fill_one_field_of_one_item_of_a_list(
        self, counted, tmp_path
    ):
        run = execute(
            campaign(
                Step(id="a", capability="test.many", params={"left": 0}),
                Step(
                    id="b",
                    capability="test.collect",
                    params={"items": [{"name": "first"}, {"name": "second"}]},
                    bind={
                        "items.0.amount": Binding("a", "values.0"),
                        "items.1.amount": Binding("a", "values.2"),
                    },
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.outputs("b") == {"total": 40, "names": "first,second"}

    def test_a_binding_creates_the_list_and_items_it_needs(self, counted, tmp_path):
        run = execute(
            campaign(
                Step(id="a", capability="test.many", params={"left": 0}),
                Step(
                    id="b",
                    capability="test.collect",
                    bind={
                        "items.0.amount": Binding("a", "values.1"),
                        "items.1.amount": Binding("a", "values.2"),
                    },
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.outputs("b")["total"] == 50

    def test_binding_to_an_item_far_past_the_end_fails_that_step(
        self, counted, tmp_path
    ):
        run = execute(
            campaign(
                Step(id="a", capability="test.many", params={"left": 0}),
                Step(
                    id="b",
                    capability="test.collect",
                    params={"items": [{"name": "only"}]},
                    bind={"items.3.amount": Binding("a", "values.0")},
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.steps["b"].status == "failed"
        assert "no item 3" in run.steps["b"].error

    def test_binding_into_a_number_fails_that_step(self, counted, tmp_path):
        run = execute(
            campaign(
                Step(id="a", capability="test.many", params={"left": 0}),
                Step(
                    id="b",
                    capability="test.add",
                    params={"left": 1},
                    bind={"left.deeper": Binding("a", "values.0")},
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.steps["b"].status == "failed"
        assert "neither a mapping nor a list" in run.steps["b"].error

    def test_binding_does_not_change_the_campaign_it_came_from(self, counted, tmp_path):
        step = Step(
            id="b",
            capability="test.collect",
            params={"items": [{"name": "only"}]},
            bind={"items.0.amount": Binding("a", "values.0")},
        )
        plan = campaign(Step(id="a", capability="test.many", params={"left": 0}), step)
        execute(plan, run_dir=tmp_path / "one")
        assert step.params == {"items": [{"name": "only"}]}

    def test_an_item_past_the_end_of_a_list_fails_that_step(self, counted, tmp_path):
        run = execute(
            campaign(
                Step(id="a", capability="test.many", params={"left": 0}),
                Step(
                    id="b",
                    capability="test.add",
                    bind={"left": Binding("a", "values.3")},
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.steps["b"].status == "failed"
        assert "values.3" in run.steps["b"].error

    def test_binding_to_a_missing_output_fails_that_step(self, counted, tmp_path):
        run = execute(
            campaign(
                Step(id="a", capability="test.add", params={"left": 1}),
                Step(
                    id="b", capability="test.add", bind={"left": Binding("a", "nope")}
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.steps["b"].status == "failed"
        assert "nope" in run.steps["b"].error

    def test_a_cycle_is_refused_when_the_campaign_is_built(self):
        with pytest.raises(ValueError, match="cycle"):
            campaign(
                Step(
                    id="a", capability="test.add", bind={"left": Binding("b", "total")}
                ),
                Step(
                    id="b", capability="test.add", bind={"left": Binding("a", "total")}
                ),
            ).ordered()

    def test_a_step_follows_the_steps_it_binds_to_whatever_the_declared_order(self):
        plan = campaign(
            Step(id="c", capability="test.add", bind={"left": Binding("b", "total")}),
            Step(id="b", capability="test.add", bind={"left": Binding("a", "total")}),
            Step(id="a", capability="test.add", params={"left": 1}),
            Step(id="d", capability="test.add", params={"left": 2}),
        )
        assert [s.id for s in plan.ordered()] == ["a", "d", "b", "c"]

    def test_ordering_computes_no_value_that_it_discards(self):
        assert "ready" not in Campaign.ordered.__code__.co_varnames

    def test_binding_to_an_unknown_step_is_refused(self):
        with pytest.raises(ValueError, match="unknown step"):
            campaign(
                Step(
                    id="a", capability="test.add", bind={"left": Binding("z", "total")}
                )
            )


class TestCache:
    def test_an_unchanged_step_is_reused(self, counted, tmp_path):
        plan = campaign(Step(id="a", capability="test.add", params={"left": 2}))
        first = execute(plan, run_dir=tmp_path)
        second = execute(plan, run_dir=tmp_path)
        assert first.steps["a"].reused is False
        assert second.steps["a"].reused is True
        assert second.outputs("a")["total"] == 3
        assert len(counted) == 1

    def test_a_changed_request_recomputes(self, counted, tmp_path):
        execute(
            campaign(Step(id="a", capability="test.add", params={"left": 2})),
            run_dir=tmp_path,
        )
        execute(
            campaign(Step(id="a", capability="test.add", params={"left": 3})),
            run_dir=tmp_path,
        )
        assert len(counted) == 2

    def test_adding_a_step_reuses_the_others(self, counted, tmp_path):
        first = Step(id="a", capability="test.add", params={"left": 2})
        execute(campaign(first), run_dir=tmp_path)
        run = execute(
            campaign(
                first,
                Step(
                    id="b", capability="test.add", bind={"left": Binding("a", "total")}
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.steps["a"].reused is True
        assert run.steps["b"].reused is False
        assert len(counted) == 2

    def test_fresh_ignores_the_cache(self, counted, tmp_path):
        plan = campaign(Step(id="a", capability="test.add", params={"left": 2}))
        execute(plan, run_dir=tmp_path)
        execute(plan, run_dir=tmp_path, fresh=True)
        assert len(counted) == 2

    def test_a_damaged_cache_entry_is_a_miss_not_a_failure(self, counted, tmp_path):
        plan = campaign(Step(id="a", capability="test.add", params={"left": 2}))
        execute(plan, run_dir=tmp_path)
        for path in (tmp_path / "cache").rglob("*.json"):
            path.write_text("{ not json")
        run = execute(plan, run_dir=tmp_path)
        assert run.ok and len(counted) == 2


class TestCampaignFromData:
    def test_a_mapping_becomes_a_campaign(self, counted):
        plan = campaign_from_data(
            {
                "id": "demo",
                "steps": [
                    {"id": "a", "capability": "test.add", "params": {"left": 1}},
                    {"id": "b", "capability": "test.add", "bind": {"left": "a.total"}},
                ],
            }
        )
        assert plan.id == "demo"
        assert plan.steps[1].bind["left"] == Binding("a", "total")

    def test_unknown_step_field_is_named(self, counted):
        with pytest.raises(RequestError, match="unknown field method"):
            campaign_from_data(
                {"steps": [{"id": "a", "capability": "test.add", "method": "x"}]}
            )

    def test_a_malformed_binding_is_explained(self, counted):
        with pytest.raises(RequestError, match="step.output"):
            campaign_from_data(
                {
                    "steps": [
                        {"id": "a", "capability": "test.add", "bind": {"left": "a"}}
                    ]
                }
            )

    def test_no_steps_is_refused(self):
        with pytest.raises(RequestError, match="at least 1 step"):
            campaign_from_data({"steps": []})


class TestReport:
    def test_a_report_is_one_self_contained_file(self, counted, tmp_path):
        run = execute(
            campaign(Step(id="a", capability="test.add", params={"left": -2})),
            run_dir=tmp_path,
        )
        page = run.report(tmp_path / "report.html")
        text = page.read_text()
        assert text.startswith("<!doctype html>")
        assert "test.add" in text and "Add two numbers" in text
        # Notes appear, and nothing is fetched from the network.
        assert "negative left operand" in text
        assert "http://" not in text and "https://" not in text

    def test_a_failure_is_visible_in_the_report(self, counted, tmp_path):
        run = execute(
            campaign(Step(id="a", capability="test.boom", params={"left": 1})),
            run_dir=tmp_path,
        )
        assert "deliberate" in run.report(tmp_path / "r.html").read_text()


class TestRegistry:
    def test_a_duplicate_capability_id_is_refused(self, counted):
        with pytest.raises(ValueError, match="already registered"):

            @registry.capability("test.add", family="measure", summary="again")
            def other(request: AddRequest, ctx: StepContext) -> AddResult:
                """Duplicate."""
                return AddResult(total=0)

    def test_an_unannotated_capability_is_refused(self, counted):
        with pytest.raises(TypeError, match="annotated"):

            @registry.capability("test.loose", family="measure", summary="loose")
            def loose(request, ctx):
                """No annotations."""
                return None

    def test_an_unknown_preset_lists_the_known_ones(self, counted):
        @registry.capability(
            "test.preset",
            family="measure",
            summary="presets",
            presets={"quick": {"right": 9}},
        )
        def preset(request: AddRequest, ctx: StepContext) -> AddResult:
            """Presets."""
            return AddResult(total=request.left + request.right)

        with pytest.raises(KeyError, match="quick"):
            registry.get("test.preset").preset("slow")

    def test_a_preset_supplies_defaults_that_params_override(self, counted, tmp_path):
        @registry.capability(
            "test.preset2",
            family="measure",
            summary="presets",
            presets={"quick": {"right": 9}},
        )
        def preset2(request: AddRequest, ctx: StepContext) -> AddResult:
            """Presets."""
            return AddResult(total=request.left + request.right)

        run = execute(
            campaign(
                Step(
                    id="a",
                    capability="test.preset2",
                    preset="quick",
                    params={"left": 1},
                ),
                Step(
                    id="b",
                    capability="test.preset2",
                    preset="quick",
                    params={"left": 1, "right": 2},
                ),
            ),
            run_dir=tmp_path,
        )
        assert run.outputs("a")["total"] == 10
        assert run.outputs("b")["total"] == 3


def test_an_input_name_too_long_for_the_file_system_is_reported_as_a_missing_file(
    tmp_path,
):
    from alambiq.core.context import StepContext
    from alambiq.core.store import Store

    context = StepContext(
        step_id="s", store=Store(tmp_path / "store"), workdir=tmp_path, base=tmp_path
    )
    with pytest.raises(FileNotFoundError) as caught:
        context.resolve("MKV" * 200)
    assert "no input file at MKVMKV" in str(caught.value)
    assert len(str(caught.value)) < 200
