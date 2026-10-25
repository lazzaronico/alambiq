"""What a researcher asks for: a campaign of steps.

A campaign is data. YAML and the Python API both build this object, and
nothing downstream cares which one did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Binding:
    """The assignment of one step's output to a field of another step's request."""

    step: str
    output: str

    @classmethod
    def parse(cls, text: str) -> Binding:
        """Read `step.output` or `step.output.nested.path`."""
        step, _, output = text.partition(".")
        if not step or not output:
            raise ValueError(f"Binding {text!r} is not of the form 'step.output'")
        return cls(step=step, output=output)


@dataclass(frozen=True, slots=True)
class Step:
    """One use of a capability in a campaign."""

    id: str
    capability: str
    preset: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    bind: dict[str, Binding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"Step id {self.id!r} contains other characters; an id has only letters, "
                "digits, hyphens and underscores"
            )


@dataclass(frozen=True, slots=True)
class Campaign:
    """A list of steps with a base directory for relative input paths."""

    steps: tuple[Step, ...]
    id: str = "campaign"
    base: Path = Path()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(
                    f"Step id {step.id!r} appears more than once; ids must be unique"
                )
            seen.add(step.id)
        for step in self.steps:
            for name, binding in step.bind.items():
                if binding.step not in seen:
                    raise ValueError(
                        f"Step {step.id!r} binds {name} to unknown step "
                        f"{binding.step!r}; the steps are {', '.join(sorted(seen))}"
                    )
                if binding.step == step.id:
                    raise ValueError(
                        f"Step {step.id!r} binds to itself; a step binds only to other steps"
                    )

    def ordered(self) -> tuple[Step, ...]:
        """Steps in dependency order, refusing cycles."""
        remaining = {s.id: s for s in self.steps}
        done: list[Step] = []
        satisfied: set[str] = set()
        while remaining:
            progressed = False
            for step in list(self.steps):
                if step.id not in remaining:
                    continue
                if all(b.step in satisfied for b in step.bind.values()):
                    done.append(step)
                    satisfied.add(step.id)
                    del remaining[step.id]
                    progressed = True
            if not progressed:
                raise ValueError(
                    "Steps form a cycle or depend on one: " + ", ".join(sorted(remaining))
                )
        return tuple(done)
