"""Choose which candidates to make and measure next.

The result lists up to `count` candidates from the pool partition of a candidate panel (defined in `candidate.fit`). The stored record gives the reason that each other candidate was not chosen.

Selection is greedy. At each pick, the candidate that ranks first under the strategy is chosen. A seed fixes the order that resolves ties. The constraints are an eligibility list, a cap per group, and a budget against per-candidate costs.

The strategies are:

- `random` orders the pool by a seeded shuffle.
- `diverse` picks first the candidate that is first in the seeded shuffle. At each later pick, it chooses the candidate with the largest distance to its nearest chosen candidate.
- `highest-predicted` and `lowest-predicted` pick the candidate with the highest or the lowest predicted response.
- `uncertain` picks the candidate with the largest predictive standard deviation.
- `highest-predicted-diverse` picks the candidate with the largest sum of two terms: its predicted response, scaled 0 to 1 over the eligible candidates (0.5 for every candidate when all values are equal), and `diversity_weight` times its distance to the nearest chosen candidate.
- `pareto` ranks candidates by non-dominated front over 2 to 5 fitted objectives. A candidate is dominated when another candidate is at least as good on every objective and better on at least one. Front 0 is the set of non-dominated candidates. Front 1 is the set that is non-dominated once front 0 is removed, and so on.

The strategies `highest-predicted`, `lowest-predicted`, `uncertain` and `highest-predicted-diverse` require a stored `candidate.fit` record of the same candidate panel. `pareto` requires one such record per objective. `diverse` and `highest-predicted-diverse` require the request field `diversity`, set to `sequence` or `embedding-cosine`.

The stored record lists every panel candidate that is not chosen, with its reasons:

- `partition-<name>`: the candidate is outside the pool partition;
- `researcher-eligibility`;
- `group-cap`;
- `remaining-budget`;
- `not-selected-in-requested-batch`.

When fewer than `count` candidates satisfy the constraints, the result lists fewer candidates than `requested`, and a note states the shortfall.

Limits. Greedy selection with a seed is reproducible. Each pick ranks first under the strategy among the eligible candidates, given the earlier picks.

- Sequence distance is the fraction of positions that differ. It requires eligible candidates of equal length.
- `candidate.fit` measures the prediction error on the evaluation partition. The pool partition is unmeasured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alambiq.capabilities.engineer.candidate_select_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

Strategy = Literal[
    "random",
    "diverse",
    "highest-predicted",
    "lowest-predicted",
    "uncertain",
    "highest-predicted-diverse",
    "pareto",
]


@dataclass(frozen=True, slots=True)
class CandidateSelectRequest:
    """The candidate panel, the batch size, the strategy and the constraints of a selection.

    panel:
        Path to the candidate panel. Only its `pool` candidates are eligible.
    count, seed:
        The number of candidates to choose (1 to 500), and the seed (0 to
        2,147,483,647) that fixes the order and resolves ties.
    strategy:
        One of the strategies listed in the capability description. Default `random`.
        `highest-predicted`, `lowest-predicted`, `uncertain` and
        `highest-predicted-diverse` require `model`. `highest-predicted-diverse`
        also requires `diversity`. `pareto` requires `models` and `objectives`. The
        first pick of `highest-predicted-diverse` has no chosen candidate and uses the
        predicted response alone.
    model:
        Path to a stored `candidate.fit` record of the same candidate panel, for the
        strategies that use predictions.
    models, objectives:
        For `pareto`: 2 to 5 stored `candidate.fit` records of the same candidate
        panel, and a list of the same length. The list gives, for each record in
        order, whether to `maximize` or `minimize` its predicted response. Fronts are
        taken in order from front 0. Within a front, the objectives are scaled 0 to 1
        over the eligible candidates (a `minimize` objective with its sign reversed)
        and added with equal weight. The candidate with the largest sum is picked
        first.
    diversity_weight:
        For `highest-predicted-diverse`: the weight of the distance from the chosen
        set relative to the predicted response, 0 to 10. Default 1. A weight of 0 is
        equivalent to `highest-predicted`. A larger weight increases the influence of
        the distance.
    diversity, embeddings:
        For `diverse` and `highest-predicted-diverse`: `sequence` distance, or
        `embedding-cosine` with a path to a stored embedding record.
    eligible_ids, exclude_ids:
        `eligible_ids`: only these pool candidates are eligible. All pool candidates
        are eligible when the list is empty. `exclude_ids`: these candidates are
        removed from the eligible set.
    max_per_group:
        At most this many candidates from any one group (1 to 500).
    costs, budget, cost_unit:
        `costs`: a positive cost for every eligible pool candidate, keyed by
        `sample_id`. `budget`: a positive upper bound on the sum of the costs of the
        chosen candidates. `cost_unit`: the unit of both, as nonblank text. Supply the
        three together or not at all.
    """

    panel: str
    count: int
    seed: int
    strategy: Strategy = "random"
    model: str | None = None
    models: list[str] = field(default_factory=list)
    objectives: list[Literal["maximize", "minimize"]] = field(default_factory=list)
    diversity_weight: float | None = None
    diversity: Literal["sequence", "embedding-cosine"] | None = None
    embeddings: str | None = None
    eligible_ids: list[str] = field(default_factory=list)
    exclude_ids: list[str] = field(default_factory=list)
    max_per_group: int | None = None
    costs: dict[str, float] = field(default_factory=dict)
    budget: float | None = None
    cost_unit: str | None = None


@dataclass(frozen=True, slots=True)
class Chosen:
    """One selected candidate: sample id, group id and cost (None when no costs are supplied)."""

    sample_id: str
    group_id: str
    cost: float | None


@dataclass(frozen=True, slots=True)
class CandidateSelectResult:
    """The batch, its total cost, and the reasons that the other candidates were not chosen."""

    selected: list[Chosen]
    requested: int
    total_cost: float | None
    cost_unit: str | None
    rejected: int
    reasons: dict[str, int]
    """The number of candidates for each rejection reason."""
    record: str
    """Path to the stored selection record."""


@capability(
    "candidate.select",
    version="1.1",
    family="engineer",
    summary="Choose the next batch of candidates under explicit constraints",
    render=render,
)
def select(request: CandidateSelectRequest, ctx: StepContext) -> CandidateSelectResult:
    """Choose the batch and record the reason that each other candidate was not chosen."""
    from alambiq.capabilities.engineer.science.selection import select_candidates

    parameters: dict[str, object] = {
        "strategy": request.strategy,
        "count": request.count,
        "seed": request.seed,
    }
    for name in ("eligible_ids", "exclude_ids"):
        value = getattr(request, name)
        if value:
            parameters[name] = list(value)
    if request.objectives:
        parameters["objectives"] = list(request.objectives)
    for name in (
        "max_per_group",
        "diversity",
        "budget",
        "cost_unit",
        "diversity_weight",
    ):
        value = getattr(request, name)
        if value is not None:
            parameters[name] = value
    if request.costs:
        parameters["costs"] = dict(request.costs)

    record = select_candidates(
        ctx.read_json(request.panel),
        parameters,
        model=ctx.read_json(request.model) if request.model is not None else None,
        models=[ctx.read_json(m) for m in request.models] if request.models else None,
        embeddings=(
            ctx.read_json(request.embeddings)
            if request.embeddings is not None
            else None
        ),
    )

    for message in record["diagnostics"]:
        ctx.note(message)
    reasons: dict[str, int] = {}
    for row in record["rejected"]:
        for reason in row["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    return CandidateSelectResult(
        selected=[
            Chosen(
                sample_id=c["sample_id"],
                group_id=c["group_id"],
                cost=c.get("cost"),
            )
            for c in record["selected"]
        ],
        requested=request.count,
        total_cost=record["total_cost"],
        cost_unit=request.cost_unit,
        rejected=len(record["rejected"]),
        reasons=dict(sorted(reasons.items())),
        record=ctx.keep_json(record),
    )
