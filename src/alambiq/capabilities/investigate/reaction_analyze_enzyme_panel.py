"""Evaluate five predictors of enzyme activity on held-out measurements of a panel.

The result reports how well sequence and reaction similarity predicts held-out activity
on this panel under the declared split. Two baselines fit no model: the training mean and
the nearest measured training group by joint similarity.

A panel contains measured activities of enzymes on reactions under stated conditions.
Each observation has a source, a status (`measured`, `inactive`, `preparation-failed`,
`untested` or `unreported`) and optional independent and technical replicate
identifiers. Before fitting, the panel declares which observations are in the training
set and which are held out. It also declares the novelty of the held-out observations: a
new sequence, a new reaction, both, or a new pairing only. The capability checks three
conditions:

- All observations of a group are in one partition.
- No sequence-and-reaction pair appears in both partitions.
- The declared novelty holds.

The predictors are:

- the training mean;
- the nearest measured training group (`nearest-reference`);
- kernel ridge on sequence (amino-acid 3-mer cosine);
- kernel ridge on the reaction (chirality-aware Morgan fingerprints of the selected
  participants);
- kernel ridge on both kernels multiplied together.

A group is the set of observations that share a sequence (by SHA-256 digest), an enzyme
form, a reaction (by equation identity) and a condition. Its value is the mean of the
independent replicate means. The fit uses only observations with status `measured` and
a finite value. Observations with another status are counted and retained in the panel,
and their values are not fitted. An activity that the source omits has status
`unreported`.

The request supplies the ridge penalty `alpha`. Choose `alpha` without the held-out
observations, because choosing it with them biases the held-out errors downward. A
predictor improves on a baseline when its held-out mean absolute error is lower than that
of the baseline.

Limits:

- Errors apply to the numeric measured subset under its conditions, for the enzymes and
  reactions in the panel.
- A sequence split separates identical sequences and a reaction split separates identical
  equations. Homologous sequences and chemically similar reactions can occur on both sides
  of the split. The record gives, for each held-out group, the sequence, reaction and joint
  similarity to its nearest training group.
- A family- or scaffold-disjoint test requires a partition made outside the capability.
- Each split requires at least 3 training groups and 2 held-out groups of measured
  observations.
- A panel is refused if it has more than 2,000 groups. It is also refused unless it has
  1 to 500 enzymes, 1 to 100 reactions, 1 to 100 conditions, 1 to 100 sources and 1 to
  10,000 observations, and each enzyme sequence has 3 to 10,000 residues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.investigate.reaction_analyze_enzyme_panel_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

METHODS = (
    "training-mean",
    "nearest-reference",
    "sequence-ridge",
    "reaction-ridge",
    "joint-ridge",
)


@dataclass(frozen=True, slots=True)
class Split:
    """One train and evaluation partition, and the novelty it claims.

    id:
        Identifier of the split.
    novelty:
        What must be new in the evaluation observations: `sequence`, `reaction`,
        `both`, or `pair` (the pairing only). The capability refuses the split if
        the claim does not hold.
    train, evaluation:
        Observation ids. All replicates of a sequence-reaction-condition pair
        belong to the same partition.
    """

    id: str
    novelty: Literal["sequence", "reaction", "both", "pair"]
    train: list[str]
    evaluation: list[str]


@dataclass(frozen=True, slots=True)
class PanelRequest:
    """The panel, the splits to evaluate and the ridge penalty.

    panel:
        A path to the panel as JSON: the activity endpoint (metric, unit and
        description), the sources, the enzymes with their sequences, the balanced
        reactions with the participants selected for fingerprints, the conditions and
        the observations.
    alpha:
        The ridge penalty, from 1e-8 to 1e6, chosen without the held-out observations.
    splits:
        1 to 10 splits to evaluate.
    candidate_ids:
        The enzymes to export as a candidate panel for `candidate.fit` and
        `candidate.select`: 1 to 500 distinct identifiers, each in the panel. The export
        keeps the order of the request.
    """

    panel: str
    alpha: float
    splits: list[Split]
    candidate_ids: list[str]


@dataclass(frozen=True, slots=True)
class Metrics:
    """How well one predictor did on the evaluation groups."""

    n: int
    mae: float
    rmse: float
    bias: float
    """Mean of predicted minus observed."""
    spearman_rho: float | None
    """Rank correlation, or None when either side has no spread."""


@dataclass(frozen=True, slots=True)
class SplitResult:
    """One split's evaluation."""

    id: str
    novelty: str
    training_groups: int
    evaluation_groups: int
    training_state_counts: dict[str, int]
    evaluation_state_counts: dict[str, int]
    metrics: dict[str, Metrics]
    """By predictor: training-mean, nearest-reference, sequence-ridge, reaction-ridge, joint-ridge."""


@dataclass(frozen=True, slots=True)
class PanelResult:
    """The evaluation of every split, and the exported candidates."""

    panel_id: str
    endpoint: dict[str, str]
    observation_counts: dict[str, int]
    splits: list[SplitResult]
    candidate_panel: str
    """A stored candidate panel for `candidate.fit` and `candidate.select`."""
    candidate_fasta: str
    """The same candidates as a FASTA, for `protein.inspect` or `protein.search`."""
    record: str
    versions: dict[str, str]


@capability(
    "reaction.analyze_enzyme_panel",
    version="1.0",
    family="investigate",
    summary="Evaluate activity prediction across enzymes and reactions under declared splits",
    render=render,
)
def analyze_enzyme_panel(request: PanelRequest, ctx: StepContext) -> PanelResult:
    """Validate the panel and the splits, then evaluate the five predictors."""
    import json

    from alambiq.capabilities.investigate.science.enzyme_panel import analyze_panel
    from alambiq.capabilities.measure.science.tables import _unique_object

    panel = json.loads(ctx.read_text(request.panel), object_pairs_hook=_unique_object)
    parameters = {
        "alpha": request.alpha,
        "splits": [
            {
                "id": s.id,
                "novelty": s.novelty,
                "train": list(s.train),
                "evaluation": list(s.evaluation),
            }
            for s in request.splits
        ],
        "candidate_ids": list(request.candidate_ids),
    }
    record = analyze_panel(panel, parameters)

    for evaluation in record["evaluations"]:
        best = min(METHODS, key=lambda m: evaluation["metrics"][m]["mae"])
        if best in ("training-mean", "nearest-reference"):
            ctx.note(
                f"In split {evaluation['id']} the best predictor by mean absolute error is "
                f"{best}, a lookup baseline."
            )
    candidates = record["candidate_panel"]
    fasta = "".join(
        f">{c['sample_id']}\n{c['sequence']}\n" for c in candidates["candidates"]
    )
    return PanelResult(
        panel_id=panel["id"],
        endpoint=panel["endpoint"],
        observation_counts=record["counts"],
        splits=[
            SplitResult(
                id=e["id"],
                novelty=e["novelty"],
                training_groups=len(e["training"]),
                evaluation_groups=len(e["rows"]),
                training_state_counts=e["training_state_counts"],
                evaluation_state_counts=e["evaluation_state_counts"],
                metrics={m: Metrics(**e["metrics"][m]) for m in METHODS},
            )
            for e in record["evaluations"]
        ],
        candidate_panel=ctx.keep_json(candidates),
        candidate_fasta=str(ctx.keep_text(fasta, "text/x-fasta").path),
        record=ctx.keep_json(record),
        versions=record["versions"],
    )
