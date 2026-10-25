"""Propose single substitutions of a protein to test, ranked by language-model score, predicted stability, or both.

Each substitution has up to three scores. The first is the
log-likelihood ratio (`llr`) assigned by a protein language model
(`protein.score_substitutions`); a higher value means that the model assigns
the mutant residue a higher probability than the wild-type residue. The second
is the predicted change in folding free energy (`protein.predict_stability_change`);
a lower value means a more stabilizing substitution. The third, available when an
alignment of homologs is supplied, is the number of homologs that carry the mutant
residue at that position. The homolog count is reported and can restrict the
proposals (`homolog_only`). The ranking
criterion (`ranking`) is the language-model score (`llr`), the predicted
stability (`stability`), or the mean of the two percentile ranks of a
substitution among the substitutions that pass the filters (`combined`; 0 is the
best rank, and tied values share their mean rank).

Five filters are available: a lower bound on the language-model score, an upper
bound on the predicted destabilization, a restriction to residues present in the
homologs, a cap per position, and the positions to consider or to exclude. The
result counts the substitutions removed under the name of the reason:
`position-not-selected`, `position-excluded`, `below-minimum-llr`,
`no-stability-prediction`, `destabilizing`, `not-seen-in-homologs` and
`position-cap`. The count for `position-cap` includes only the substitutions
skipped before `count` proposals were chosen.

The proposals are also returned in the request form of
`protein.predict_stability_change` (`stability_request`), so that a binding
connects the two capabilities: propose by language-model score, predict the
stability of the proposals, then propose again with `ranking` set to `combined`.

The result lists at most `count` substitutions with their scores, ranked
deterministically from the supplied records. Ties in the ranking criterion are
ordered by language-model score (higher first) when `ranking` is not `llr`, then
by position and by mutant residue.

Limits. The scores are predictions, and how well they rank real variants depends on
the protein and on the property measured. The retrospective benchmark of this
capability (`docs/methods/variant-proposal.md`) states, for each reference dataset,
how the language-model score and two baselines (a random ranking and the BLOSUM62
score of the substitution) ranked its measured variants. The stability
prediction has its own benchmark (`docs/methods/embeddings-and-prediction.md`).
The benchmark results apply to the reference datasets. The combined rank is the
unweighted mean of the two percentile ranks; its values are percentile ranks and
the weights are set to be equal. Single substitutions are scored one at a time;
each score applies to the substitution alone. The language-model score reflects
the probability of a substitution among natural proteins; the stability score
reflects the predicted change in folding free energy. In the benchmark the
language-model score correlates positively with the measurements for single
substitutions of 3 natural-enzyme datasets and negatively for an engineered
transaminase lineage.
Homolog counts describe the supplied alignment, including its redundancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alambiq.capabilities.engineer.variant_propose_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class ProposeRequest:
    """The score records, the optional evidence, and the ranking and filter settings.

    scores:
        Path to a stored `protein.score_substitutions` record.
    stability:
        Optional. Path to a stored `protein.predict_stability_change` record for the
        same sequence. A substitution that is absent from this record is excluded when
        `ranking` is `stability` or `combined`; when `ranking` is `llr`, it is ranked
        by its language-model score. The `maximum_ddg_kcal_mol` filter does not
        apply to a substitution that is absent from this record.
    homologs:
        Optional. Path to an alignment of homologs whose first row is the sequence
        that is changed (the wild type of `scores`).
    homologs_format:
        `fasta` for aligned FASTA with `-` gaps, or `a3m`, where lowercase letters are
        insertions that are ignored. Default `fasta`.
    ranking:
        `llr` (log-likelihood ratio) ranks by the language-model score, `stability` by
        the predicted change, `combined` by the mean of the two percentile ranks.
        Default `llr`.
    count:
        Number of substitutions to propose, 1 to 2,000. Default 20.
    positions:
        Consider only these one-based positions; all when empty.
    exclude_positions:
        No substitution is proposed at these positions, for example the catalytic
        residues.
    homolog_only:
        Propose only residues that at least one homolog has at that position.
        Requires `homologs`.
    max_per_position:
        At most this many proposals at any one position, at least 1. Without
        `max_per_position`, the number per position is not limited.
    minimum_llr:
        Exclude substitutions whose log-likelihood ratio is below this value.
    maximum_ddg_kcal_mol:
        Exclude substitutions whose predicted destabilization exceeds this value, in
        kcal/mol.
    author_offset:
        The residue number in the structure minus the sequence position, used to write
        the stability request; 0 when the structure is numbered like the sequence
        (position 1 is residue 1).
    """

    scores: str
    stability: str | None = None
    homologs: str | None = None
    homologs_format: Literal["fasta", "a3m"] = "fasta"
    ranking: Literal["llr", "stability", "combined"] = "llr"
    count: int = 20
    positions: list[int] = field(default_factory=list)
    exclude_positions: list[int] = field(default_factory=list)
    homolog_only: bool = False
    max_per_position: int | None = None
    minimum_llr: float | None = None
    maximum_ddg_kcal_mol: float | None = None
    author_offset: int = 0


@dataclass(frozen=True, slots=True)
class Proposal:
    """One proposed substitution and the evidence for it."""

    rank: int
    variant: str
    position: int
    wild_type: str
    mutant: str
    llr: float
    """Log-likelihood ratio, in nats: positive when the model prefers the mutant."""
    ddg_kcal_mol: float | None
    """Predicted folding free-energy change; positive values are destabilizing."""
    llr_rank: float
    """Percentile rank among the substitutions by language-model score; 0 is the best rank."""
    stability_rank: float | None
    combined_rank: float | None
    homolog_count: int | None
    homolog_fraction: float | None


@dataclass(frozen=True, slots=True)
class StabilityQuery:
    """One substitution in the form that `protein.predict_stability_change` takes as a request."""

    residue: str
    wild_type: str
    mutant: str


@dataclass(frozen=True, slots=True)
class ProposeResult:
    """The proposals and the count of substitutions excluded under each reason."""

    protein_id: str
    ranking: str
    candidates: int
    """Number of substitutions that passed the filters."""
    excluded: dict[str, int]
    """Number of substitutions removed under each reason."""
    proposals: list[Proposal]
    stability_request: list[StabilityQuery]
    """The proposals in the request form of `protein.predict_stability_change`, for chaining the two capabilities."""
    score_model: str
    stability_model: str | None
    record: str


@capability(
    "variant.propose",
    version="1.0",
    family="engineer",
    summary="Propose substitutions ranked by language-model score, predicted stability or both",
    render=render,
)
def propose(request: ProposeRequest, ctx: StepContext) -> ProposeResult:
    """Filter and rank the substitutions and count the exclusions by reason."""
    from alambiq.capabilities.engineer.science.variants import (
        homolog_counts,
        rank_variants,
    )

    if not 1 <= request.count <= 2000:
        raise ValueError(f"count must be from 1 to 2,000; found {request.count}.")
    if request.max_per_position is not None and request.max_per_position < 1:
        raise ValueError(
            f"max_per_position must be at least 1; found {request.max_per_position}."
        )
    if request.homolog_only and request.homologs is None:
        raise ValueError(
            "homolog_only is set but homologs is not; homolog_only requires an "
            "alignment of homologs in homologs."
        )
    scores = ctx.read_json(request.scores)
    if "rows" not in scores or "sequence" not in scores:
        raise ValueError(
            "scores must be a protein.score_substitutions record with the fields rows "
            "and sequence."
        )
    stability = ctx.read_json(request.stability) if request.stability else None
    seen = (
        homolog_counts(
            ctx.read_text(request.homologs),
            request.homologs_format,
            scores["sequence"],
        )
        if request.homologs
        else None
    )
    ranked = rank_variants(
        scores,
        stability,
        seen,
        ranking=request.ranking,
        count=request.count,
        positions=list(request.positions),
        exclude_positions=list(request.exclude_positions),
        homolog_only=request.homolog_only,
        max_per_position=request.max_per_position,
        minimum_llr=request.minimum_llr,
        maximum_ddg=request.maximum_ddg_kcal_mol,
    )
    variants = ranked["variants"]
    if len(variants) < request.count:
        ctx.note(
            f"{len(variants)} of the {request.count} requested proposals were found: "
            f"only {ranked['candidates']} candidates passed the filters."
        )
    return ProposeResult(
        protein_id=scores.get("protein_id", ""),
        ranking=request.ranking,
        candidates=ranked["candidates"],
        excluded=ranked["excluded"],
        proposals=[
            Proposal(
                rank=v["rank"],
                variant=v["variant"],
                position=v["position"],
                wild_type=v["wild_type"],
                mutant=v["mutant"],
                llr=v["llr"],
                ddg_kcal_mol=v["ddg_kcal_mol"],
                llr_rank=v["llr_rank"],
                stability_rank=v["stability_rank"],
                combined_rank=v["combined"],
                homolog_count=v["homolog_count"],
                homolog_fraction=v["homolog_fraction"],
            )
            for v in variants
        ],
        stability_request=[
            StabilityQuery(
                residue=str(v["position"] + request.author_offset),
                wild_type=v["wild_type"],
                mutant=v["mutant"],
            )
            for v in variants
        ],
        score_model=scores.get("model", ""),
        stability_model=stability.get("model") if stability else None,
        record=ctx.keep_json(ranked),
    )
