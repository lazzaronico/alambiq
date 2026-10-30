"""Score substitutions of a protein with a protein language model.

Returns a score for each single substitution at the requested positions, and for
each variant of several substitutions that the request lists. The score ranks
substitutions by the probability that a protein language model assigns to them.

The method:

- For each position, the residue is masked and ESM C 300M runs on the sequence.
- The model gives a log-probability for each of the 20 standard amino acids at
  that position.
- The score of a substitution is the log-likelihood ratio (masked-marginal
  score, Meier et al., 2021): the log-probability of the mutant minus that of
  the wild type at the same position. A positive value means the model
  assigns the mutant a higher probability than the wild type in that context.
- The unit is the natural logarithm (nats). The values come from the
  pretrained model without fitting.
- The score of a variant is the sum of the scores of its substitutions, each
  computed on the wild-type sequence. A variant is written as its substitutions
  joined by colons, such as `K183N:T209K`. The positions of the variants are scored
  in addition to `positions`; when `positions` is empty and variants are given,
  only the positions of the variants are scored.

Meier et al., 2021 evaluated the log-likelihood ratio against deep mutational
scans. The Spearman correlation with measured effects is given per dataset in
`docs/methods/variant-proposal.md`.

Limits.

- The model was trained on natural sequences. The score is the likelihood of a
  substitution in natural sequences.
- The score does not depend on the function to be improved. A variant that is
  better at an unnatural function can receive a lower score.
- The score evaluates one position at a time with the rest of the sequence as
  given. The score of a variant is the sum of its substitutions and ignores
  interactions between positions, so it does not distinguish a combination whose
  substitutions affect each other.
- The score is a function of the sequence alone. Structure, cofactors,
  temperature and pH are not inputs.
- The score applies to sequences within the training distribution of the model.
- A sequence of more than 1,024 residues is refused.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from alambiq.capabilities.characterize.protein_score_substitutions_report import render
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

RESOURCE = "esmc-300m"
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
MAX_VARIANTS = 1000


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    """The protein, the positions to score and where the model runs.

    protein:
        A path to a FASTA with one protein of 1 to 1,024 residues from the 20
        standard amino acids.
    positions:
        The one-based positions to score; every position when empty and no
        variant is given. Each position is one model pass. Duplicates are merged,
        and positions are scored in ascending order.
    variants:
        Variants to score, 0 to 1,000, each written as 1 to 50 substitutions
        joined by colons (`K183N:T209K`). A substitution names the wild-type residue,
        its one-based position and a different residue of the 20 amino acids, and
        no two substitutions of a variant share a position. The score of a variant
        is the sum of the scores of its substitutions.
    device:
        `cpu`, `mps` or `cuda`; default `cpu`. An unavailable device is an
        error.
    threads:
        Host CPU threads used by the engine on every device, 1 to 32; default
        4.
    batch_size:
        Masked copies of the sequence to run at once, 1 to 64; default 8.
    """

    protein: str
    positions: list[int] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    threads: int = 4
    batch_size: int = 8


@dataclass(frozen=True, slots=True)
class VariantScore:
    """The score of one variant: the sum of the scores of its substitutions."""

    variant: str
    """The substitutions joined by colons, in the order of the request."""
    substitutions: list[str]
    llr: float
    """The sum of the log-likelihood ratios of the substitutions, in nats."""


@dataclass(frozen=True, slots=True)
class PositionScore:
    """The distribution of the model over the residues at one position."""

    position: int
    wild_type: str
    wild_type_log_prob: float
    top_residue: str
    top_log_prob: float
    entropy_bits: float
    """Entropy of the distribution over the 20 residues at this position. A low value means few residues carry the probability."""


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """The scored positions, the fraction of positions where the wild type is most probable, and the stored record."""

    protein_id: str
    length: int
    sequence_sha256: str
    model: str
    revision: str
    license: str
    positions_scored: int
    substitutions: int
    wild_type_is_top: float
    """The fraction of scored positions at which the residue of highest probability is the wild type."""
    positions: list[PositionScore]
    record: str
    """The log-probabilities and log-likelihood ratio of every substitution, as `variant.propose` reads them."""
    device: str
    engine_versions: dict[str, Any]
    variant_scores: list[VariantScore] = field(default_factory=list)
    """The score of each requested variant, in the order of the request."""


@capability(
    "protein.score_substitutions",
    version="1.1",
    family="characterize",
    summary="Score substitutions and variants of a protein with a protein language model",
    render=render,
    environment="esm",
    resources=(RESOURCE,),
)
def score_substitutions(request: ScoreRequest, ctx: StepContext) -> ScoreResult:
    """Mask each position, read the model's log-probabilities, and tabulate the ratios."""
    import math

    from alambiq.capabilities.shared.sequences import parse_fasta

    name, sequence = parse_fasta(ctx.read_text(request.protein))
    if not 1 <= len(sequence) <= 1024 or set(sequence) - set(CANONICAL):
        raise ValueError(
            "protein: expected one protein of 1–1,024 canonical residues (no X, B, Z, U or O)"
        )
    from alambiq.capabilities.characterize.science.substitutions import parse_variant

    if len(request.variants) > MAX_VARIANTS:
        raise ValueError(f"At most {MAX_VARIANTS} variants are scored in one step")
    parsed = [parse_variant(text, sequence) for text in request.variants]
    given = sorted(set(request.positions))
    if any(not 1 <= p <= len(sequence) for p in given):
        raise ValueError(
            f"positions must lie between 1 and {len(sequence)}; a given value is "
            "outside that range"
        )
    needed = {position for variant in parsed for _, position, _ in variant}
    if given or needed:
        positions = sorted(set(given) | needed)
    else:
        positions = list(range(1, len(sequence) + 1))
    for label, value, low, high in (
        ("threads", request.threads, 1, 32),
        ("batch_size", request.batch_size, 1, 64),
    ):
        if not low <= value <= high:
            raise ValueError(f"{label} must be an integer from {low} to {high}")

    root = resources.verify(RESOURCE)
    pinned = resources.spec(RESOURCE)
    found = ctx.run_in_environment(
        "esm",
        "esmc_score:run",
        {
            "sequence": sequence,
            "positions": [p - 1 for p in positions],
            "model_root": str(root / "model"),
            "device": request.device,
            "threads": request.threads,
            "batch_size": request.batch_size,
        },
        timeout=4 * 3600,
    )

    rows: list[dict[str, Any]] = []
    summaries: list[PositionScore] = []
    top_is_wild = 0
    for position in positions:
        values = found["log_probs"][str(position - 1)]
        wild = sequence[position - 1]
        wild_log = values[CANONICAL.index(wild)]
        probabilities = [math.exp(v) for v in values]
        total = sum(probabilities)
        entropy = -sum(
            (p / total) * math.log2(p / total) for p in probabilities if p > 0
        )
        best = max(range(len(CANONICAL)), key=lambda i: values[i])
        top_is_wild += CANONICAL[best] == wild
        summaries.append(
            PositionScore(
                position=position,
                wild_type=wild,
                wild_type_log_prob=wild_log,
                top_residue=CANONICAL[best],
                top_log_prob=values[best],
                entropy_bits=entropy,
            )
        )
        for index, mutant in enumerate(CANONICAL):
            if mutant == wild:
                continue
            rows.append(
                {
                    "variant": f"{wild}{position}{mutant}",
                    "position": position,
                    "wild_type": wild,
                    "mutant": mutant,
                    "log_prob": values[index],
                    "wild_type_log_prob": wild_log,
                    "llr": values[index] - wild_log,
                }
            )
    llr = {row["variant"]: row["llr"] for row in rows}
    variant_scores = [
        VariantScore(
            variant=":".join(f"{w}{p}{m}" for w, p, m in variant),
            substitutions=[f"{w}{p}{m}" for w, p, m in variant],
            llr=sum(llr[f"{w}{p}{m}"] for w, p, m in variant),
        )
        for variant in parsed
    ]
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    record = {
        "schema_version": "0.1",
        "method": "masked-marginal log-likelihood ratio (Meier et al. 2021)",
        "protein_id": name,
        "sequence": sequence,
        "sequence_sha256": digest,
        "model": "biohub/ESMC-300M",
        "revision": pinned["version"],
        "license": pinned["license"],
        "resource_identity": resources.identity(RESOURCE),
        "versions": found["versions"],
        "device": found["device"],
        "positions": positions,
        "rows": rows,
        "variants": [
            {"variant": v.variant, "substitutions": v.substitutions, "llr": v.llr}
            for v in variant_scores
        ],
    }
    return ScoreResult(
        protein_id=name,
        length=len(sequence),
        sequence_sha256=digest,
        model="biohub/ESMC-300M",
        revision=pinned["version"],
        license=pinned["license"],
        positions_scored=len(positions),
        substitutions=len(rows),
        wild_type_is_top=top_is_wild / len(positions),
        positions=summaries,
        record=ctx.keep_json(record),
        device=found["device"],
        engine_versions=found["versions"],
        variant_scores=variant_scores,
    )
