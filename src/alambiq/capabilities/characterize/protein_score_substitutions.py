"""Score single substitutions of a protein with a protein language model.

For each requested position the residue is masked and ESM C 300M is run on the
sequence. The log-probability that the model assigns to each of the 20
standard amino acids at that position is recorded. The log-likelihood ratio
(masked-marginal score, Meier et al., 2021) of a substitution is the
log-probability of the mutant minus that of the wild type at the same position:
positive when the model assigns the mutant a higher probability than the residue of the
protein in that context, negative when it assigns a lower one. The values come
from the pretrained model without fitting.

What the result supports: a ranking of substitutions by the probability that a
protein language model assigns to them. Meier et al., 2021 evaluated the
log-likelihood ratio against deep mutational scans. The score is in
natural-log units (nats).

Limits. The model was trained on natural sequences, so it favors substitutions
that natural proteins make; the score applies to the likelihood of a
substitution in natural sequences. The score is independent of the function to
be improved, so a variant that is better at an unnatural function can receive a
lower score. The Spearman correlation of the log-likelihood ratio with measured
effects is given per dataset in `docs/methods/variant-proposal.md`. The score
evaluates one position at a time with the rest of the sequence as given, so
each substitution is scored on its own; the score of a combination of two
substitutions is not defined. The score is a function of the sequence alone;
structure, cofactors, temperature and pH are not inputs. The score applies to
sequences within the training distribution of the model. A sequence of more than
1,024 residues is refused.
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


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    """The protein, the positions to score, and where the model runs.

    protein:
        A path to a FASTA with one protein of 1 to 1,024 residues from the 20 standard
        amino acids.
    positions:
        The one-based positions to score; every position when empty. Each is one
        model pass. Duplicate positions are merged, and positions are scored in
        ascending order.
    device:
        `cpu`, `mps` or `cuda`; default `cpu`. An unavailable device is an
        error; there is no fallback to another device.
    threads:
        Threads of the host CPU used by the engine, applied on every device, 1
        to 32; default 4.
    batch_size:
        Number of masked copies of the sequence to run at once, 1 to 64;
        default 8.
    """

    protein: str
    positions: list[int] = field(default_factory=list)
    device: Literal["cpu", "mps", "cuda"] = "cpu"
    threads: int = 4
    batch_size: int = 8


@dataclass(frozen=True, slots=True)
class PositionScore:
    """The distribution of the model over the residues at one position."""

    position: int
    wild_type: str
    wild_type_log_prob: float
    top_residue: str
    top_log_prob: float
    entropy_bits: float
    """Entropy of the distribution over the 20 residues at this position; a low value means that the probability is concentrated on few residues."""


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """The scored positions with the log-probability of the wild-type residue, the most probable residue and the entropy, the fraction of positions at which the wild type is most probable, and the stored record."""

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
    """The log-probabilities and log-likelihood ratio of every substitution, in the form that `variant.propose` reads."""
    device: str
    engine_versions: dict[str, Any]


@capability(
    "protein.score_substitutions",
    version="1.0",
    family="characterize",
    summary="Score single substitutions of a protein by the log-likelihood ratio of a language model",
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
    positions = sorted(set(request.positions)) or list(range(1, len(sequence) + 1))
    if any(not 1 <= p <= len(sequence) for p in positions):
        raise ValueError(
            f"positions: values outside 1 to {len(sequence)} are given; positions "
            f"must lie between 1 and {len(sequence)}"
        )
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
    )
