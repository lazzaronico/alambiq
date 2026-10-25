"""Scoring a reference panel and transferring literature residues.

Pure functions over sequences and alignment rows an engine returned. Nothing
here reads a file or runs a program.
"""

from __future__ import annotations

import re
from typing import Any

SEQUENCE_ID = re.compile(r"[A-Za-z0-9_.-]+")
PROTEIN = re.compile(r"[A-Z]+")
MAX_PROTEINS = 2000


def parse_proteins(text: str, label: str) -> dict[str, str]:
    """Read a protein FASTA into {identifier: sequence}, refusing anything loose."""
    records: dict[str, str] = {}
    name: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].split()[0] if line[1:].strip() else ""
            if not SEQUENCE_ID.fullmatch(name):
                raise ValueError(
                    f"{label}: identifier {name!r} must use only letters, digits, "
                    "underscore, dot and hyphen"
                )
            if name in records:
                raise ValueError(f"{label}: identifier {name} appears twice")
            records[name] = ""
        elif name is None:
            raise ValueError(f"{label}: sequence lines must follow a named header")
        else:
            records[name] += line.upper()
    if not records:
        raise ValueError(f"{label}: no protein sequences found")
    if len(records) > MAX_PROTEINS:
        raise ValueError(
            f"{label}: the file holds {len(records)} proteins; at most {MAX_PROTEINS} "
            "are accepted"
        )
    for identifier, sequence in records.items():
        if not PROTEIN.fullmatch(sequence):
            raise ValueError(
                f"{label}: {identifier} must be uppercase amino-acid letters"
            )
    return records


def fasta(records: dict[str, str]) -> str:
    return "".join(f">{name}\n{sequence}\n" for name, sequence in records.items())


def auc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Probability a random positive outscores a random negative; ties count half."""
    comparisons = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                comparisons += 1.0
            elif positive == negative:
                comparisons += 0.5
    return round(comparisons / (len(positive_scores) * len(negative_scores)), 6)


def check_panel(
    references: dict[str, str],
    candidates: dict[str, str],
    positive: list[str],
    negative: list[str],
) -> None:
    if len(references) < 2:
        raise ValueError(
            f"references: the file holds {len(references)} proteins; at least two "
            "proteins are required"
        )
    overlap = sorted(set(references) & set(candidates))
    if overlap:
        raise ValueError(
            "candidates: reference and candidate identifiers overlap ("
            + ", ".join(overlap)
            + "); an identifier belongs to one set only"
        )
    if not positive or not negative:
        raise ValueError(
            "positive_controls and negative_controls: name at least one positive "
            "control among the references and one negative control among the candidates"
        )
    for name in positive:
        if name not in references:
            raise ValueError(
                f"positive_controls: {name!r} must be one of the references, "
                "which are the proteins with experimental support"
            )
    for name in negative:
        if name not in candidates:
            raise ValueError(
                f"negative_controls: {name!r} must be one of the candidates"
            )
    if len(set(positive)) != len(positive) or len(set(negative)) != len(negative):
        raise ValueError(
            "positive_controls or negative_controls: a control is named twice; "
            "each control is named once"
        )


def score_panel(
    references: dict[str, str],
    candidates: dict[str, str],
    alignments: list[dict[str, Any]],
    positive: list[str],
    negative: list[str],
    minimum_bits: float,
) -> dict[str, Any]:
    """Rank every protein by its best alignment to a reference, and judge the controls.

    `alignments` are the engine's rows, with every reference as a query and
    every reference and candidate as a target. A protein is never scored against
    itself.
    """
    everything = references | candidates
    hits: dict[str, list[dict[str, Any]]] = {name: [] for name in everything}
    for row in alignments:
        if row["query"] != row["target"]:
            hits[row["target"]].append(row)

    positive_set, negative_set = set(positive), set(negative)
    scored: dict[str, dict[str, Any]] = {}
    for name, sequence in sorted(everything.items()):
        best = min(
            hits[name],
            key=lambda row: (-row["bits"], row["evalue"], row["query"]),
            default=None,
        )
        if name in positive_set:
            evidence_class = "positive-control"
        elif name in negative_set:
            evidence_class = "hard-negative-control"
        elif name in references:
            evidence_class = "reference"
        else:
            evidence_class = "unresolved-candidate"
        scored[name] = {
            "identifier": name,
            "evidence_class": evidence_class,
            "sequence_length": len(sequence),
            "sequence": sequence,
            "best_reference": None if best is None else best["query"],
            "bits": 0.0 if best is None else best["bits"],
            "evalue": None if best is None else best["evalue"],
            "identity": None if best is None else best["identity"],
            "alignment_length": None if best is None else best["alignment_length"],
            "query_coverage": None if best is None else best["query_coverage"],
            "target_coverage": None if best is None else best["target_coverage"],
        }
    ranked = sorted(
        scored.values(), key=lambda item: (-item["bits"], item["identifier"])
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    positive_scores = [scored[name]["bits"] for name in sorted(positive_set)]
    negative_scores = [scored[name]["bits"] for name in sorted(negative_set)]
    recall = sum(score >= minimum_bits for score in positive_scores) / len(
        positive_scores
    )
    lowest_positive, highest_negative = min(positive_scores), max(negative_scores)

    def length_score(name: str) -> float:
        others = [s for ref, s in references.items() if ref != name]
        return float(-min(abs(len(everything[name]) - len(s)) for s in others))

    sequence_auc = auc(positive_scores, negative_scores)
    baseline_auc = auc(
        [length_score(n) for n in sorted(positive_set)],
        [length_score(n) for n in sorted(negative_set)],
    )
    return {
        "reference_count": len(references),
        "positive_control_count": len(positive_set),
        "hard_negative_control_count": len(negative_set),
        "unresolved_candidate_count": len(set(candidates) - negative_set),
        "positive_control_recall": round(recall, 6),
        "minimum_positive_bits": lowest_positive,
        "maximum_hard_negative_bits": highest_negative,
        "hard_negative_margin_bits": round(lowest_positive - highest_negative, 6),
        "all_hard_negatives_below_positives": highest_negative < lowest_positive,
        "control_auroc": sequence_auc,
        "length_only_baseline_auroc": baseline_auc,
        "control_auroc_lift_over_length": round(sequence_auc - baseline_auc, 6),
        "ranked_panel": ranked,
    }


def check_features(features: list[dict[str, Any]], reference: str) -> None:
    """Refuse a feature list that does not describe this reference."""
    if not features:
        raise ValueError("features: name at least one reference feature")
    positions = [f["position"] for f in features]
    if len(set(positions)) != len(positions):
        raise ValueError("features: the reference feature positions must be unique")
    for f in features:
        position, residue = f["position"], f["residue"]
        if position < 1:
            raise ValueError(
                f"features: position {position} must be a positive integer"
            )
        if len(residue) != 1 or not residue.isalpha() or not residue.isupper():
            raise ValueError(
                f"features: residue {residue!r} at position {position} must be "
                "one uppercase amino-acid letter"
            )
        if not f["role"].strip() or not f["citation"].strip():
            raise ValueError(
                f"features: the feature at position {position} needs a role and "
                "the citation that establishes it"
            )
        if position > len(reference) or reference[position - 1] != residue:
            raise ValueError(
                "features: the reference has "
                f"{reference[position - 1] if position <= len(reference) else 'no residue'}"
                f" at position {position}, not {residue}"
            )


def transfer_residues(
    features: list[dict[str, Any]], alignment: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Read what each reference feature aligns to in the candidate."""
    observed: dict[int, tuple[int | None, str | None]] = {}
    if alignment is not None:
        by_position = {f["position"] for f in features}
        query_position = alignment["query_start"]
        target_position = alignment["target_start"]
        for q, t in zip(
            alignment["query_alignment"], alignment["target_alignment"], strict=True
        ):
            if q != "-" and query_position in by_position:
                observed[query_position] = (
                    target_position if t != "-" else None,
                    None if t == "-" else t,
                )
            if q != "-":
                query_position += 1
            if t != "-":
                target_position += 1
    transfers = []
    for feature in sorted(features, key=lambda f: f["position"]):
        position, residue = observed.get(feature["position"], (None, None))
        transfers.append(
            {
                "reference_position": feature["position"],
                "expected_residue": feature["residue"],
                "role": feature["role"],
                "citation": feature["citation"],
                "target_position": position,
                "observed_residue": residue,
                "conserved": residue == feature["residue"],
            }
        )
    return transfers


def assess_candidates(
    panel: list[dict[str, Any]],
    features: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    policy: dict[str, float],
) -> list[dict[str, Any]]:
    """Decide advance, withhold or reject for every protein in the panel."""
    best: dict[str, dict[str, Any]] = {}
    for row in alignments:
        previous = best.get(row["target"])
        if previous is None or row["bits"] > previous["bits"]:
            best[row["target"]] = row

    assessments = []
    for item in sorted(panel, key=lambda record: record["identifier"]):
        name = item["identifier"]
        alignment = best.get(name)
        transfers = transfer_residues(features, alignment)
        mapped = sum(t["target_position"] is not None for t in transfers) / len(
            transfers
        )
        conserved = sum(t["conserved"] for t in transfers) / len(transfers)
        retrieval_bits = float(item["bits"])
        coverage = 0.0 if alignment is None else alignment["query_coverage"]

        if alignment is None:
            decision = "reject"
            reasons = ["no-reference-alignment"]
        elif retrieval_bits < policy["minimum_retrieval_bits"]:
            decision = "reject"
            reasons = ["insufficient-retrieval-evidence"]
            if conserved < policy["minimum_conserved_feature_fraction"]:
                reasons.append("literature-features-not-conserved")
        elif coverage < policy["minimum_query_coverage"]:
            decision = "withhold"
            reasons = ["reference-coverage-outside-applicability"]
        elif mapped < policy["minimum_mapped_feature_fraction"]:
            decision = "withhold"
            reasons = ["literature-features-not-fully-mapped"]
        elif conserved < policy["minimum_conserved_feature_fraction"]:
            decision = "reject"
            reasons = ["literature-features-not-conserved"]
        else:
            decision = "advance"
            reasons = ["retrieval-and-literature-features-compatible"]

        assessments.append(
            {
                "identifier": name,
                "evidence_class": item["evidence_class"],
                "decision": decision,
                "reason_codes": reasons,
                "retrieval_bits": retrieval_bits,
                "alignment_bits": 0.0 if alignment is None else alignment["bits"],
                "identity": 0.0 if alignment is None else alignment["identity"],
                "query_coverage": coverage,
                "target_coverage": (
                    0.0 if alignment is None else alignment["target_coverage"]
                ),
                "mapped_feature_fraction": round(mapped, 6),
                "conserved_feature_fraction": round(conserved, 6),
                "residue_transfers": transfers,
            }
        )
    return assessments
