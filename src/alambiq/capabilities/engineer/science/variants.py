"""Rank single substitutions by a language model's score and a predicted stability change.

A substitution is described by a position in the starting sequence, the residue there and
the residue it becomes. Each candidate may carry a log-likelihood ratio from a protein
language model (higher is more natural), a predicted change in folding free energy (lower
is more stabilizing) and the residues that homologous sequences show at its position. This
module filters the candidates, ranks them, and says why each was kept or left out.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


def homolog_counts(
    text: str, fmt: str, sequence: str
) -> tuple[dict[int, Counter[str]], int]:
    """Residues seen at each query position among aligned homologs.

    The first record of the alignment is the starting sequence and must equal it once gaps
    are removed. Returns the counts per one-based position and the number of homolog
    rows. In A3M, lowercase letters are insertions relative to the query and are ignored.
    """
    records: list[tuple[str, str]] = []
    name, chunks = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name, chunks = line[1:].strip() or f"row{len(records)}", []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        records.append((name, "".join(chunks)))
    if len(records) < 2:
        raise ValueError(
            "The homolog alignment must contain the starting sequence and at least 1 "
            f"further row; found {len(records)} rows."
        )
    rows = []
    for _, aligned in records:
        if fmt == "a3m":
            aligned = "".join(c for c in aligned if not c.islower())
        rows.append(aligned.upper().replace(".", "-"))
    query = rows[0]
    if query.replace("-", "") != sequence:
        raise ValueError(
            "The first row of the homolog alignment, without gaps, differs from the "
            "starting sequence; the first row must be the starting sequence."
        )
    if any(len(r) != len(query) for r in rows):
        raise ValueError(
            "All rows of the homolog alignment must have the length of the first row, "
            f"{len(query)} columns."
        )
    position_of: dict[int, int] = {}
    count = 0
    for column, char in enumerate(query):
        if char != "-":
            count += 1
            position_of[column] = count
    seen: dict[int, Counter[str]] = {p: Counter() for p in range(1, len(sequence) + 1)}
    for row in rows[1:]:
        for column, position in position_of.items():
            if row[column] in CANONICAL:
                seen[position][row[column]] += 1
    return seen, len(rows) - 1


def percentile_ranks(values: list[float]) -> list[float]:
    """Ranks scaled to 0..1, the best (largest) value first; ties share their mean rank."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean = (i + j) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = mean
        i = j + 1
    scale = max(len(values) - 1, 1)
    return [r / scale for r in ranks]


def rank_variants(
    scores: dict[str, Any],
    stability: dict[str, Any] | None,
    homologs: tuple[dict[int, Counter[str]], int] | None,
    *,
    ranking: str,
    count: int,
    positions: list[int],
    exclude_positions: list[int],
    homolog_only: bool,
    max_per_position: int | None,
    minimum_llr: float | None,
    maximum_ddg: float | None,
) -> dict[str, Any]:
    """Filter and rank the candidates; return them with the reason for every exclusion."""
    sequence = scores["sequence"]
    ddg: dict[tuple[int, str], float] = {}
    if stability is not None:
        if stability.get("sequence") != sequence:
            raise ValueError(
                "The sequence of the stability record differs from the sequence of the "
                "scores record; both records must refer to the same sequence."
            )
        for row in stability["rows"]:
            ddg[(row["position"] + 1, row["mutant"])] = row["predicted_ddg_kcal_mol"]
    if ranking in ("stability", "combined") and not ddg:
        raise ValueError(
            f"ranking is {ranking}, which requires a stability record with at least 1 "
            "prediction."
        )
    candidates = []
    excluded: Counter[str] = Counter()
    for row in scores["rows"]:
        p, mutant = row["position"], row["mutant"]
        reason = None
        if positions and p not in positions:
            reason = "position-not-selected"
        elif p in exclude_positions:
            reason = "position-excluded"
        elif minimum_llr is not None and row["llr"] < minimum_llr:
            reason = "below-minimum-llr"
        elif ranking in ("stability", "combined") and (p, mutant) not in ddg:
            reason = "no-stability-prediction"
        elif (
            maximum_ddg is not None
            and (p, mutant) in ddg
            and ddg[(p, mutant)] > maximum_ddg
        ):
            reason = "destabilizing"
        elif homolog_only and homologs is not None:
            if homologs[0][p][mutant] == 0:
                reason = "not-seen-in-homologs"
        if reason:
            excluded[reason] += 1
            continue
        item = {
            "variant": row["variant"],
            "position": p,
            "wild_type": row["wild_type"],
            "mutant": mutant,
            "llr": row["llr"],
            "ddg_kcal_mol": ddg.get((p, mutant)),
            "homolog_count": homologs[0][p][mutant] if homologs else None,
            "homolog_fraction": (
                homologs[0][p][mutant] / homologs[1] if homologs else None
            ),
        }
        candidates.append(item)
    if not candidates:
        raise ValueError("No substitution passes the filters.")
    llr_rank = percentile_ranks([c["llr"] for c in candidates])
    for c, r in zip(candidates, llr_rank, strict=True):
        c["llr_rank"] = r
    have = [i for i, c in enumerate(candidates) if c["ddg_kcal_mol"] is not None]
    if have:
        stable_rank = percentile_ranks([-candidates[i]["ddg_kcal_mol"] for i in have])
        for i, r in zip(have, stable_rank, strict=True):
            candidates[i]["stability_rank"] = r
    for c in candidates:
        c.setdefault("stability_rank", None)
        c["combined"] = (
            (c["llr_rank"] + c["stability_rank"]) / 2
            if c["stability_rank"] is not None
            else None
        )

    def key(c):
        if ranking == "llr":
            return (-c["llr"], c["position"], c["mutant"])
        if ranking == "stability":
            return (c["ddg_kcal_mol"], -c["llr"], c["position"], c["mutant"])
        return (c["combined"], -c["llr"], c["position"], c["mutant"])

    chosen: list[dict[str, Any]] = []
    per_position: Counter[int] = Counter()
    capped = 0
    for c in sorted(candidates, key=key):
        if len(chosen) >= count:
            break
        if (
            max_per_position is not None
            and per_position[c["position"]] >= max_per_position
        ):
            capped += 1
            continue
        per_position[c["position"]] += 1
        chosen.append({**c, "rank": len(chosen) + 1})
    if capped:
        excluded["position-cap"] += capped
    return {
        "ranking": ranking,
        "candidates": len(candidates),
        "excluded": dict(sorted(excluded.items())),
        "variants": chosen,
    }


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman's rank correlation, with ties given their mean rank."""
    ra = percentile_ranks(a)
    rb = percentile_ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")
