"""Transfer of annotations from the hits of a homology search to the query.

A hit is a reviewed protein that resembles the query over the aligned region. The
annotations of the strongest hits that meet the thresholds are listed for the query. Each
listed term has its support, the number of selected hits that carry it, the highest
identity among them and the accessions of those hits.
"""

from __future__ import annotations

from typing import Any

CATEGORIES = ("ec", "go_function", "go_process", "go_component", "location")
IDENTICAL = 0.999


def select_hits(
    rows: list[dict[str, Any]],
    *,
    min_identity: float,
    min_coverage: float,
    max_evalue: float,
    max_hits: int,
    exclude_identical: bool,
) -> list[dict[str, Any]]:
    """The hits of one query that meet the thresholds, strongest first, at most `max_hits`.

    Strength is the bit score, then the accession. A hit is identical when it matches the
    query over the whole of both sequences at 100 % identity.
    """
    kept = [
        row
        for row in rows
        if row["identity"] >= min_identity
        and row["coverage"] >= min_coverage
        and row["evalue"] <= max_evalue
        and not (
            exclude_identical
            and row["identity"] >= IDENTICAL
            and row["coverage"] >= IDENTICAL
            and row["target_coverage"] >= IDENTICAL
        )
    ]
    kept.sort(key=lambda row: (-row["bits"], row["target"]))
    return kept[:max_hits]


def _split(category: str, value: str) -> tuple[str, str]:
    if category.startswith("go_"):
        identifier, _, name = value.partition("|")
        return identifier, name
    return value, ""


def transfer(hits: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The terms carried by the hits, per category, best supported first.

    Each term is a dictionary with `term`, `name` (the Gene Ontology label; empty for EC
    numbers and locations), `support`, `best_identity` and `sources`.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for category in CATEGORIES:
        terms: dict[str, dict[str, Any]] = {}
        for hit in hits:
            for value in hit["annotations"].get(category, []):
                identifier, name = _split(category, value)
                entry = terms.setdefault(
                    identifier,
                    {
                        "term": identifier,
                        "name": name,
                        "support": 0,
                        "best_identity": 0.0,
                        "sources": [],
                    },
                )
                entry["support"] += 1
                entry["best_identity"] = max(entry["best_identity"], hit["identity"])
                entry["sources"].append(hit["target"])
        result[category] = sorted(
            terms.values(),
            key=lambda t: (-t["support"], -t["best_identity"], t["term"]),
        )
    return result
