"""Search a supplied protein set for relatives of a query.

MMseqs2 (`easy-search`) runs the search in its own engine environment. The
capability passes the query and the protein set to it and selects the hits that
meet the thresholds of the request. A hit reports the target, expectation
value, identity, coverage, bit score and alignment length; the alignment itself
is not returned. At most 300 results per query pass the MMseqs2 prefilter (its
default `--max-seqs`), so a protein set with more relatives yields at most 300
hits. The presets `close-relatives` (`min_identity` 0.6, `max_evalue` 1e-10)
and `distant-relatives` (`min_identity` 0.2, `sensitivity` 7.5) set the
fields named.

What the result supports: sequence similarity against the supplied protein set.
A hit has an expectation value at or below `max_evalue`; the lower the value,
the stronger the evidence of homology. Homologous proteins can differ in
function.

Limits. A relative that is absent from the protein set cannot be found. An
expectation value depends on the amount of sequence searched, so the same
alignment has a lower (better) expectation value against a small protein set
than against a large one, and values are comparable only between searches of
the same protein set. Identity is measured over the aligned region, which can
be a small part of either sequence; coverage is the fraction of the query that
the aligned region spans (target coverage is unconstrained). A higher
sensitivity finds more distant relatives and increases the run time; the
thresholds apply at every sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alambiq.capabilities.characterize.protein_search_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """The query, the protein set and the thresholds that a hit must meet.

    query:
        A path to a protein FASTA with one sequence.
    panel:
        A path to a protein FASTA of the proteins to search (the protein set).
    max_evalue:
        Hits at or below this expectation value are selected; default 0.001.
    min_identity:
        Hits whose sequence identity is at least this fraction are selected;
        default 0.3.
    min_coverage:
        Hits whose alignment covers at least this fraction of the query are
        selected; default 0.5.
    sensitivity:
        MMseqs2 sensitivity (`-s`), 1 to 7.5; default 5.7, the default of
        MMseqs2. A higher value finds more distant relatives and increases the
        run time.
    """

    query: str
    panel: str
    max_evalue: float = 1e-3
    min_identity: float = 0.3
    min_coverage: float = 0.5
    sensitivity: float = 5.7


@dataclass(frozen=True, slots=True)
class Hit:
    """One selected relative."""

    target: str
    evalue: float
    identity: float
    coverage: float
    bits: float
    alignment_length: int


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The selected hits, and the protein set that was searched (field `panel_size`)."""

    query: str
    panel_size: int
    hits: list[Hit]
    considered: int
    thresholds: dict[str, float]
    engine_versions: dict[str, Any]


@capability(
    "protein.search",
    version="1.0",
    family="characterize",
    summary="Search a supplied protein set for relatives of a query",
    render=render,
    environment="mmseqs",
    presets={
        "close-relatives": {"min_identity": 0.6, "max_evalue": 1e-10},
        "distant-relatives": {"min_identity": 0.2, "sensitivity": 7.5},
    },
)
def search(request: SearchRequest, ctx: StepContext) -> SearchResult:
    """Search the protein set and return the hits that meet the thresholds."""
    from alambiq.capabilities.characterize.science.profiles import check_sensitivity

    check_sensitivity(request.sensitivity)
    query = ctx.resolve(request.query)
    panel = ctx.resolve(request.panel)

    found = ctx.run_in_environment(
        "mmseqs",
        "mmseqs_search:run",
        {
            "query": str(query),
            "panel": str(panel),
            "max_evalue": request.max_evalue,
            "min_identity": request.min_identity,
            "min_coverage": request.min_coverage,
            "sensitivity": request.sensitivity,
        },
    )

    hits = [
        Hit(
            target=row["target"],
            evalue=float(row["evalue"]),
            identity=float(row["identity"]),
            coverage=float(row["coverage"]),
            bits=float(row["bits"]),
            alignment_length=int(row["alignment_length"]),
        )
        for row in found.get("hits", [])
    ]
    if not hits:
        ctx.note("No panel sequence met the thresholds.")
    return SearchResult(
        query=found.get("query_id", query.stem),
        panel_size=int(found.get("panel_size", 0)),
        hits=hits,
        considered=int(found.get("considered", len(hits))),
        thresholds={
            "max_evalue": request.max_evalue,
            "min_identity": request.min_identity,
            "min_coverage": request.min_coverage,
        },
        engine_versions=found.get("versions", {}),
    )
