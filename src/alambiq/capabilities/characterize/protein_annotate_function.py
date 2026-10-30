"""Transfer function annotations to proteins from their closest reviewed relatives.

Returns, for each protein, the Enzyme Commission (EC) numbers, Gene Ontology terms and
subcellular locations that its closest relatives in UniProtKB/Swiss-Prot carry, with the
evidence for each term. A term states what the relatives are annotated with. The identity of
each source is given, so that the reader can judge how far a term applies to the protein.

The method:

- MMseqs2 searches the pinned Swiss-Prot release with each protein.
- A hit meets the thresholds when its identity, its coverage of the protein and its
  expectation value pass the request. The strongest hits by bit score, at most `max_hits`,
  are the sources of the protein.
- Every EC number, Gene Ontology term and location of a source is listed for the protein
  with its support (the number of sources that carry it), the highest identity among those
  sources and their accessions. Terms are ordered by support, then by identity.
- With `exclude_identical`, a hit that matches the protein over both full sequences at 100 %
  identity is not a source. This annotates a Swiss-Prot protein from the database without its
  own entry.

Limits.

- The terms are those of reviewed entries. A protein with no hit that meets the thresholds is
  returned without annotations.
- Homologous proteins can differ in function. A term from a source at low identity is a
  weaker statement than one from a source at high identity.
- Gene Ontology terms are listed as the entries carry them, whatever their evidence code.
- A location is the first term of each location sentence of an entry, without notes and
  isoform prefixes.
- The coverage threshold applies to the protein. A hit that covers part of a multidomain
  protein carries the annotations of its whole entry.
- A protein of 10 to 5,000 residues from the 20 amino acids and X is searched. A request
  with more than 2,000 proteins is refused.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from alambiq.capabilities.characterize.protein_annotate_function_report import render
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

DATABASE = "swissprot-2026_03"
SEARCH_TIMEOUT_SECONDS = 2 * 3600


@dataclass(frozen=True, slots=True)
class AnnotateRequest:
    """The proteins, the pinned database and the thresholds of a source.

    proteins:
        A path to a FASTA of 1 to 2,000 proteins, each 10 to 5,000 residues from the 20
        amino acids and X, with unique identifiers.
    database:
        The pinned database to search. The only value is `swissprot-2026_03` (UniProtKB/
        Swiss-Prot release 2026_03).
    min_identity:
        A hit is a source when its sequence identity over the aligned region is at least
        this fraction, 0 to 1; default 0.4.
    min_coverage:
        A hit is a source when its alignment covers at least this fraction of the protein,
        0 to 1; default 0.8.
    max_evalue:
        A hit is a source when its expectation value is at most this value, above 0 and at
        most 1,000,000; default 1e-5.
    max_hits:
        The most sources per protein, 1 to 50; default 5.
    exclude_identical:
        Whether a hit that matches the protein over both full sequences at 100 % identity
        is dropped. Default false.
    sensitivity:
        MMseqs2 sensitivity (`-s`), 1 to 7.5; default 5.7, the default of MMseqs2.
    threads:
        Threads used by MMseqs2, 1 to 64; default 8.
    """

    proteins: str
    database: Literal["swissprot-2026_03"] = "swissprot-2026_03"
    min_identity: float = 0.4
    min_coverage: float = 0.8
    max_evalue: float = 1e-5
    max_hits: int = 5
    exclude_identical: bool = False
    sensitivity: float = 5.7
    threads: int = 8


@dataclass(frozen=True, slots=True)
class Source:
    """One hit that annotates a protein."""

    accession: str
    name: str
    organism: str
    identity: float
    coverage: float
    target_coverage: float
    evalue: float
    bits: float


@dataclass(frozen=True, slots=True)
class Term:
    """One annotation term and the sources that carry it."""

    term: str
    """An EC number, a Gene Ontology identifier or a location."""
    name: str
    """The Gene Ontology label; empty for EC numbers and locations."""
    support: int
    """The number of sources that carry the term."""
    best_identity: float
    """The highest identity among those sources."""
    sources: list[str]
    """Their accessions."""


@dataclass(frozen=True, slots=True)
class ProteinAnnotation:
    """The sources and the transferred terms of one protein."""

    id: str
    length: int
    sources: list[Source]
    ec: list[Term]
    go_function: list[Term]
    go_process: list[Term]
    go_component: list[Term]
    location: list[Term]


@dataclass(frozen=True, slots=True)
class AnnotateResult:
    """The annotation of every protein and the database searched."""

    database: str
    database_version: str
    database_identity: str
    proteins: list[ProteinAnnotation]
    annotated: int
    """The proteins with at least one source."""
    unmatched: int
    thresholds: dict[str, Any]
    record: str
    engine_versions: dict[str, Any]


def _check(request: AnnotateRequest) -> None:
    from alambiq.capabilities.characterize.science.profiles import check_sensitivity

    check_sensitivity(request.sensitivity)
    for name, value in (
        ("min_identity", request.min_identity),
        ("min_coverage", request.min_coverage),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not math.isfinite(request.max_evalue) or not 0 < request.max_evalue <= 1e6:
        raise ValueError("max_evalue must be above 0 and at most 1000000")
    for name, value, low, high in (
        ("max_hits", request.max_hits, 1, 50),
        ("threads", request.threads, 1, 64),
    ):
        if not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")


@capability(
    "protein.annotate_function",
    version="1.0",
    family="characterize",
    summary="Transfer function annotations to proteins from their closest reviewed relatives",
    render=render,
    environment="mmseqs",
    resources=(DATABASE,),
)
def annotate_function(request: AnnotateRequest, ctx: StepContext) -> AnnotateResult:
    """Search Swiss-Prot with each protein and list the annotations of the strongest hits."""
    from alambiq.capabilities.characterize.science.enzyme import fasta, parse_proteins
    from alambiq.capabilities.characterize.science.profiles import check_query
    from alambiq.capabilities.characterize.science.transfer import select_hits, transfer

    proteins = parse_proteins(ctx.read_text(request.proteins), "proteins")
    for name, sequence in proteins.items():
        try:
            check_query(sequence)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from None
    _check(request)

    root = resources.verify(DATABASE)
    pinned = resources.spec(DATABASE)
    identity = resources.identity(DATABASE)

    query = ctx.workdir / "proteins.fasta"
    query.write_text(fasta(proteins))
    found = ctx.run_in_environment(
        "mmseqs",
        "mmseqs_annotate:run",
        {
            "proteins": str(query),
            "database": str(root / "swissprot-db"),
            "annotations": str(root / "annotations.tsv"),
            "threads": request.threads,
            "sensitivity": request.sensitivity,
            "max_evalue": request.max_evalue,
            "min_identity": request.min_identity,
            "min_coverage": request.min_coverage,
            "timeout": SEARCH_TIMEOUT_SECONDS,
        },
        timeout=SEARCH_TIMEOUT_SECONDS + 600,
    )

    by_query: dict[str, list[dict[str, Any]]] = {name: [] for name in proteins}
    for hit in found["hits"]:
        by_query[hit["query"]].append(hit)

    annotated: list[ProteinAnnotation] = []
    for name, sequence in proteins.items():
        chosen = select_hits(
            by_query[name],
            min_identity=request.min_identity,
            min_coverage=request.min_coverage,
            max_evalue=request.max_evalue,
            max_hits=request.max_hits,
            exclude_identical=request.exclude_identical,
        )
        terms = transfer(chosen)
        annotated.append(
            ProteinAnnotation(
                id=name,
                length=len(sequence),
                sources=[
                    Source(
                        accession=h["target"],
                        name=h["annotations"]["name"],
                        organism=h["annotations"]["organism"],
                        identity=h["identity"],
                        coverage=h["coverage"],
                        target_coverage=h["target_coverage"],
                        evalue=h["evalue"],
                        bits=h["bits"],
                    )
                    for h in chosen
                ],
                **{
                    category: [Term(**t) for t in terms[category]] for category in terms
                },
            )
        )

    unmatched = [p for p in annotated if not p.sources]
    if unmatched:
        shown = ", ".join(p.id for p in unmatched[:5])
        more = f" and {len(unmatched) - 5} more" if len(unmatched) > 5 else ""
        ctx.note(
            f"{len(unmatched)} of {len(annotated)} proteins have no hit that meets the "
            f"thresholds ({shown}{more})."
        )
    thresholds = {
        "min_identity": request.min_identity,
        "min_coverage": request.min_coverage,
        "max_evalue": request.max_evalue,
        "max_hits": request.max_hits,
        "exclude_identical": request.exclude_identical,
        "sensitivity": request.sensitivity,
    }
    record = {
        "database": pinned["logical_id"],
        "database_version": pinned["version"],
        "database_identity": identity,
        "database_source_md5": pinned.get("source_archive_md5"),
        "settings": thresholds,
        "proteins": [
            {
                "id": p.id,
                "length": p.length,
                "sources": [asdict(s) for s in p.sources],
                **{
                    category: [asdict(t) for t in getattr(p, category)]
                    for category in (
                        "ec",
                        "go_function",
                        "go_process",
                        "go_component",
                        "location",
                    )
                },
            }
            for p in annotated
        ],
        "engine_versions": found["versions"],
    }
    return AnnotateResult(
        database=pinned["logical_id"],
        database_version=pinned["version"],
        database_identity=identity,
        proteins=annotated,
        annotated=len(annotated) - len(unmatched),
        unmatched=len(unmatched),
        thresholds=thresholds,
        record=ctx.keep_json(record),
        engine_versions=found["versions"],
    )
