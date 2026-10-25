"""Build a multiple sequence alignment (MSA) for each protein from UniRef50.

Each protein is searched against the pinned UniRef50 release with iterative
MMseqs2, and its relatives are written as an A3M (an alignment format in which
lowercase letters mark insertions relative to the query) with the protein as
the first row. The other rows are sorted by identifier and then by sequence.
Each alignment is then judged against depth rules before anything uses it: the
number of distinct aligned rows (rows with identical match columns count once,
the query included; at least `minimum_alignment_sequences`; the result field is
`unique_sequences`), the number of diverse sequences (distinct match-column
strings of rows other than the query that cover at least
`minimum_query_coverage` of the protein and are at most 90 % identical to it
over the columns in which the row has no gap; at least
`minimum_diverse_sequences`), and the fraction of the protein covered by the
median row (at least `minimum_query_coverage`). An alignment that meets the
rules has decision `generated`; one that fails has decision `withheld` and is
stored for inspection. The value of `minimum_query_coverage` also constrains
which relatives MMseqs2 retains (see the request field).

What the result supports: which UniRef50 representatives resemble each protein,
aligned to it. An alignment with decision `generated` meets the depth rules of the
request and can be used as input to structure models and co-evolution models.

Limits. The alignment is limited by the database: UniRef50 clusters sequences at
50 % identity and retains one representative per cluster, so close relatives are
collapsed, and a protein with few relatives in UniRef50 has a shallow
alignment. Each protein is aligned on its own (the result lists the proteins as
`chains`) and rows are unpaired across proteins, so the co-evolutionary signal
covers residues within one protein. The number of sequences depends on the
family of the protein: a domain present in many proteins increases the count,
and a novel protein reduces it. The number of sequences and the number of
diverse sequences measure alignment depth; the accuracy of a structure or
co-evolution model on the alignment is determined by running that model.
Near-identical sequences are retained: the MMseqs2 option that sets a maximum
identity does not remove hits close to the query (tested on a toy database with
a hit that differs from the query at 3 or fewer positions), so no such setting
is offered, and an alignment can contain the UniRef50 representative of the
query itself. In a test on a toy database, searches with 1 and 4 threads gave
identical A3M files. The index files of the database total about
118 GB, and a search loads them before searching. The search timeout is 4 hours
for each protein. A request with more than 8 proteins is refused.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from alambiq.capabilities.characterize.sequence_generate_msa_report import render
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

DATABASE = "uniref50-2026_02"
MAX_PROTEINS = 8
SEARCH_TIMEOUT_SECONDS = 4 * 3600


@dataclass(frozen=True, slots=True)
class MsaRequest:
    """The proteins to align and the depth and stringency of the search.

    proteins:
        A path to a FASTA of 1 to 8 proteins, each 10 to 5,000 residues from
        the 20 amino acids and X, with unique identifiers.
    database:
        The pinned sequence database to search. The only value is
        `uniref50-2026_02` (UniRef50 release 2026_02).
    sensitivity:
        MMseqs2 sensitivity (`-s`), 1 to 7.5 (default 7.5).
    evalue:
        A relative is retained only if its expectation value is at most this
        value (above 0 and at most 1,000,000; default 0.001).
    iterations:
        Search iterations, 1 to 5; default 3. Each iteration builds a profile
        from the hits of the previous iteration and finds more distant
        relatives.
    minimum_query_coverage:
        A relative is retained only if the alignment covers at least this
        fraction of the protein and at least this fraction of the relative
        (MMseqs2 `--cov-mode 0`), 0 to 1; default 0.5. The same value is the
        coverage threshold of the diverse sequences and of the median row (see
        the module text).
    maximum_sequences:
        Maximum number of results per protein that pass the MMseqs2 prefilter
        (the option `--max-seqs`), 1 to 10,000; default 4096. MMseqs2 decides
        which relatives are kept when there are more.
    minimum_alignment_sequences, minimum_diverse_sequences:
        The depth that an alignment requires to be not withheld: the number of
        distinct aligned rows (1 to 10,000; default 32) and the number of
        diverse sequences (0 to 10,000; default 16). A diverse sequence covers
        at least `minimum_query_coverage` of the protein and is at most 90 %
        identical to it over the columns in which the row has no gap.
    threads:
        Threads used by MMseqs2, 1 to 64; default 8. The limits state a test
        of the effect of the thread count on the A3M.
    """

    proteins: str
    database: Literal["uniref50-2026_02"] = "uniref50-2026_02"
    sensitivity: float = 7.5
    evalue: float = 0.001
    iterations: int = 3
    minimum_query_coverage: float = 0.5
    maximum_sequences: int = 4096
    minimum_alignment_sequences: int = 32
    minimum_diverse_sequences: int = 16
    threads: int = 8


@dataclass(frozen=True, slots=True)
class Chain:
    """The alignment of one protein and the decision on its depth."""

    id: str
    length: int
    decision: Literal["generated", "withheld"]
    reason_codes: list[str]
    sequences: int
    unique_sequences: int
    diverse_sequences: int
    query_coverage: float
    median_query_coverage: float
    alignment: str
    """The stored A3M, which `sequence.inspect_msa` reads with format `a3m`."""


@dataclass(frozen=True, slots=True)
class MsaResult:
    """The alignment of every protein, and the database that was searched."""

    database: str
    database_version: str
    database_identity: str
    chains: list[Chain]
    generated: int
    withheld: int
    record: str
    engine_versions: dict[str, Any]


@capability(
    "sequence.generate_msa",
    version="1.0",
    family="characterize",
    summary="Build a multiple sequence alignment for each protein from UniRef50",
    render=render,
    environment="mmseqs",
    resources=(DATABASE,),
)
def generate_msa(request: MsaRequest, ctx: StepContext) -> MsaResult:
    """Search the database for each protein and judge the depth of its alignment."""
    from alambiq.capabilities.characterize.science.enzyme import (
        fasta,
        parse_proteins,
    )
    from alambiq.capabilities.characterize.science.msa import assess, canonicalize
    from alambiq.capabilities.characterize.science.profiles import check_query

    proteins = parse_proteins(ctx.read_text(request.proteins), "proteins")
    if len(proteins) > MAX_PROTEINS:
        raise ValueError(f"At most {MAX_PROTEINS} proteins are searched in one step")
    for name, sequence in proteins.items():
        try:
            check_query(sequence)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from None
    for name, value, low, high in (
        ("sensitivity", request.sensitivity, 1.0, 7.5),
        ("evalue", request.evalue, 0.0, 1e6),
        ("minimum_query_coverage", request.minimum_query_coverage, 0.0, 1.0),
    ):
        if not low <= value <= high or (name == "evalue" and value == 0):
            raise ValueError(f"{name} must be between {low:g} and {high:g}")
    for name, value, low, high in (
        ("iterations", request.iterations, 1, 5),
        ("maximum_sequences", request.maximum_sequences, 1, 10000),
        ("minimum_alignment_sequences", request.minimum_alignment_sequences, 1, 10000),
        ("minimum_diverse_sequences", request.minimum_diverse_sequences, 0, 10000),
        ("threads", request.threads, 1, 64),
    ):
        if not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")

    root = resources.verify(DATABASE)
    pinned = resources.spec(DATABASE)
    identity = resources.identity(DATABASE)
    database = str(root / "uniref50-db")

    chains: list[Chain] = []
    versions: dict[str, Any] = {}
    for index, (name, sequence) in enumerate(proteins.items(), start=1):
        query = ctx.workdir / f"chain-{index}.fasta"
        raw = ctx.workdir / f"chain-{index}.raw.a3m"
        query.write_text(fasta({"query": sequence}))
        found = ctx.run_in_environment(
            "mmseqs",
            "mmseqs_msa:run",
            {
                "query": str(query),
                "database": database,
                "output": str(raw),
                "threads": request.threads,
                "sensitivity": request.sensitivity,
                "evalue": request.evalue,
                "iterations": request.iterations,
                "minimum_query_coverage": request.minimum_query_coverage,
                "maximum_sequences": request.maximum_sequences,
            },
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        versions = found["versions"]
        text = canonicalize(raw.read_text(encoding="ascii"))
        verdict = assess(
            text,
            sequence,
            minimum_query_coverage=request.minimum_query_coverage,
            minimum_alignment_sequences=request.minimum_alignment_sequences,
            minimum_diverse_sequences=request.minimum_diverse_sequences,
        )
        stored = ctx.keep_text(text, "text/x-a3m")
        chains.append(
            Chain(
                id=name,
                length=len(sequence),
                decision="generated" if verdict["quality_passed"] else "withheld",
                reason_codes=verdict["reason_codes"],
                sequences=verdict["sequence_count"],
                unique_sequences=verdict["unique_sequence_count"],
                diverse_sequences=verdict["diverse_sequence_count"],
                query_coverage=verdict["query_coverage_fraction"],
                median_query_coverage=verdict["median_query_coverage_fraction"],
                alignment=str(stored.path),
            )
        )

    withheld = [c for c in chains if c.decision == "withheld"]
    if withheld:
        ctx.note(
            f"{len(withheld)} of {len(chains)} alignments were withheld as too shallow "
            f"({', '.join(c.id for c in withheld)})."
        )
    record = {
        "database": pinned["logical_id"],
        "database_version": pinned["version"],
        "database_identity": identity,
        "database_source_md5": pinned.get("source_archive_md5"),
        "settings": {
            name: getattr(request, name)
            for name in (
                "sensitivity",
                "evalue",
                "iterations",
                "minimum_query_coverage",
                "maximum_sequences",
                "minimum_alignment_sequences",
                "minimum_diverse_sequences",
            )
        },
        "pairing": "unpaired-per-chain",
        "chains": [
            {
                **{
                    k: getattr(c, k) for k in c.__dataclass_fields__ if k != "alignment"
                },
                "alignment": c.alignment,
                "query_sha256": hashlib.sha256(proteins[c.id].encode()).hexdigest(),
            }
            for c in chains
        ],
        "engine_versions": versions,
    }
    return MsaResult(
        database=pinned["logical_id"],
        database_version=pinned["version"],
        database_identity=identity,
        chains=chains,
        generated=len(chains) - len(withheld),
        withheld=len(withheld),
        record=ctx.keep_json(record),
        engine_versions=versions,
    )
