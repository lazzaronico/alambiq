"""Find the family profiles that match a protein, in a named library.

The protein is searched with MMseqs2 against sequence profiles: either the
pinned Pfam 38.2 library, or profiles built from supplied Stockholm
alignments at the time of the search. Every returned alignment is reported as a
separate match, with its span on the protein and on the profile, its bit score
and expectation value, and the fraction of the profile that it covers. A match
is `selected` when it meets the expectation and coverage thresholds and
`filtered` when it does not.

What the result supports: that a region of the protein resembles the seed
alignment of a family, in a named database at a named version.

Limits. The matches are MMseqs2 profile matches: the profiles are built from
the seed alignments, and the thresholds of the request replace the curated
gathering thresholds of Pfam, so a hit can be one that Pfam would not report,
and a Pfam match can be absent from the result. Expectation values depend on the
size of the library searched, so they are comparable within one library only.
Overlapping families are all reported; no best family is chosen. For supplied
profiles with a `CL` line, a note names the selected families that share a
clan. Positions on the protein are one-based and inclusive; profile positions
are counted in the match columns of the profile, not in the columns of the
original alignment. The pinned Pfam library has no clan or entry-type
metadata, so those columns are empty for it. A family match assigns the
matched region to the family of the profile; regions without a profile in the
library are unannotated. A library search verifies the size of each file of the
resource against the record and recomputes its SHA-256 when no receipt of an
earlier verification matches the size and modification time of the file. A
supplied set of profiles holds at most 64 alignments; each alignment has 2 to
10,000 sequences and 5 to 5,000 columns, and all alignments together have at
most 1,000,000 cells. Accessions are unique; the residues are the 20 amino
acids and B, X, Z, with gaps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from alambiq.capabilities.characterize.protein_match_profiles_report import render
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

LIBRARY = "pfam-38.2-profiles"


@dataclass(frozen=True, slots=True)
class MatchRequest:
    """The protein, the profiles to search, and the thresholds that a match must meet.

    protein:
        A path to a FASTA with one protein of 10 to 5,000 residues from the 20
        amino acids and X.
    profiles:
        A path to a Stockholm 1.0 file of 1 to 64 alignments, each with AC, ID
        and DE lines and a unique accession; an optional `CL` line gives the
        clan. Exactly one of this field and `library` is required.
    library:
        The pinned profile library to search instead of `profiles`. The only
        value is `pfam-38.2`.
    evalue:
        Matches above this expectation value are filtered. The search itself
        reports matches up to the same value. Above 0 and at most 1,000,000;
        default 0.001.
    sensitivity:
        MMseqs2 sensitivity (`-s`), 1 to 7.5; default 7.5. A higher value finds
        more distant matches.
    min_profile_coverage:
        Matches covering less than this fraction of the profile are filtered.
        0 to 1; default 0.5.
    alternative_alignments:
        Number of further alignments of the same profile to report, 0 to 20,
        for families that repeat in a protein; default 0.
    """

    protein: str
    profiles: str | None = None
    library: Literal["pfam-38.2"] | None = None
    evalue: float = 0.001
    sensitivity: float = 7.5
    min_profile_coverage: float = 0.5
    alternative_alignments: int = 0


@dataclass(frozen=True, slots=True)
class Match:
    """One alignment of the protein to a profile."""

    accession: str
    name: str
    description: str
    clan_accession: str | None
    entry_type: str | None
    status: Literal["selected", "filtered"]
    reasons: list[str]
    evalue: float
    bits: float
    query_start: int
    query_end: int
    profile_start: int
    profile_end: int
    profile_length: int
    profile_coverage: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """The matches, and the library that was searched."""

    query_id: str
    query_length: int
    library: str
    profiles_searched: int
    matches: list[Match]
    selected: int
    filtered: int
    families_returned: int
    record: str
    engine_versions: dict[str, Any]


@capability(
    "protein.match_profiles",
    version="1.0",
    family="characterize",
    summary="Find the family profiles a protein matches",
    render=render,
    environment="mmseqs",
    resources=(LIBRARY,),
)
def match_profiles(request: MatchRequest, ctx: StepContext) -> MatchResult:
    """Search the profiles and report every alignment, selected or filtered."""
    from alambiq.capabilities.characterize.science.profiles import (
        check_query,
        check_settings,
        parse_hits,
        parse_stockholm,
    )
    from alambiq.capabilities.shared.sequences import parse_fasta

    if (request.profiles is None) == (request.library is None):
        raise ValueError(
            "profiles and library: supply either profiles or library, not both and "
            "not neither"
        )
    check_settings(
        request.evalue,
        request.sensitivity,
        request.min_profile_coverage,
        request.alternative_alignments,
    )
    query_id, sequence = parse_fasta(ctx.read_text(request.protein))
    check_query(sequence)

    query = ctx.workdir / "query.fasta"
    query.write_text(f">q0\n{sequence}\n")
    engine_request: dict[str, Any] = {
        "query": str(query),
        "evalue": request.evalue,
        "sensitivity": request.sensitivity,
        "alternative_alignments": request.alternative_alignments,
    }
    if request.profiles is not None:
        source_text = ctx.read_text(request.profiles)
        catalogue_rows = parse_stockholm(source_text)
        engine_request["stockholm"] = str(ctx.resolve(request.profiles))
        library_name = "supplied Stockholm alignments"
        source = {
            "kind": "stockholm",
            "sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        }
    else:
        root = resources.verify(LIBRARY)
        pinned = resources.spec(LIBRARY)
        catalogue_rows = [
            {
                **{k: v for k, v in row.items() if k != "native_mapped_residues"},
                "clan_accession": row.get("clan_accession"),
                "entry_type": row.get("entry_type"),
            }
            for row in json.loads((root / "catalogue.json").read_text())
        ]
        engine_request["database"] = str(root / "profiles")
        library_name = (
            f"Pfam {pinned['version'].removeprefix('Pfam')} (MMseqs2 profiles)"
        )
        source = {
            "kind": "library",
            "logical_id": pinned["logical_id"],
            "version": pinned["version"],
            "license": pinned["license"],
            "identity": resources.identity(LIBRARY),
        }
    engine_request["profile_count"] = len(catalogue_rows)

    found = ctx.run_in_environment("mmseqs", "mmseqs_profiles:run", engine_request)
    catalogue = {row["accession"]: row for row in catalogue_rows}
    rows = parse_hits(
        found["hits"],
        sequence,
        catalogue,
        evalue=request.evalue,
        min_profile_coverage=request.min_profile_coverage,
        alternatives=request.alternative_alignments,
    )

    selected = [r for r in rows if r["status"] == "selected"]
    filtered = [r for r in rows if r["status"] == "filtered"]
    families = {r["accession"] for r in rows}
    if not rows:
        ctx.note(
            f"No profile in {library_name} matched at an expectation value of "
            f"{request.evalue:g}."
        )
    if filtered:
        ctx.note(
            f"{len(filtered)} alignments were returned but filtered by the "
            "expectation or coverage thresholds."
        )
    clans: dict[str, set[str]] = {}
    for row in selected:
        if row["clan_accession"]:
            clans.setdefault(row["clan_accession"], set()).add(row["accession"])
    shared = {c: sorted(a) for c, a in clans.items() if len(a) > 1}
    if shared:
        listing = "; ".join(
            c + ": " + ", ".join(accessions) for c, accessions in sorted(shared.items())
        )
        ctx.note(
            f"Selected matches of more than one profile belong to one source clan "
            f"({listing})."
        )

    record = {
        "query_id": query_id,
        "query_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "library": library_name,
        "profile_source": source,
        "profiles_searched": len(catalogue_rows),
        "settings": {
            "evalue": request.evalue,
            "sensitivity": request.sensitivity,
            "min_profile_coverage": request.min_profile_coverage,
            "alternative_alignments": request.alternative_alignments,
        },
        "construction": found["construction"],
        "coordinate_convention": (
            "One-based inclusive protein positions; profile positions count the "
            "profile's own match columns, not the original alignment's columns."
        ),
        "matches": rows,
        "engine_versions": found["versions"],
    }
    return MatchResult(
        query_id=query_id,
        query_length=len(sequence),
        library=library_name,
        profiles_searched=len(catalogue_rows),
        matches=[
            Match(
                accession=r["accession"],
                name=r["name"],
                description=r["description"],
                clan_accession=r["clan_accession"],
                entry_type=r["entry_type"],
                status=r["status"],
                reasons=r["reasons"],
                evalue=r["evalue"],
                bits=r["bits"],
                query_start=r["query_start"],
                query_end=r["query_end"],
                profile_start=r["profile_start"],
                profile_end=r["profile_end"],
                profile_length=r["profile_length"],
                profile_coverage=r["profile_coverage"],
            )
            for r in rows
        ],
        selected=len(selected),
        filtered=len(filtered),
        families_returned=len(families),
        record=ctx.keep_json(record),
        engine_versions=found["versions"],
    )
