"""Reading Stockholm alignments and profile search hits.

Pure functions over text and the rows an engine returned. The Stockholm reader
reads what a profile needs (identity metadata, rows, width), skips other
annotation lines and refuses alignment lines that it cannot read.
"""

from __future__ import annotations

import math
import re
from typing import Any

MAX_PROFILES = 64
MAX_CELLS = 1_000_000
PROTEIN = re.compile(r"[ACDEFGHIKLMNPQRSTVWYXBZ.-]+")
QUERY = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]{10,5000}")
IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]+")


def _field(source: str, key: str, *, required: bool) -> str | None:
    values = re.findall(
        r"^#=GF[ \t]+" + key + r"(?:[ \t]+([^\r\n]*))?$", source, re.MULTILINE
    )
    if len(values) > 1 or (values and not values[0].strip()):
        raise ValueError(f"Metadata {key} must occur once and be nonempty")
    if required and not values:
        raise ValueError(f"Each profile requires one {key} metadata field")
    return values[0].strip() if values else None


def parse_stockholm(text: str) -> list[dict[str, Any]]:
    """Describe every alignment in a Stockholm file, refusing what a profile cannot use."""
    chunks = re.split(r"^//\s*$", text, flags=re.MULTILINE)
    if chunks[-1].strip() or not 1 <= len(chunks) - 1 <= MAX_PROFILES:
        raise ValueError(
            f"Supply 1–{MAX_PROFILES} complete Stockholm alignments ending in //"
        )
    rows: list[dict[str, Any]] = []
    cells = 0
    for chunk in chunks[:-1]:
        source = chunk.strip() + "\n//\n"
        if not source.startswith("# STOCKHOLM 1.0\n"):
            raise ValueError("Each profile needs a Stockholm 1.0 header")
        metadata = {
            "AC": _field(source, "AC", required=True),
            "ID": _field(source, "ID", required=True),
            "DE": _field(source, "DE", required=True),
            "CL": _field(source, "CL", required=False),
            "TP": _field(source, "TP", required=False),
        }
        if not IDENTIFIER.fullmatch(metadata["AC"]):
            raise ValueError("Profile accession must be a plain unique identifier")
        if metadata["CL"] is not None and not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,64}", metadata["CL"]
        ):
            raise ValueError("Source clan accession must be a plain identifier")

        aligned: dict[str, str] = {}
        for line in source.splitlines():
            if not line.strip() or line.startswith("#") or line.strip() == "//":
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Cannot read alignment line: {line[:60]!r}")
            aligned[parts[0]] = aligned.get(parts[0], "") + parts[1].upper()
        widths = {len(v) for v in aligned.values()}
        if len(widths) != 1:
            raise ValueError("Every aligned sequence must have the same length")
        width = widths.pop()
        if not 2 <= len(aligned) <= 10000 or not 5 <= width <= 5000:
            raise ValueError(
                "Each profile needs 2–10000 aligned sequences and 5–5000 columns"
            )
        if any(not PROTEIN.fullmatch(v) for v in aligned.values()):
            raise ValueError(
                "Profiles accept standard protein residues, B/Z/X and gaps"
            )
        cells += len(aligned) * width
        if cells > MAX_CELLS:
            raise ValueError(
                "profiles: the alignments hold more than 1,000,000 cells in all; "
                "at most 1,000,000 alignment cells are accepted"
            )
        rows.append(
            {
                "accession": metadata["AC"],
                "name": metadata["ID"],
                "description": metadata["DE"],
                "clan_accession": metadata["CL"],
                "entry_type": metadata["TP"],
                "sequence_count": len(aligned),
                "alignment_columns": width,
                "ambiguous_residues": sum(
                    c in "BZX" for v in aligned.values() for c in v
                ),
            }
        )
    if len({r["accession"] for r in rows}) != len(rows):
        raise ValueError("Duplicate profile accession")
    return rows


def check_query(sequence: str) -> None:
    if not QUERY.fullmatch(sequence):
        raise ValueError("Query requires 10–5000 standard protein residues (X allowed)")


def check_sensitivity(value: float) -> None:
    """Refuse an MMseqs2 sensitivity outside 1 to 7.5 (inclusive) or not finite."""
    if not math.isfinite(value) or not 1.0 <= value <= 7.5:
        raise ValueError("sensitivity must be between 1 and 7.5")


def check_settings(
    evalue: float, sensitivity: float, min_profile_coverage: float, alternatives: int
) -> None:
    for name, value, low, high in (
        ("evalue", evalue, 0.0, 1e6),
        ("sensitivity", sensitivity, 1.0, 7.5),
        ("min_profile_coverage", min_profile_coverage, 0.0, 1.0),
    ):
        if (
            not math.isfinite(value)
            or not low <= value <= high
            or (name == "evalue" and value == 0)
        ):
            raise ValueError(f"{name} must be between {low:g} and {high:g}")
    if not 0 <= alternatives <= 20:
        raise ValueError("alternative_alignments must be an integer between 0 and 20")


def parse_hits(
    rows: list[list[str]],
    sequence: str,
    catalogue: dict[str, dict[str, Any]],
    *,
    evalue: float,
    min_profile_coverage: float,
    alternatives: int,
) -> list[dict[str, Any]]:
    """Turn the engine's rows into matches, validating every coordinate.

    One-based and inclusive on both sides. Each alignment is its own match, and
    is `filtered` if its expectation value or the share of the profile it covers
    misses the threshold.
    """
    seen: dict[str, list[dict[str, Any]]] = {}
    for parts in rows:
        if len(parts) != 10 or parts[0] != "q0" or parts[1] not in catalogue:
            raise ValueError("Malformed or unknown native profile hit")
        _, accession, ev, bits, qs, qe, ts, te, ql, tl = parts
        ev, bits = float(ev), float(bits)
        qs, qe, ts, te, ql, tl = map(int, (qs, qe, ts, te, ql, tl))
        if not (
            all(math.isfinite(v) and v >= 0 for v in (ev, bits))
            and 1 <= qs <= qe <= ql == len(sequence)
            and 1 <= ts <= te <= tl
        ):
            raise ValueError("Invalid profile scores, lengths or one-based coordinates")
        previous = seen.setdefault(accession, [])
        if len(previous) >= 1 + alternatives or any(
            (qs, qe, ts, te)
            == (r["query_start"], r["query_end"], r["profile_start"], r["profile_end"])
            or tl != r["profile_length"]
            for r in previous
        ):
            raise ValueError(
                "Duplicate, excessive or inconsistent native profile alignments"
            )
        coverage = (te - ts + 1) / tl
        reasons = []
        if ev > evalue:
            reasons.append("evalue")
        if coverage < min_profile_coverage:
            reasons.append("profile_coverage")
        previous.append(
            {
                "evalue": ev,
                "bits": bits,
                "query_start": qs,
                "query_end": qe,
                "profile_start": ts,
                "profile_end": te,
                "profile_length": tl,
                "profile_coverage": coverage,
                "status": "filtered" if reasons else "selected",
                "reasons": reasons,
            }
        )
    return [
        {**catalogue[accession], **hit}
        for accession in catalogue
        for hit in sorted(
            seen.get(accession, []),
            key=lambda r: (r["query_start"], r["query_end"], r["profile_start"]),
        )
    ]
