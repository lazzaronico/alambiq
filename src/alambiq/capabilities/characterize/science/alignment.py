"""Inspect supplied protein alignments as given."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

INFORMATIVE = set("ACDEFGHIKLMNPQRSTVWYOU")


def alignment_selection_key(text, full_sequence_context):
    """Link views only when aligned bytes and full-sequence identities agree."""
    return hashlib.sha256(
        json.dumps([text, full_sequence_context], sort_keys=True).encode()
    ).hexdigest()


def inspect_alignment(text, format, *, full_sequence_context=None):
    if format not in {"fasta", "a3m"}:
        raise ValueError(f"format is {format!r}; it must be fasta or a3m")
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line or (format == "a3m" and line.startswith("#")):
            continue
        if line.startswith(">"):
            header = line[1:].strip()
            if not header:
                raise ValueError("alignment: a header line has no identifier")
            records.append([header, ""])
        elif not records:
            raise ValueError("alignment: sequence data precedes the first header")
        else:
            records[-1][1] += line
    if not 1 <= len(records) <= 200:
        raise ValueError(
            f"alignment: the file holds {len(records)} rows; 1–200 alignment rows "
            "are accepted at a time"
        )
    rows, identifiers, seen = [], set(), {}
    for header, raw in records:
        identifier = header.split()[0]
        if identifier in identifiers:
            raise ValueError(
                f"alignment: identifier {identifier!r} appears twice; identifiers "
                "must be unique"
            )
        identifiers.add(identifier)
        if not raw or not re.fullmatch(
            r"[ACDEFGHIKLMNPQRSTVWYBXZJUOacdefghiklmnpqrstvwybxzjuo.-]+", raw
        ):
            raise ValueError(
                f"alignment: row {identifier!r} is empty or contains an unsupported "
                "residue"
            )
        if format == "fasta" and "." in raw:
            raise ValueError(
                f"alignment: row {identifier!r} contains a dot; aligned FASTA uses "
                "'-' for gaps, and dots require A3M"
            )
        if not rows and (raw != raw.upper() or "-" in raw or "." in raw):
            raise ValueError(
                f"alignment: the first row {identifier!r} of the "
                f"{'A3M' if format == 'a3m' else 'aligned FASTA'} input contains "
                "gaps or lowercase letters; the first row must be an ungapped "
                "uppercase query"
            )
        aligned, positions, insertions, sequence = [], [], [], []
        for residue in raw:
            if format == "a3m" and residue == ".":
                continue
            if residue != "-":
                sequence.append(residue.upper())
            if format == "a3m" and residue.islower():
                if insertions and insertions[-1]["after_column"] == len(aligned):
                    insertions[-1]["sequence"] += residue.upper()
                else:
                    insertions.append(
                        {
                            "after_column": len(aligned),
                            "sequence": residue.upper(),
                            "start": len(sequence),
                        }
                    )
            else:
                aligned.append(residue.upper())
                positions.append(len(sequence) if residue != "-" else None)
        aligned_text = "".join(aligned)
        if not 1 <= len(aligned) <= 5000 or len(rows) * len(aligned) > 250_000:
            raise ValueError(
                f"alignment: row {identifier!r} exceeds the limits of 5000 columns "
                "and 250,000 aligned cells"
            )
        if rows and len(aligned) != len(rows[0]["aligned"]):
            raise ValueError(
                f"alignment: row {identifier!r} has {len(aligned)} match columns, "
                f"not {len(rows[0]['aligned'])}; all rows must have the same "
                "match-column count"
            )
        if not sequence:
            raise ValueError(
                f"alignment: row {identifier!r} has no residue; every row must "
                "contain a protein residue"
            )
        rows.append(
            {
                "id": identifier,
                "description": header,
                "source_alignment": raw,
                "aligned": aligned_text,
                "sequence": "".join(sequence),
                "positions": positions,
                "insertions": insertions,
                "duplicate_of": seen.get(aligned_text),
            }
        )
        seen.setdefault(aligned_text, identifier)
    if len(rows) * len(rows[0]["aligned"]) > 250_000:
        raise ValueError(
            "alignment: the alignment exceeds the limit of 250,000 aligned cells"
        )
    if full_sequence_context is not None:
        if (
            not isinstance(full_sequence_context, dict)
            or set(full_sequence_context) != identifiers
        ):
            raise ValueError(
                "full_sequences: the identifiers must cover exactly the supplied rows"
            )
        for row in rows:
            context = full_sequence_context[row["id"]]
            if not isinstance(context, dict) or set(context) != {"sequence", "start"}:
                raise ValueError(
                    f"full_sequences: the entry for {row['id']!r} needs a sequence "
                    "and a one-based start"
                )
            full, start = context["sequence"], context["start"]
            if (
                not isinstance(full, str)
                or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]{1,5000}", full)
                or type(start) is not int
                or start < 1
                or full[start - 1 : start - 1 + len(row["sequence"])] != row["sequence"]
            ):
                raise ValueError(
                    f"full_sequences: row {row['id']!r} does not match its "
                    "full-protein interval"
                )
            row["full_sequence"] = full
            row["full_sequence_positions"] = [
                None if p is None else p + start - 1 for p in row["positions"]
            ]
    columns = []
    for i, residues in enumerate(zip(*(row["aligned"] for row in rows), strict=True)):
        counts = Counter(residue for residue in residues if residue in INFORMATIVE)
        total = sum(counts.values())
        ordered = sorted(counts, key=lambda residue: (-counts[residue], residue))
        columns.append(
            {
                "column": i + 1,
                "query_position": rows[0]["positions"][i],
                "gap_fraction": residues.count("-") / len(rows),
                "ambiguous_fraction": sum(
                    r not in INFORMATIVE and r != "-" for r in residues
                )
                / len(rows),
                "informative_count": total,
                "counts": dict(sorted(counts.items())),
                "consensus": ordered[0] if ordered else None,
                "consensus_fraction": counts[ordered[0]] / total if total else None,
                "entropy_bits": -sum(
                    (n / total) * math.log2(n / total) for n in counts.values()
                )
                if total
                else None,
            }
        )
    return {
        "schema_version": "0.1",
        "format": format,
        "query_id": rows[0]["id"],
        "row_count": len(rows),
        "column_count": len(columns),
        "unique_aligned_rows": len(seen),
        "rows": rows,
        "columns": columns,
        "source_text": text,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "selection_key": alignment_selection_key(text, full_sequence_context),
        "position_basis": "One-based ungapped positions within each supplied row, including A3M insertions. Search rows may be fragments.",
        "method": "Unweighted row counts; gaps and B/J/X/Z excluded from residue entropy/consensus; U/O retained; ties alphabetic; duplicate match rows retained.",
    }
