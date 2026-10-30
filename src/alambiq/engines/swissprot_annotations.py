"""The annotation table of the Swiss-Prot resource.

Builds a tab-separated table from the UniProt flat file `uniprot_sprot.dat.gz` and reads it
back. Uses the standard library only, so it runs in the MMseqs2 environment and anywhere
else.

The table has one row per reviewed entry, ordered by accession, with the columns in
`COLUMNS`. Terms that repeat in a cell are separated by `; `. A Gene Ontology term is
written `GO:ID|name`. Subcellular locations are the first term of each location sentence
of the entry, without evidence codes, notes and isoform prefixes.

Build the table with:

    python -m alambiq.engines.swissprot_annotations uniprot_sprot.dat.gz annotations.tsv
"""

from __future__ import annotations

import gzip
import re
import sys
from collections.abc import Iterator
from pathlib import Path

COLUMNS = (
    "accession",
    "name",
    "organism",
    "length",
    "ec",
    "go_function",
    "go_process",
    "go_component",
    "location",
)
SEPARATOR = "; "
_ASPECT = {"F": "go_function", "P": "go_process", "C": "go_component"}
_EVIDENCE = re.compile(r"\s*\{[^}]*\}")
_ISOFORM = re.compile(r"^\[?Isoform [^\]:]*\]?:\s*")
_EC = re.compile(r"EC=([0-9]+\.[0-9n\-]+\.[0-9n\-]+\.[0-9n\-]+)")
_NAME = re.compile(r"RecName: Full=([^;{]+)")
_LENGTH = re.compile(r"^SQ\s+SEQUENCE\s+(\d+) AA;")
_LOCATION_TOPIC = re.compile(r"-.-\s+SUBCELLULAR LOCATION:")


def locations(text: str) -> list[str]:
    """The subcellular locations named in the text of a SUBCELLULAR LOCATION comment."""
    text = _EVIDENCE.sub("", text)
    text = text.split("Note=")[0]
    found: list[str] = []
    for sentence in re.split(r"\.(?:\s+|$)", text):
        sentence = _ISOFORM.sub("", sentence.strip())
        term = sentence.split(";")[0].strip()
        if term and term not in found:
            found.append(term)
    return found


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def entries(lines: Iterator[str]) -> Iterator[dict[str, str]]:
    """The rows of the table, one per entry of the flat file."""
    row: dict[str, list[str]] = {}
    organism: list[str] = []
    comment: list[str] | None = None
    description: list[str] = []
    for line in lines:
        code = line[:2]
        if code == "ID":
            row = {column: [] for column in COLUMNS}
            organism, description, comment = [], [], None
        elif code == "AC" and not row["accession"]:
            row["accession"] = [line[5:].split(";")[0].strip()]
        elif code == "DE":
            description.append(line[5:].strip())
        elif code == "OS":
            organism.append(line[5:].strip())
        elif code == "DR" and line[5:].startswith("GO;"):
            parts = [p.strip() for p in line[5:].rstrip(".\n").split(";")]
            if len(parts) >= 3 and parts[2][:2] in ("F:", "P:", "C:"):
                row[_ASPECT[parts[2][0]]].append(f"{parts[1]}|{parts[2][2:]}")
        elif code == "CC":
            body = line[5:]
            topic = _LOCATION_TOPIC.match(body)
            if comment is not None and body.startswith("  ") and not topic:
                comment.append(body.strip())
                continue
            if comment is not None:
                row["location"] += locations(" ".join(comment))
            comment = [body[topic.end() :].strip()] if topic else None
        elif code == "SQ":
            match = _LENGTH.match(line)
            row["length"] = [match.group(1)] if match else []
        elif code == "//":
            if comment is not None:
                row["location"] += locations(" ".join(comment))
            text = " ".join(description)
            name = _NAME.search(text)
            row["name"] = [name.group(1).strip()] if name else []
            row["ec"] = _unique(_EC.findall(text))
            row["organism"] = [" ".join(organism).rstrip(".")]
            yield {
                column: SEPARATOR.join(_unique(row[column]))
                if column not in ("accession", "name", "organism", "length")
                else "".join(row[column])
                for column in COLUMNS
            }
            row = {}


def build(source: Path, output: Path) -> int:
    """Write the table for the flat file `source`; the number of rows."""
    count = 0
    with (
        gzip.open(source, "rt", encoding="utf-8") as handle,
        output.open("w", encoding="utf-8") as out,
    ):
        out.write("\t".join(COLUMNS) + "\n")
        rows = sorted(entries(handle), key=lambda r: r["accession"])
        for entry in rows:
            out.write("\t".join(entry[c] for c in COLUMNS) + "\n")
            count += 1
    return count


def read(path: Path, accessions: set[str]) -> dict[str, dict[str, object]]:
    """The rows of the accessions in the table; absent accessions are not in the result."""
    found: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if tuple(header) != COLUMNS:
            raise ValueError(f"{path} has the columns {header}, not {list(COLUMNS)}")
        for line in handle:
            cells = line.rstrip("\n").split("\t")
            if cells[0] not in accessions:
                continue
            row = dict(zip(COLUMNS, cells, strict=True))
            found[cells[0]] = {
                "accession": row["accession"],
                "name": row["name"],
                "organism": row["organism"],
                "length": int(row["length"]),
                **{c: [v for v in row[c].split(SEPARATOR) if v] for c in COLUMNS[4:]},
            }
            if len(found) == len(accessions):
                break
    return found


if __name__ == "__main__":  # pragma: no cover - build command
    if len(sys.argv) != 3:
        raise SystemExit("usage: swissprot_annotations SOURCE.dat.gz OUTPUT.tsv")
    print(build(Path(sys.argv[1]), Path(sys.argv[2])), "entries")
