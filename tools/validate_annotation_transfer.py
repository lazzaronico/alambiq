"""Measure how well `protein.annotate_function` transfers EC numbers and locations.

Run it with the mmseqs environment configured and the Swiss-Prot resource installed:

    ALAMBIQ_ENV_MMSEQS=/path/to/python python tools/validate_annotation_transfer.py \
        --fasta uniprot_sprot.fasta --out annotation-transfer.json

`uniprot_sprot.fasta` is the file the resource was built from (see the `build` field of
`alambiq resource show swissprot-2026_03`).

A random sample of Swiss-Prot enzymes (entries with an EC number, 10 to 5,000 residues) is
annotated with `exclude_identical`, so that no protein is annotated from its own entry. The
top transferred EC number is compared with the EC numbers of the entry, in full and at the
first three levels, and the top transferred location with the locations of the entry. The
results are grouped by the identity of the strongest source. A protein without a source is
counted apart.

The test is within Swiss-Prot. Close relatives of the sampled proteins are present in the
database, as they are for most proteins a user annotates, but the sample is not a held-out
set of proteins unknown to the database.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tempfile
from pathlib import Path

from alambiq import api
from alambiq.core import resources
from alambiq.engines.swissprot_annotations import SEPARATOR

DATABASE = "swissprot-2026_03"
BANDS = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            parts = line[1:].split()[0].split("|")
            name = parts[1] if len(parts) == 3 else parts[0]
            sequences[name] = ""
        elif name:
            sequences[name] += line.strip()
    return sequences


def sample(table: Path, sequences: dict[str, str], count: int, seed: int):
    csv.field_size_limit(10**9)
    truth: dict[str, dict[str, list[str]]] = {}
    with table.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            accession = row["accession"]
            if row["ec"] and 10 <= len(sequences.get(accession, "")) <= 5000:
                truth[accession] = {
                    "ec": row["ec"].split(SEPARATOR),
                    "location": [v for v in row["location"].split(SEPARATOR) if v],
                }
    chosen = random.Random(seed).sample(sorted(truth), count)
    return {accession: truth[accession] for accession in chosen}


def wilson(correct: int, total: int) -> list[float]:
    """The 95 % Wilson interval of a proportion."""
    if total == 0:
        return [0.0, 0.0]
    p, z = correct / total, 1.96
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return [round(centre - half, 4), round(centre + half, 4)]


def level3(ec: str) -> str:
    return ".".join(ec.split(".")[:3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    table = resources.verify(DATABASE) / "annotations.tsv"
    truth = sample(table, read_fasta(arguments.fasta), arguments.sample, arguments.seed)
    sequences = read_fasta(arguments.fasta)
    with tempfile.TemporaryDirectory() as directory:
        query = Path(directory) / "proteins.faa"
        query.write_text("".join(f">{a}\n{sequences[a]}\n" for a in truth))
        step = {
            "id": "annotate",
            "capability": "protein.annotate_function",
            "params": {
                "proteins": str(query),
                "exclude_identical": True,
                "threads": arguments.threads,
            },
        }
        outcome = api.run(
            {"id": "validate", "steps": [step]}, run_dir=Path(directory) / "run"
        )
        if not outcome.ok:
            raise SystemExit([s.error for s in outcome.steps.values() if s.error])
        proteins = outcome.outputs("annotate")["proteins"]

    bands = {
        f"{low:.1f}-{min(high, 1.0):.1f}": {
            "n": 0,
            "ec": 0,
            "ec3": 0,
            "loc_n": 0,
            "loc": 0,
        }
        for low, high in BANDS
    }
    without_source = 0
    for protein in proteins:
        if not protein["sources"]:
            without_source += 1
            continue
        identity = protein["sources"][0]["identity"]
        key = next(
            f"{low:.1f}-{min(high, 1.0):.1f}"
            for low, high in BANDS
            if low <= identity < high
        )
        cell, real = bands[key], truth[protein["id"]]
        cell["n"] += 1
        top_ec = protein["ec"][0]["term"] if protein["ec"] else None
        cell["ec"] += top_ec in real["ec"]
        cell["ec3"] += top_ec is not None and level3(top_ec) in {
            level3(e) for e in real["ec"]
        }
        if real["location"]:
            cell["loc_n"] += 1
            top = protein["location"][0]["term"] if protein["location"] else None
            cell["loc"] += top in real["location"]
    for cell in bands.values():
        cell["ec_interval"] = wilson(cell["ec"], cell["n"])
        cell["ec3_interval"] = wilson(cell["ec3"], cell["n"])
        cell["loc_interval"] = wilson(cell["loc"], cell["loc_n"])
    result = {
        "database": DATABASE,
        "sample": len(truth),
        "seed": arguments.seed,
        "without_source": without_source,
        "bands": bands,
    }
    arguments.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(truth)} proteins, {without_source} without a source")
    print("identity   n   EC full   EC 3 levels   location (n)")
    for name, c in bands.items():
        if c["n"]:
            print(
                f"{name}  {c['n']:4d}  {c['ec'] / c['n']:7.1%}  {c['ec3'] / c['n']:11.1%}  "
                f"{(c['loc'] / c['loc_n'] if c['loc_n'] else 0):9.1%} ({c['loc_n']})"
            )


if __name__ == "__main__":
    main()
