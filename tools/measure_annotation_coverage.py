"""Measure how many protein families `protein.annotate_function` can annotate.

Run it with the mmseqs environment configured, `mmseqs` on the path, and the UniRef50 and
Swiss-Prot resources installed:

    ALAMBIQ_ENV_MMSEQS=/path/to/python python tools/measure_annotation_coverage.py \
        --out annotation-coverage.json

A random sample of UniRef50 cluster representatives is annotated from Swiss-Prot at the
default thresholds of the capability and at looser ones. UniRef50 holds one representative
per cluster of proteins that are at least 50 % identical, so the sample is drawn over
families of related proteins, not over proteins: a family that many proteins share counts
once. The result is the share of families with a Swiss-Prot relative at the stated
thresholds, and the share of those that receive each kind of annotation. A single protein
of a well-studied organism usually has a relative more often than a random family does.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

from alambiq import api
from alambiq.core import resources

ACCEPTED = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]{10,5000}")
SETTINGS = {
    "default thresholds": {},
    "looser thresholds": {
        "min_identity": 0.3,
        "min_coverage": 0.5,
        "max_evalue": 0.001,
        "sensitivity": 7.5,
    },
}


def sample(database: Path, count: int, seed: int, work: Path) -> Path:
    keys = [
        line.split("\t")[0] for line in (database.parent / "uniref50-db.index").open()
    ]
    chosen = random.Random(seed).sample(keys, count)
    (work / "keys.txt").write_text("\n".join(chosen) + "\n")
    for command in (
        ["createsubdb", str(work / "keys.txt"), str(database), str(work / "sub")],
        ["convert2fasta", str(work / "sub"), str(work / "sample.faa")],
    ):
        subprocess.run(["mmseqs", *command, "-v", "1"], check=True, capture_output=True)
    records, name, sequence = [], None, ""
    for line in (work / "sample.faa").read_text().splitlines() + [">"]:
        if line.startswith(">"):
            if name and ACCEPTED.fullmatch(sequence):
                records.append(f">{name}\n{sequence}\n")
            name, sequence = line[1:].split()[0] if len(line) > 1 else None, ""
        else:
            sequence += line.strip()
    path = work / "queries.faa"
    path.write_text("".join(records[:2000]))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    database = resources.verify("uniref50-2026_02") / "uniref50-db"
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        queries = sample(database, min(arguments.sample, 2000), arguments.seed, work)
        steps = [
            {
                "id": label.replace(" ", "_"),
                "capability": "protein.annotate_function",
                "params": {
                    "proteins": str(queries),
                    "threads": arguments.threads,
                    **extra,
                },
            }
            for label, extra in SETTINGS.items()
        ]
        outcome = api.run({"id": "coverage", "steps": steps}, run_dir=work / "run")
        if not outcome.ok:
            raise SystemExit([s.error for s in outcome.steps.values() if s.error])
        result = {
            "sample": min(arguments.sample, 2000),
            "seed": arguments.seed,
            "settings": {},
        }
        for step, (label, extra) in zip(steps, SETTINGS.items(), strict=True):
            proteins = outcome.outputs(step["id"])["proteins"]
            n = len(proteins)
            with_source = [p for p in proteins if p["sources"]]
            result["settings"][label] = {
                "parameters": extra,
                "proteins": n,
                "with_a_source": len(with_source),
                "with_ec": sum(bool(p["ec"]) for p in proteins),
                "with_go_function": sum(bool(p["go_function"]) for p in proteins),
                "with_location": sum(bool(p["location"]) for p in proteins),
            }
    arguments.out.write_text(json.dumps(result, indent=2) + "\n")
    for label, c in result["settings"].items():
        n = c["proteins"]
        print(
            f"{label:20s} {n} proteins: source {c['with_a_source'] / n:.1%}, "
            f"EC {c['with_ec'] / n:.1%}, GO function {c['with_go_function'] / n:.1%}, "
            f"location {c['with_location'] / n:.1%}"
        )


if __name__ == "__main__":
    main()
