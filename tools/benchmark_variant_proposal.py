"""Benchmark `protein.score_substitutions` against measured variants, retrospectively.

Run it with the esm environment configured:

    ALAMBIQ_ENV_ESM=/path/to/esm/bin/python python tools/benchmark_variant_proposal.py \
        --studies /path/to/alambiq-studies --out docs/methods/variant-proposal-benchmark.json

The datasets are three published sets of measured variants:

* Bacillus subtilis Lipase A, the thermostability (T50) of 2,172 single substitutions
  (Nutschel et al. 2020), shipped with the package. Positions are numbered from the
  precursor, 31 more than the mature protein.
* Human glucokinase, an activity score (190 single substitutions) and an abundance
  score (186 single substitutions) (Gersing et al. 2023, 2024), from the studies repository.
* An amine transaminase lineage (Wu et al. 2022), the specific activity of 27 variants of
  one to five substitutions, from the repository history at the stated commit. Its variants
  are scored by adding the single-substitution scores of their substitutions.

Each variant is scored with the same capability a user would run, and the ranking it
gives is compared with the measurements. Two baselines are reported: a random ranking
(the mean and spread over 1,000 seeded shuffles) and the BLOSUM62 score of the substitution,
which prefers residues similar to the one replaced and stands for the "conservative" guess
a person without a model would make. Each number is the result of the first run of this
script, without tuning.

The metrics are Spearman's rank correlation between score and measurement with a
bootstrap 95 % interval, and the enrichment of the top tenth by score (at least five
variants) in variants whose measurement is in the best quarter, relative to the quarter
expected by chance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOSUM62 = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def blosum() -> dict[tuple[str, str], int]:
    lines = [line.split() for line in BLOSUM62.strip().splitlines()]
    letters = lines[0]
    return {
        (row[0], letters[i]): int(value)
        for row in lines[1:]
        for i, value in enumerate(row[1:])
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ datasets


def lipase(studies: Path) -> dict:
    manifest = (
        studies / "studies/dataset-locked-methods/bslipa-t50-manifest.yaml"
    ).read_text()
    sequence = re.search(r"AEHNPVVMVHG[A-Z]+", manifest).group(0)
    table = ROOT / "tests/fixtures/variant-benchmark/bslipa-t50-observations.csv"
    variants = []
    for row in csv.DictReader(table.open()):
        mutant = row["mutant"]
        position = int(mutant[1:-1]) - 31
        if sequence[position - 1] != mutant[0]:
            raise ValueError(f"{mutant} does not match the mature sequence")
        variants.append(
            {
                "name": mutant,
                "substitutions": [(mutant[0], position, mutant[-1])],
                "value": float(row["DMS_score"]),
            }
        )
    return {
        "id": "bslipa-t50",
        "title": "Bacillus subtilis Lipase A, T50 (Nutschel et al. 2020)",
        "property": "thermostability (T50)",
        "higher_is_better": True,
        "sequence": sequence,
        "variants": variants,
        "source": {"file": str(table.relative_to(ROOT)), "sha256": sha(table)},
        "scoring": "single substitution",
    }


def glucokinase(studies: Path) -> list[dict]:
    base = studies / "examples/applications/24-glucokinase-variants"
    reference = "".join((base / "reference.fasta").read_text().splitlines()[1:])
    out = []
    values: dict[str, dict[str, float]] = {"activity": {}, "abundance": {}}
    table = base / "quantitative/observations.csv"
    for row in csv.DictReader(table.open()):
        if row["status"] == "measured" and row["value"] != "":
            values[row["metric"]][row["sample_id"]] = float(row["value"])
    for metric in ("activity", "abundance"):
        variants = []
        for name, value in values[metric].items():
            wild, position, mutant = name[0], int(name[1:-1]), name[-1]
            if reference[position - 1] != wild:
                raise ValueError(f"{name} does not match the reference")
            variants.append(
                {
                    "name": name,
                    "substitutions": [(wild, position, mutant)],
                    "value": value,
                }
            )
        out.append(
            {
                "id": f"glucokinase-{metric}",
                "title": f"Human glucokinase, {metric} (Gersing et al. 2023, 2024)",
                "property": f"{metric} score",
                "higher_is_better": True,
                "sequence": reference,
                "variants": variants,
                "source": {
                    "file": str(table.relative_to(studies)),
                    "sha256": sha(table),
                },
                "scoring": "single substitution",
            }
        )
    return out


def transaminase(commit: str) -> dict:
    prefix = "campaigns/examples/applications/27-transaminase-engineering/recombination"

    def show(name: str) -> str:
        return subprocess.run(
            ["git", "show", f"{commit}:{prefix}/{name}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    panel = json.loads(show("panel.json"))
    sequences = {c["sample_id"]: c["sequence"] for c in panel["candidates"]}
    wild = sequences["wt"]
    readings: dict[str, list[float]] = {}
    for row in csv.DictReader(show("measurements.csv").splitlines()):
        if row["metric"] == "specific_activity" and row["value"]:
            readings.setdefault(row["sample_id"], []).append(float(row["value"]))
    variants = []
    for name, sequence in sequences.items():
        if name == "wt" or name not in readings or len(sequence) != len(wild):
            continue
        substitutions = [
            (wild[i], i + 1, sequence[i])
            for i in range(len(wild))
            if wild[i] != sequence[i]
        ]
        variants.append(
            {
                "name": name,
                "substitutions": substitutions,
                "value": sum(readings[name]) / len(readings[name]),
            }
        )
    return {
        "id": "atta-transaminase",
        "title": "Amine transaminase lineage, specific activity (Wu et al. 2022)",
        "property": "specific activity",
        "higher_is_better": True,
        "sequence": wild,
        "variants": variants,
        "source": {
            "git_commit": commit,
            "path": f"{prefix}/measurements.csv",
            "sha256": hashlib.sha256((show("measurements.csv")).encode()).hexdigest(),
        },
        "scoring": "sum of single-substitution scores",
    }


# ------------------------------------------------------------------ method and metrics


def score_with_model(dataset: dict, workdir: Path) -> tuple[dict, dict]:
    from alambiq.api import step

    positions = sorted(
        {p for v in dataset["variants"] for _, p, _ in v["substitutions"]}
    )
    fasta = workdir / f"{dataset['id']}.faa"
    fasta.write_text(f">{dataset['id']}\n{dataset['sequence']}\n")
    result = step(
        "protein.score_substitutions",
        protein=str(fasta),
        positions=positions,
        run_dir=workdir / dataset["id"],
    )
    record = json.loads(Path(result.record).read_text())
    return {(r["position"], r["mutant"]): r["llr"] for r in record["rows"]}, record


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else float("nan")


def enrichment(scores: list[float], measured: list[float]) -> dict:
    """Best-quarter measured variants among the best tenth by score, against 25 %."""
    n = len(scores)
    top = max(5, round(0.1 * n))
    cutoff = sorted(measured, reverse=True)[max(0, math.ceil(0.25 * n) - 1)]
    order = sorted(range(n), key=lambda i: -scores[i])[:top]
    hits = sum(measured[i] >= cutoff for i in order)
    return {
        "top": top,
        "hits": hits,
        "fraction": hits / top,
        "enrichment": hits / top / 0.25,
    }


def bootstrap_interval(
    scores: list[float], measured: list[float], seed: int = 0
) -> list[float]:
    rng = random.Random(seed)
    n = len(scores)
    values = []
    for _ in range(2000):
        pick = [rng.randrange(n) for _ in range(n)]
        r = spearman([scores[i] for i in pick], [measured[i] for i in pick])
        if not math.isnan(r):
            values.append(r)
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]]


def evaluate(dataset: dict, scorer, name: str) -> dict:
    sign = 1 if dataset["higher_is_better"] else -1
    measured = [sign * v["value"] for v in dataset["variants"]]
    scores = [scorer(v) for v in dataset["variants"]]
    return {
        "method": name,
        "spearman": spearman(scores, measured),
        "spearman_95_interval": bootstrap_interval(scores, measured),
        "top_tenth": enrichment(scores, measured),
    }


def random_baseline(dataset: dict, shuffles: int = 1000, seed: int = 7) -> dict:
    sign = 1 if dataset["higher_is_better"] else -1
    measured = [sign * v["value"] for v in dataset["variants"]]
    rng = random.Random(seed)
    rhos, enrich = [], []
    for _ in range(shuffles):
        scores = [rng.random() for _ in measured]
        rhos.append(spearman(scores, measured))
        enrich.append(enrichment(scores, measured)["enrichment"])
    mean = sum(rhos) / shuffles
    sd = math.sqrt(sum((r - mean) ** 2 for r in rhos) / (shuffles - 1))
    return {
        "method": "random ranking",
        "spearman": mean,
        "spearman_sd": sd,
        "top_tenth": {"enrichment": sum(enrich) / shuffles},
    }


def markdown_table(results: dict) -> str:
    """The benchmark's results as the table the methods page shows."""
    lines = [
        "| Dataset | Variants | Method | Spearman (95 % interval) | Top tenth: best quarter found |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for dataset in results["datasets"]:
        for method in dataset["methods"]:
            top = method["top_tenth"]
            if "spearman_95_interval" in method:
                low, high = method["spearman_95_interval"]
                rho = f"{method['spearman']:+.2f} ({low:+.2f} to {high:+.2f})"
                found = (
                    f"{top['hits']} of {top['top']} ({top['enrichment']:.1f}x chance)"
                )
            else:
                rho = f"{method['spearman']:+.2f} (SD {method['spearman_sd']:.2f})"
                found = f"{top['enrichment']:.1f}x chance"
            lines.append(
                f"| {dataset['title']} | {dataset['variants']:,} | {method['method']} | "
                f"{rho} | {found} |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--transaminase-commit",
        default="39f0515",
        help="repository commit that still holds the transaminase example",
    )
    parser.add_argument("--only", nargs="*", help="dataset ids to run")
    args = parser.parse_args()

    datasets = [
        lipase(args.studies),
        *glucokinase(args.studies),
        transaminase(args.transaminase_commit),
    ]
    if args.only:
        datasets = [d for d in datasets if d["id"] in args.only]
    matrix = blosum()
    results = []
    with tempfile.TemporaryDirectory(prefix="alambiq-benchmark-") as tmp:
        for dataset in datasets:
            print(
                f"scoring {dataset['id']} ({len(dataset['variants'])} variants)",
                file=sys.stderr,
            )
            table, record = score_with_model(dataset, Path(tmp))

            def model_score(variant, table=table):
                return sum(table[(p, m)] for _, p, m in variant["substitutions"])

            def blosum_score(variant):
                return sum(matrix[(w, m)] for w, _, m in variant["substitutions"])

            results.append(
                {
                    **{
                        k: v
                        for k, v in dataset.items()
                        if k not in ("variants", "sequence")
                    },
                    "variants": len(dataset["variants"]),
                    "positions_scored": len(record["positions"]),
                    "sequence_sha256": hashlib.sha256(
                        dataset["sequence"].encode()
                    ).hexdigest(),
                    "methods": [
                        evaluate(
                            dataset, model_score, "ESM C 300M log-likelihood ratio"
                        ),
                        evaluate(
                            dataset, blosum_score, "BLOSUM62 score of the substitution"
                        ),
                        random_baseline(dataset),
                    ],
                    "model": {
                        "name": record["model"],
                        "revision": record["revision"],
                        "versions": record["versions"],
                    },
                }
            )
    out = {
        "schema": 1,
        "tool": "tools/benchmark_variant_proposal.py",
        "metrics": {
            "spearman": "rank correlation between score and measurement, higher is better",
            "top_tenth": "best-quarter measured variants among the best tenth by score, "
            "relative to the 25 % expected by chance",
        },
        "datasets": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    for d in results:
        for m in d["methods"]:
            print(
                f"{d['id']:24} {m['method']:36} rho={m['spearman']:+.3f}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
