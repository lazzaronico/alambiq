"""Test `protein.predict_melting_temperature` on the held-out test set of TemBERTure.

Run it with the thermostability environment configured and the `temberture-tm` resource
installed:

    ALAMBIQ_ENV_THERMOSTABILITY=/path/to/python python tools/validate_melting_temperature.py \
        --test TemBERTureTest_reg.txt.bz2 --out melting-temperature-validation.json

The test file is the regression test set of the TemBERTure authors
(`data/TemBERTure_reg.tar.gz` of github.com/ibmm-unibe-ch/TemBERTure), a table with a
measured melting temperature `Tm` for each sequence. The authors held these sequences out of
training and split their data by clusters of sequences at 50 % identity. The script runs the
capability on every sequence that it accepts (10 to 5,000 residues from the 20 amino acids and
X) and reports, for the mean of the three replicas and for each replica, the Pearson and
Spearman correlations with the measured value, the root-mean-square error and the mean absolute
error in degrees Celsius, and the share of predictions between 60 and 80 degrees Celsius.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import io
import json
import math
import re
import tempfile
from pathlib import Path

from alambiq import api

ACCEPTED = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]{10,5000}")


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    return cov / math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def metrics(predicted: list[float], measured: list[float]) -> dict[str, float]:
    errors = [p - m for p, m in zip(predicted, measured, strict=True)]
    return {
        "pearson": round(pearson(predicted, measured), 3),
        "spearman": round(pearson(ranks(predicted), ranks(measured)), 3),
        "rmse_c": round(math.sqrt(sum(e * e for e in errors) / len(errors)), 2),
        "mae_c": round(sum(abs(e) for e in errors) / len(errors), 2),
        "share_between_60_and_80_c": round(
            sum(60 <= p <= 80 for p in predicted) / len(predicted), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()

    opener = bz2.open if arguments.test.suffix == ".bz2" else open
    with opener(arguments.test, "rt") as handle:
        rows = list(csv.DictReader(io.StringIO(handle.read())))
    kept = [
        (f"p{i:05d}", row["Sequence"].upper(), float(row["Tm"]))
        for i, row in enumerate(rows)
        if ACCEPTED.fullmatch(row["Sequence"].upper())
    ]
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        fasta = work / "test.faa"
        fasta.write_text("".join(f">{n}\n{s}\n" for n, s, _ in kept))
        step = {
            "id": "tm",
            "capability": "protein.predict_melting_temperature",
            "params": {
                "proteins": str(fasta),
                "device": arguments.device,
                "threads": arguments.threads,
            },
        }
        outcome = api.run({"id": "melting", "steps": [step]}, run_dir=work / "run")
        if not outcome.ok:
            raise SystemExit([s.error for s in outcome.steps.values() if s.error])
        proteins = outcome.outputs("tm")["proteins"]
    measured = [tm for _, _, tm in kept]
    result = {
        "test_file_rows": len(rows),
        "sequences_run": len(kept),
        "truncated": sum(p["truncated"] for p in proteins),
        "measured_share_between_60_and_80_c": round(
            sum(60 <= m <= 80 for m in measured) / len(measured), 3
        ),
        "mean": metrics([p["melting_temperature_c"] for p in proteins], measured),
        "replicas": [
            metrics([p["replica_values_c"][r] for p in proteins], measured)
            for r in range(3)
        ],
        "median_spread_c": round(
            sorted(p["spread_c"] for p in proteins)[len(proteins) // 2], 2
        ),
    }
    arguments.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
