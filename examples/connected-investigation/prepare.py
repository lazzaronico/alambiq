"""Regenerate the inputs of this example.

    ALAMBIQ_ENV_ESM=/path/to/esm/python python prepare.py

`scores.json` is the output of `protein.score_substitutions` (ESM C 300M) on the lysozyme
sequence, produced by the real model; everything else is generated here with fixed seeds
from a stated truth, so that the campaign can be checked against it:

* the activity of a returned round of proposed variants is 1 + 0.15 times the change in the
  number of arginines (a toy rule a composition model can learn), plus Gaussian noise of 0.015;
* the variant the campaign goes on to choose (the one `candidate.fit` and `candidate.select`
  pick from the unmeasured pool, found here by running them) has a Vmax of its relative
  activity times 180 uM/min and a Km of 2.5 mM, measured at ten substrate concentrations with
  3 % noise;
* the progress curve of the operation is the integrated Michaelis-Menten solution for
  those constants, in the 0.5 L reactor of `template.json`, with 1.5 % noise.

None of it is real measurement. It makes the chain from a protein to an operating
comparison runnable, with answers that are known.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LYSOZYME = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
CATALYTIC = [35, 52]
ARGININE = {"R": 1}
HEAD = (
    "measurement_id,sample_id,condition_id,metric,value,unit,"
    "independent_id,technical_id,role,status\n"
)
VMAX_WT, KM = 180.0, 2.5
XS = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 8, 20, 30]


def activity(wild: str, mutant: str) -> float:
    return 1.0 + 0.15 * (ARGININE.get(mutant, 0) - ARGININE.get(wild, 0))


def solve(s0: float, km: float, vmax: float, t: float) -> float:
    from scipy.optimize import brentq

    if t == 0:
        return s0
    return brentq(
        lambda s: km * math.log(s0 / s) + (s0 - s) - vmax * t, 1e-15, s0, xtol=1e-18
    )


def main() -> int:
    from alambiq.api import run

    (HERE / "lysozyme.fasta").write_text(
        f">hewl hen egg-white lysozyme, mature\n{LYSOZYME}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        outcome = run(
            {
                "id": "prepare",
                "steps": [
                    {
                        "id": "scores",
                        "capability": "protein.score_substitutions",
                        "params": {
                            "protein": str(HERE / "lysozyme.fasta"),
                            "batch_size": 16,
                        },
                    },
                ],
            },
            run_dir=work / "run",
        )
        if not outcome.ok:
            print(outcome.steps["scores"].error, file=sys.stderr)
            return 1
        shutil.copy(outcome.outputs("scores")["record"], HERE / "scores.json")
        proposed = run(
            {
                "id": "prepare",
                "steps": [
                    {
                        "id": "propose",
                        "capability": "variant.propose",
                        "params": {
                            "scores": str(HERE / "scores.json"),
                            "count": 24,
                            "exclude_positions": CATALYTIC,
                            "max_per_position": 1,
                        },
                    }
                ],
            },
            run_dir=work / "propose",
        )
        proposals = proposed.outputs("propose")["proposals"]
    rng = random.Random(20261023)
    delta = {
        p["variant"]: ARGININE.get(p["mutant"], 0) - ARGININE.get(p["wild_type"], 0)
        for p in proposals
    }
    order = [p["variant"] for p in proposals]
    # The pool is six unmeasured proposals, the evaluation six, the training twelve. The
    # split is shuffled with a fixed seed until the pool offers at least two gains, training
    # has seen at least three gains and a loss, and evaluation has a gain, so that the fit
    # has something to learn and something to be checked on.
    for attempt in range(1000):
        shuffled = order[:]
        random.Random(attempt).shuffle(shuffled)
        pool_ids, eval_ids, train_ids = shuffled[:6], shuffled[6:12], shuffled[12:]
        if (
            sum(delta[v] > 0 for v in pool_ids) >= 2
            and sum(delta[v] > 0 for v in train_ids) >= 3
            and any(delta[v] < 0 for v in train_ids)
            and any(delta[v] > 0 for v in eval_ids)
            and len({v[1:-1] for v in train_ids}) >= 3
        ):
            break
    else:
        raise RuntimeError("no suitable split")
    partition_of = {
        **dict.fromkeys(pool_ids, "pool"),
        **dict.fromkeys(eval_ids, "evaluation"),
        **dict.fromkeys(train_ids, "train"),
    }
    rows, candidates, truth = [], [], {}
    for item in proposals:
        sequence = (
            LYSOZYME[: item["position"] - 1]
            + item["mutant"]
            + LYSOZYME[item["position"] :]
        )
        partition = partition_of[item["variant"]]
        truth[item["variant"]] = activity(item["wild_type"], item["mutant"])
        candidates.append(
            {
                "sample_id": item["variant"],
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "group_id": f"position:{item['position']}",
                "partition": partition,
            }
        )
        if partition != "pool":
            value = truth[item["variant"]] + rng.gauss(0.0, 0.015)
            rows.append(
                f"m-{item['variant']},{item['variant']},standard,relative-activity,"
                f"{value!r},relative,,,sample,measured\n"
            )
    (HERE / "panel.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1",
                "candidates": candidates,
                "source": "Lysozyme variants proposed by variant.propose; measurements are generated (see prepare.py)",
            },
            indent=1,
        )
        + "\n"
    )
    (HERE / "returned.csv").write_text(HEAD + "".join(rows))
    pool = [c["sample_id"] for c in candidates if c["partition"] == "pool"]
    with tempfile.TemporaryDirectory() as tmp:
        picked = run(
            {
                "id": "prepare",
                "steps": [
                    {
                        "id": "round",
                        "capability": "assay.inspect",
                        "params": {
                            "table": str(HERE / "returned.csv"),
                            "conditions": {
                                "standard": "Standard assay, returned round"
                            },
                        },
                    },
                    {
                        "id": "model",
                        "capability": "candidate.fit",
                        "bind": {"assay": "round.record"},
                        "params": {
                            "panel": str(HERE / "panel.json"),
                            "metric": "relative-activity",
                            "condition_id": "standard",
                            "unit": "relative",
                        },
                    },
                    {
                        "id": "next",
                        "capability": "candidate.select",
                        "bind": {"model": "model.record"},
                        "params": {
                            "panel": str(HERE / "panel.json"),
                            "strategy": "highest-predicted",
                            "count": 1,
                            "seed": 1,
                        },
                    },
                ],
            },
            run_dir=Path(tmp) / "run",
        )
        if not picked.ok:
            print([s.error for s in picked.steps.values() if s.error], file=sys.stderr)
            return 1
        chosen = picked.outputs("next")["selected"][0]["sample_id"]
    vmax = VMAX_WT * truth[chosen]
    kinetics = [
        f"k{i},{chosen},c{i},initial-rate,{vmax * x / (KM + x) * (1 + rng.gauss(0, 0.03))!r},"
        "uM/min,,,sample,measured\n"
        for i, x in enumerate(XS)
    ]
    (HERE / "rates.csv").write_text(HEAD + "".join(kinetics))
    s0 = 0.01
    capacity = vmax * 1e-6 / 60.0
    times = [0.0, 300.0, 600.0, 1200.0, 1800.0, 2700.0, 3600.0]
    observations = []
    for t in times:
        s = solve(s0, KM * 1e-3, capacity, t)
        value = s * (1 + (rng.gauss(0, 0.015) if t else 0.0))
        observations.append(
            {
                "id": f"s-{int(t)}",
                "time_s": t,
                "state": "after",
                "quantity": "concentration",
                "target": "S",
                "unit": "M",
                "value": value,
                "partition": "evaluation",
                "source": "Generated from the stated truth with 1.5 % noise",
                "reported_error": {"kind": "SD", "value": 0.015 * s},
            }
        )
    (HERE / "progress.json").write_text(
        json.dumps(
            {
                "id": "progress-curve",
                "operation_id": "hydrolysis",
                "conditions": "Assay conditions of the kinetics measurement",
                "observations": observations,
            },
            indent=1,
        )
        + "\n"
    )
    (HERE / "truth.json").write_text(
        json.dumps(
            {
                "chosen_variant": chosen,
                "relative_activity": truth[chosen],
                "vmax_uM_min": vmax,
                "km_mM": KM,
                "proposals": [c["sample_id"] for c in candidates],
                "pool": pool,
            },
            indent=1,
        )
        + "\n"
    )
    campaign = (HERE / "campaign.yaml").read_text()
    marker = "      sample_id: "
    head, _, tail = campaign.partition("  - id: kinetics\n")
    tail = tail.replace(
        marker + tail.split(marker, 1)[1].split("\n", 1)[0], marker + chosen, 1
    )
    (HERE / "campaign.yaml").write_text(head + "  - id: kinetics\n" + tail)
    print(f"the campaign chooses {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
