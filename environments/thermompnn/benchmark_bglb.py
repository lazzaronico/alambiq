"""Reproduce the frozen BglB functional-stability benchmark with the Alambiq Python interface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom, pearsonr, spearmanr

PROTOCOL_SHA256 = "91ae377528665f7f4ae76a49997d8b69fbb091fb33a997e79ef12300d808e4fb"
TABLE_SHA256 = "3d94ee8b73d4b67cc7c1f6def54b00a40bdacc6f1eaf1b53bcd01294fe58b619"
ARCHIVE_SHA256 = "ba4dde758a0c7181b55278bd4e1797b488cf13b5d24f6017a2c38ff9df31d0b0"
STRUCTURE_SHA256 = "01c878a91c2d68d70f19a07933958ee841bbf33349bd9f541c52dacb61faf860"
STRUCTURE_MEMBER = "S1 Code/example_rosetta_ddg_run/bglb_apo.pdb"


def analyze(raw, record, timing, protocol, protocol_sha, result_sha, step_key):
    import math

    predictions = record["rows"]
    assert len(predictions) == len(raw) == 129
    records = []
    for source, prediction in zip(raw, predictions, strict=True):
        if source["mutant"] == "BglB":
            assert prediction["wild_type"] == prediction["mutant"] == "S"
            assert prediction["predicted_ddg_kcal_mol"] == 0
        else:
            assert prediction["source_residue"]["residue_id"] == source["sequence_pos"]
            assert (
                source["mutant"]
                == prediction["wild_type"]
                + source["sequence_pos"]
                + prediction["mutant"]
            )
        records.append(
            {
                "mutant": source["mutant"],
                "position": int(source["sequence_pos"])
                if source["sequence_pos"]
                else None,
                "expression": int(source["expression"]),
                "functional_midpoint_celsius": None
                if source["tm"] == "NA"
                else float(source["tm"]),
                "reported_midpoint_sd_celsius": None
                if source["err_tm"] == "NA"
                else float(source["err_tm"]),
                "kcat_min_inverse": source["kcat"],
                "km_millimolar": source["km"],
                "catalytic_efficiency_M_inverse_min_inverse": source["kcatkm"],
                "predicted_ddg_kcal_mol": prediction["predicted_ddg_kcal_mol"],
                "predicted_stability_score": -prediction["predicted_ddg_kcal_mol"],
            }
        )
    wild = next(r for r in records if r["mutant"] == "BglB")
    variants = [r for r in records if r["mutant"] != "BglB"]
    measured = [r for r in variants if r["functional_midpoint_celsius"] is not None]
    assert len(measured) == 78
    scores = np.array([r["predicted_stability_score"] for r in measured])
    midpoints = np.array([r["functional_midpoint_celsius"] for r in measured])
    positions = sorted({r["position"] for r in measured})
    groups = [
        np.array([i for i, r in enumerate(measured) if r["position"] == position])
        for position in positions
    ]
    rng = np.random.default_rng(581005)
    boot = []
    for _ in range(2000):
        indices = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        value = float(spearmanr(scores[indices], midpoints[indices]).statistic)
        assert math.isfinite(value)
        boot.append(value)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    threshold = wild["functional_midpoint_celsius"] + 1.0

    def ranking(rows):
        return sorted(
            rows, key=lambda r: (-r["predicted_stability_score"], r["mutant"])
        )

    top_measured = ranking(measured)[:10]
    gains = sum(r["functional_midpoint_celsius"] >= threshold for r in measured)
    top_gains = sum(r["functional_midpoint_celsius"] >= threshold for r in top_measured)
    pvalue = float(hypergeom.sf(top_gains - 1, len(measured), gains, 10))

    def shortlist_row(row):
        efficiency = row["catalytic_efficiency_M_inverse_min_inverse"]
        activity = (
            None
            if efficiency in ("", "NA")
            else False
            if efficiency.startswith("<")
            else float(efficiency) >= 87000
        )
        gain = (
            None
            if row["functional_midpoint_celsius"] is None
            else row["functional_midpoint_celsius"] >= threshold
        )
        return {
            **row,
            "verified_midpoint_gain_at_least_one_C": gain,
            "observed_efficiency_at_least_half_WT": activity,
            "both_observed_targets_pass": gain is True and activity is True,
        }

    full_shortlist = [shortlist_row(r) for r in ranking(variants)[:10]]
    receipt = {
        "protocol_sha256": protocol_sha,
        "status": "completed under frozen protocol",
        "step_key": step_key,
        "result_sha256": result_sha,
        "scope": protocol["scope"],
        "counts": {
            "mutants": 128,
            "measured_midpoints": 78,
            "measured_midpoint_positions": len(positions),
            "missing_mutant_midpoints": 50,
            "nonexpressed": sum(r["expression"] == 0 for r in variants),
        },
        "ranking": {
            "spearman": float(spearmanr(scores, midpoints).statistic),
            "position_bootstrap_95_percent_interval": ci,
            "bootstrap_replicates": 2000,
            "pearson_descriptive": float(pearsonr(scores, midpoints).statistic),
            "positive_ranking_criterion_passes": ci[0] > 0,
        },
        "measured_shortlist": {
            "size": 10,
            "gain_threshold_celsius": threshold,
            "population_measured_gains": gains,
            "observed_shortlist_gains": top_gains,
            "random_expected_gains": 10 * gains / len(measured),
            "enrichment_ratio": top_gains / (10 * gains / len(measured))
            if gains
            else None,
            "hypergeometric_one_sided_p": pvalue,
            "enrichment_criterion_passes": pvalue <= 0.05,
            "rows": [shortlist_row(r) for r in top_measured],
        },
        "all_variants_shortlist": {
            "size": 10,
            "observed_midpoints": sum(
                r["functional_midpoint_celsius"] is not None for r in full_shortlist
            ),
            "both_observed_targets_pass": sum(
                r["both_observed_targets_pass"] for r in full_shortlist
            ),
            "rows": full_shortlist,
        },
        "qualification": "This single-enzyme ranking/enrichment endpoint qualifies only if both predeclared criteria pass; calibrated folding-ddG and catalytic-rate prediction remain unqualified.",
        "qualified_for_declared_single_enzyme_endpoint": ci[0] > 0 and pvalue <= 0.05,
        "timing": timing,
        "rows": records,
    }
    return receipt, boot


def main():
    import gemmi

    from alambiq.api import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources", type=Path, help="Directory containing S1-Table.csv and S1-Code.zip"
    )
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    protocol_bytes = Path(__file__).with_name("bglb-protocol.json").read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("Frozen benchmark protocol changed")
    table_bytes = (args.sources / "S1-Table.csv").read_bytes()
    archive_bytes = (args.sources / "S1-Code.zip").read_bytes()
    if (
        hashlib.sha256(table_bytes).hexdigest() != TABLE_SHA256
        or hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256
    ):
        raise ValueError("Experimental table or supplementary archive identity changed")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        pdb = archive.read(STRUCTURE_MEMBER)
    if hashlib.sha256(pdb).hexdigest() != STRUCTURE_SHA256:
        raise ValueError("Author-prepared coordinate identity changed")
    raw = list(csv.DictReader(io.StringIO(table_bytes.decode())))
    structure = gemmi.read_pdb_string(pdb.decode())
    sequence = "".join(
        gemmi.find_tabulated_residue(r.name).one_letter_code for r in structure[0]["A"]
    )
    mutations = [
        {
            "residue": r["sequence_pos"],
            "wild_type": r["mutant"][0],
            "mutant": r["mutant"][-1],
        }
        for r in raw
        if r["sequence_pos"]
    ]
    mutations.append({"residue": "14", "wild_type": "S", "mutant": "S"})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_bytes(protocol_bytes)
    source = args.output / "author-prepared-apo.pdb"
    source.write_bytes(pdb)
    construct = args.output / "author-prepared-apo.fasta"
    construct.write_text(f">BglB\n{sequence}\n")
    campaign = {
        "id": "bglb-functional-stability-qualification",
        "base": str(args.output.resolve()),
        "steps": [
            {
                "id": "predict",
                "capability": "protein.predict_stability_change",
                "params": {
                    "structure": str(source.resolve()),
                    "sequence": str(construct.resolve()),
                    "chain_id": "A",
                    "structure_format": "pdb",
                    "mutations": mutations,
                    "threads": 4,
                    "seed": 581004,
                },
            }
        ],
    }
    (args.output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    started = time.perf_counter()
    outcome = run(campaign, run_dir=args.output / "run")
    elapsed = time.perf_counter() - started
    predict = outcome.steps["predict"]
    if not outcome.ok:
        raise RuntimeError(
            f"Stability campaign failed: {predict.error}; inspect {outcome.run_dir}"
        )
    record = json.loads(Path(predict.outputs["record"]).read_text())
    serialized = {
        "campaign": campaign,
        "step_key": predict.key,
        "outputs": predict.outputs,
        "record": record,
    }
    result_bytes = (json.dumps(serialized, indent=2, allow_nan=False) + "\n").encode()
    (args.output / "result.json").write_bytes(result_bytes)
    runtime = record["runtime"] | {
        "provider_seconds": predict.seconds,
        "provider_scope": (
            "Structure validation, resource verification, worker startup, model loading "
            "and inference; excludes installation, downloads and workflow orchestration"
        ),
    }
    timing = {
        "campaign_execution_seconds": elapsed,
        "provider": runtime,
        "scope": "Fresh campaign with existing installation/model assets; installation/download cold cost unmeasured",
    }
    analysis, bootstrap = analyze(
        raw,
        record,
        timing,
        json.loads(protocol_bytes),
        PROTOCOL_SHA256,
        hashlib.sha256(result_bytes).hexdigest(),
        predict.key,
    )
    analysis["analysis_versions"] = {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy")
    }
    (args.output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "bootstrap.json").write_text(
        json.dumps(bootstrap, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "qualified_for_declared_single_enzyme_endpoint": analysis[
                    "qualified_for_declared_single_enzyme_endpoint"
                ],
                "ranking": analysis["ranking"],
                "provider_seconds": runtime["provider_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
