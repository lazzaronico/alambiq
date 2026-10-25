"""Each shipped example runs and returns the values it was built to return.

The examples are the documentation's worked cases, so each is run from its own
`campaign.yaml` and checked at its headline numbers. An example that needs an engine
environment is skipped when the environment variable that names it is not set; the
quantitative ones need only NumPy and SciPy. `connected-investigation` has its own test
file, which checks every stage.
"""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def needs(*modules: str):
    return pytest.mark.skipif(
        not all(find_spec(m) for m in modules),
        reason=f"requires {', '.join(modules)}",
    )


def needs_env(variable: str):
    return pytest.mark.skipif(
        not os.environ.get(variable), reason=f"set {variable} to run this example"
    )


needs_mmseqs = needs_env("ALAMBIQ_ENV_MMSEQS")
quantitative = needs("numpy", "scipy", "sklearn")


def run_example(name: str, tmp_path: Path):
    from alambiq.api import run

    outcome = run(EXAMPLES / name / "campaign.yaml", run_dir=tmp_path / "run")
    assert outcome.ok, [
        f"{step}: {result.error}"
        for step, result in outcome.steps.items()
        if result.error
    ]
    return outcome


def test_every_example_has_a_readme_and_a_campaign():
    names = sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir())
    assert names
    for name in names:
        assert (EXAMPLES / name / "campaign.yaml").is_file(), name
        readme = (EXAMPLES / name / "README.md").read_text()
        assert readme.startswith("# "), name
        if name != "connected-investigation":
            for heading in ("**Question.**", "**What to look at.**", "**Limits.**"):
                assert heading in readme, (name, heading)


@quantitative
def test_kinetics_recovers_the_constants_that_generated_its_rates(tmp_path):
    outcome = run_example("kinetics-from-measurements", tmp_path)
    found = {p["name"]: p["value"] for p in outcome.outputs("fit")["parameters"]}
    # The rates are Vmax x / (Km + x) at Vmax = 180 uM/min, Km = 2.5 mM, with no noise.
    assert found["Vmax"] == pytest.approx(180.0, rel=1e-5)
    assert found["Km"] == pytest.approx(2.5, rel=1e-5)
    assert outcome.report(tmp_path / "report.html").read_text().startswith("<")


@needs("gemmi", "mdtraj")
def test_first_look_maps_two_of_four_chains_exactly(tmp_path):
    out = run_example("protein-first-look", tmp_path).outputs("dossier")
    assert (out["identifier"], out["sequence_length"]) == ("P69905", 142)
    assert out["reference_features"] == out["mapped_features"] == 214
    mapping = {c["chain"]: c["mapping"] for c in out["chain_mappings"]}
    assert mapping == {"A": "exact", "B": "withheld", "C": "exact", "D": "withheld"}


@needs_mmseqs
def test_relatives_find_the_two_globin_chains_and_the_neighbourhood(tmp_path):
    outcome = run_example("relatives-and-context", tmp_path)
    hits = {h["target"]: h["identity"] for h in outcome.outputs("relatives")["hits"]}
    assert set(hits) == {"P69905", "P68871", "P69891"}
    assert hits["P68871"] == pytest.approx(0.434, abs=1e-3)
    msa = outcome.outputs("conservation")
    assert (msa["rows"], msa["columns"], len(msa["fully_conserved_positions"])) == (
        4,
        142,
        23,
    )
    near = outcome.outputs("neighbourhood")
    assert [a["status"] for a in near["anchors"]] == ["mapped", "mapped"]
    assert near["neighbourhood_proteins"] == 25


@needs("gemmi")
@needs_mmseqs
def test_lanmodulin_comparison_maps_the_structures_the_hit_and_the_cluster(tmp_path):
    outcome = run_example("lanmodulin-comparison", tmp_path)
    mex = outcome.outputs("mex-lanm")
    assert (mex["identifier"], mex["sequence_length"]) == ("ACS39628.1", 133)
    assert mex["reference_features"] == mex["mapped_features"] == 42
    assert [
        (c["chain"], c["mapping"], c["query_offset"]) for c in mex["chain_mappings"]
    ] == [("A", "exact", 28)]
    hans = outcome.outputs("hans-lanm")
    assert hans["reference_features"] == hans["mapped_features"] == 3
    assert [c["mapping"] for c in hans["chain_mappings"]] == ["exact"] * 4
    assert {c["query_offset"] for c in hans["chain_mappings"]} == {23}

    hits = {h["target"]: h for h in outcome.outputs("comparison")["hits"]}
    assert set(hits) == {"ACS39628.1", "A0A4V2JDD3"}
    assert hits["ACS39628.1"]["identity"] == 1.0
    assert hits["A0A4V2JDD3"]["identity"] == pytest.approx(0.36, abs=1e-3)
    assert hits["A0A4V2JDD3"]["alignment_length"] == 100
    assert hits["A0A4V2JDD3"]["bits"] == pytest.approx(73.0, abs=0.5)

    msa = outcome.outputs("alignment")
    assert (msa["rows"], msa["columns"], len(msa["fully_conserved_positions"])) == (
        2,
        133,
        36,
    )

    cluster = outcome.outputs("cluster")
    assert {a["id"]: a["status"] for a in cluster["anchors"]} == {
        "ACS39628.1": "mapped",
        "A0A4V2JDD3": "unmapped",
    }
    (window,) = cluster["windows"]
    assert (window["start"], window["end"], window["features"]) == (
        1848038,
        1868439,
        10,
    )
    assert cluster["neighbourhood_proteins"] == 10
    detail = json.loads(Path(cluster["record"]).read_text())
    anchor = next(
        f
        for f in detail["neighbourhoods"][0]["features"]
        if f["id"] == "MexAM1_META1p1786"
    )
    assert (anchor["start"], anchor["end"]) == (1858038, 1858439)


@quantitative
def test_representative_panel_recommends_three_across_the_family(tmp_path):
    out = run_example("representative-panel", tmp_path).outputs("layout")
    assert out["recommended"] == ["P69891", "P60709", "P02144"]
    assert out["explained_variance_ratio"][0] == pytest.approx(0.6553, abs=1e-4)


@needs("gemmi")
@needs_env("ALAMBIQ_ENV_P2RANK")
def test_pockets_rank_a_confident_first_pocket(tmp_path):
    out = run_example("pockets-and-complex", tmp_path).outputs("pockets")
    top = out["pockets"][0]
    assert top["rank"] == 1
    assert top["probability"] == pytest.approx(0.98, abs=0.005)
    assert out["residue_memberships"] == 572


@needs("pydna", "Bio", "dnachisel")
def test_variant_to_construct_meets_its_design_rules(tmp_path):
    outcome = run_example("variant-to-construct", tmp_path)
    insert = outcome.outputs("insert")
    assert (insert["length_bp"], insert["protein_length"]) == (429, 142)
    recode = outcome.outputs("recode")
    assert recode["changed_codons"] == 13
    assert recode["before"]["motif_matches"] == 1
    assert recode["after"]["motif_matches"] == 0
    assert recode["after"]["satisfied"] is True
    assert recode["after"]["maximum_window_gc"] <= 0.65
    build = outcome.outputs("build")
    assert (build["length"], build["features"]) == (3115, 8)


@needs("rdkit", "doranet", "equilibrator_api")
def test_transformation_balances_and_is_downhill(tmp_path):
    outcome = run_example("transformation-and-routes", tmp_path)
    assert outcome.outputs("account")["balance"]["balanced"] is True
    routes = outcome.outputs("routes")
    assert routes["target_status"] == "inventory-route-found"
    assert len(routes["routes"]) == 1
    (reaction,) = outcome.outputs("energy")["reactions"]
    assert reaction["dg0_prime_kj_mol"] == pytest.approx(-19.94, abs=0.02)
    assert reaction["dg_prime_kj_mol"] == pytest.approx(-37.05, abs=0.02)


@needs_mmseqs
def test_dehalogenase_funnel_advances_positives_and_rejects_confounders(tmp_path):
    outcome = run_example("transformation-to-enzymes", tmp_path)
    ranked = outcome.outputs("retrieve")
    assert ranked["positive_control_recall"] == 1.0
    assert ranked["all_hard_negatives_below_positives"] is True
    assert ranked["control_auroc"] == 1.0
    assert ranked["length_only_baseline_auroc"] == pytest.approx(0.8)
    out = outcome.outputs("characterize")
    assert out["advanced"] == ["P24070", "P60527", "Q53464", "Q60099", "Q7DI63"]
    assert out["rejected"] == ["A0A140N5J7", "O06995", "P22643", "P32662", "Q8KLS9"]
    assert out["withheld"] == []


@quantitative
def test_measured_rounds_say_the_model_did_not_beat_the_mean(tmp_path):
    outcome = run_example("measured-variant-rounds", tmp_path)
    fit = outcome.outputs("model")
    assert (fit["trained_on"], fit["evaluated_on"]) == (82, 103)
    assert fit["model"]["rmse"] == pytest.approx(2.9255, abs=1e-3)
    assert fit["baseline"]["rmse"] == pytest.approx(2.3892, abs=1e-3)
    assert fit["model"]["rmse"] > fit["baseline"]["rmse"]
    assert fit["interval_95_coverage"] == pytest.approx(0.8155, abs=1e-3)
    batches = {
        step: [row["sample_id"] for row in outcome.outputs(step)["selected"]]
        for step in ("highest-predicted", "most-uncertain", "random")
    }
    assert all(len(batch) == 8 for batch in batches.values())
    assert len({tuple(b) for b in batches.values()}) == 3
    # At most one variant per position in the two model-driven batches.
    for step in ("highest-predicted", "most-uncertain"):
        groups = [row["group_id"] for row in outcome.outputs(step)["selected"]]
        assert len(groups) == len(set(groups))


@quantitative
def test_progress_fits_reproduce_the_published_comparison(tmp_path):
    outcome = run_example("operation-from-progress-curves", tmp_path)

    def errors(step):
        out = outcome.outputs(step)
        return out["fit_error"], out["evaluation_error"]

    assert errors("free-first-order") == pytest.approx((0.871, 1.435), abs=2e-3)
    assert errors("hydrogel-first-order") == pytest.approx((1.953, 5.477), abs=2e-3)
    assert errors("hydrogel-saturation") == pytest.approx((0.997, 1.853), abs=2e-3)
    saturation = {
        p["name"]: p for p in outcome.outputs("hydrogel-saturation")["parameters"]
    }
    assert saturation["Km"]["value"] == pytest.approx(6.235e-3, rel=2e-3)
    assert saturation["capacity"]["value"] == pytest.approx(5.029e-5, rel=2e-3)
    free = {p["name"]: p for p in outcome.outputs("free-saturation")["parameters"]}
    assert free["Km"]["at_bound"] is True
    final = outcome.outputs("reuse-fitted-operation")["final_concentrations_M"]
    assert final["acetophenone"] + final["converted-substrate"] == pytest.approx(5e-3)


@quantitative
def test_regeneration_lifts_output_above_the_cofactor_supply(tmp_path):
    outcome = run_example("cofactor-regeneration", tmp_path)
    without = outcome.outputs("without-output")["net_formed_mol"]
    with_regeneration = outcome.outputs("with-output")["net_formed_mol"]
    # 1 mM NADH in 1 mL cannot make more than 1 umol of product without regeneration.
    assert 0.99e-6 < without <= 1.0e-6
    assert with_regeneration == pytest.approx(32.7e-6, rel=5e-3)
    assert with_regeneration > 30 * without
    for step in ("without", "with"):
        final = outcome.outputs(step)["final_concentrations_M"]
        moiety = final["NADH"] + final["NAD"] + final["NADH_lost"]
        assert moiety == pytest.approx(1e-3, rel=1e-9)


@pytest.mark.engine("openmm")
@needs("mdtraj", "gemmi")
@needs_env("ALAMBIQ_ENV_OPENMM")
def test_short_implicit_solvent_ensemble_compares_all_225_couplings(tmp_path):
    out = run_example("ensemble-vs-nmr", tmp_path).outputs("couplings")
    assert len(out["couplings"]) == 225
    assert out["replicas"] == ["replica-1", "replica-2", "replica-3"]
    assert out["frames_per_replica"] == 100
    assert out["atom_naming"] == "iupac-beta-methylene"
    assert 0.4 < out["mae_hz"] < out["rmse_hz"] < 1.5
