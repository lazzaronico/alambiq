"""reaction.estimate_thermodynamics against the locked eQuilibrator model.

ATP hydrolysis at pH 7.5, pMg 3 and ionic strength 0.25 M is the reference:
ΔG′° = -29.64 kJ/mol from the installed model, in the range textbooks give for ATP
hydrolysis (about -30 kJ/mol). Everything else is arithmetic that follows from it
(ΔG′ = ΔG′° + RT ln Q) or a property the covariance must have.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("equilibrator_api")
pytest.importorskip("rdkit")

EXPECTED_DG0 = -29.64


def atp() -> dict:
    return {
        "id": "atp-hydrolysis",
        "conditions": {
            "pH": 7.5,
            "pMg": 3.0,
            "ionic_strength_M": 0.25,
            "temperature_K": 298.15,
        },
        "assumption_note": "Ideal dilute aqueous activities, concentration divided by 1 M; water activity is one.",
        "compounds": [
            {"id": "atp", "accession": "kegg:C00002", "activity": 0.001},
            {"id": "water", "accession": "kegg:C00001", "activity": 1},
            {"id": "adp", "accession": "kegg:C00008", "activity": 0.001},
            {"id": "phosphate", "accession": "kegg:C00009", "activity": 0.001},
        ],
        "reactions": [
            {
                "id": "hydrolysis",
                "stoichiometry": {"atp": -1, "water": -1, "adp": 1, "phosphate": 1},
                "source": {
                    "kind": "curated-reference",
                    "citation": "eQuilibrator ATP hydrolysis tutorial",
                },
            }
        ],
        "route_weights": {"hydrolysis": 1},
    }


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {
        "id": "thermo",
        "capability": "reaction.estimate_thermodynamics",
        "params": params,
    }
    return execute({"id": "t", "steps": [step]}, run_dir=tmp_path / name)


def estimate(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def out(outcome) -> dict:
    return outcome.outputs("thermo")


def record(outcome) -> dict:
    return json.loads(Path(out(outcome)["record"]).read_text())


class TestAtpHydrolysis:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        return estimate(atp(), tmp_path_factory.mktemp("atp"))

    def test_the_standard_energy_is_the_locked_value_and_in_the_textbook_range(
        self, outcome
    ):
        row = out(outcome)["reactions"][0]
        assert row["status"] == "estimated"
        assert row["dg0_prime_kj_mol"] == pytest.approx(EXPECTED_DG0, abs=0.02)
        assert -32.5 < row["dg0_prime_kj_mol"] < -28.5

    def test_the_activity_correction_is_rt_ln_q_exactly(self, outcome):
        result = out(outcome)
        row = result["reactions"][0]
        # Q = [ADP][Pi]/[ATP] with water at unit activity: 0.001 × 0.001 / 0.001.
        assert row["ln_q"] == pytest.approx(math.log(0.001), abs=1e-12)
        assert row["dg_prime_kj_mol"] == pytest.approx(
            row["dg0_prime_kj_mol"] + result["rt_kj_mol"] * math.log(0.001), abs=1e-10
        )
        assert row["ln_k_prime"] == pytest.approx(
            -row["dg0_prime_kj_mol"] / result["rt_kj_mol"], abs=1e-10
        )

    def test_rt_is_the_gas_constant_equilibrator_uses_times_298_15_kelvin(
        self, outcome
    ):
        # eQuilibrator defines R as 8.31 J/(mol K), slightly under the CODATA value,
        # so RT is 2.4776 kJ/mol here and not 2.4790.
        assert out(outcome)["rt_kj_mol"] == pytest.approx(8.31e-3 * 298.15, rel=1e-9)

    def test_the_model_reports_an_uncertainty_and_the_route_net_equation(self, outcome):
        assert out(outcome)["reactions"][0]["model_sd_kj_mol"] > 0
        assert out(outcome)["route"]["net_stoichiometry"] == {
            "adp": "1",
            "atp": "-1",
            "phosphate": "1",
            "water": "-1",
        }

    def test_the_single_reaction_route_is_the_reaction(self, outcome):
        route, row = out(outcome)["route"], out(outcome)["reactions"][0]
        assert route["status"] == "estimated"
        assert route["dg_prime_kj_mol"] == pytest.approx(
            row["dg_prime_kj_mol"], abs=1e-12
        )
        assert route["model_sd_kj_mol"] == pytest.approx(
            row["model_sd_kj_mol"], abs=1e-9
        )

    def test_every_compound_resolved_and_the_model_identity_is_recorded(self, outcome):
        assert {c["status"] for c in out(outcome)["compounds"]} == {"resolved"}
        model = out(outcome)["model"]
        assert set(model["packages"]) == {
            "equilibrator-api",
            "equilibrator-cache",
            "component-contribution",
            "equilibrator-cache-data",
        }
        assert set(model["assets"]) == {"mini_compounds.sqlite", "cc_params.npz"}
        assert model["data_license"] == "CC-BY-4.0"

    def test_the_report_lists_the_reactions(self, outcome, tmp_path):
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "hydrolysis" in page

    def test_the_equation_at_equilibrium_activities_has_zero_driving_force(
        self, outcome, tmp_path
    ):
        params = atp()
        k = math.exp(out(outcome)["reactions"][0]["ln_k_prime"])
        # Q = [ADP][Pi]/[ATP]: with ADP and Pi at 1e-3, Q = 1e-6/[ATP] equals K' when [ATP] = 1e-6/K'.
        params["compounds"][0]["activity"] = 1e-6 / k
        assert out(estimate(params, tmp_path))["route"][
            "dg_prime_kj_mol"
        ] == pytest.approx(0, abs=1e-9)


class TestRoutes:
    def test_a_reaction_and_its_reverse_cancel_energy_activity_and_variance(
        self, tmp_path
    ):
        params = atp()
        reverse = copy.deepcopy(params["reactions"][0])
        reverse["id"] = "synthesis"
        reverse["stoichiometry"] = {k: -v for k, v in reverse["stoichiometry"].items()}
        params["reactions"].append(reverse)
        params["route_weights"]["synthesis"] = 1
        outcome = estimate(params, tmp_path)
        route = out(outcome)["route"]
        assert route["net_stoichiometry"] == {}
        assert route["dg_prime_kj_mol"] == pytest.approx(0, abs=1e-10)
        assert route["model_sd_kj_mol"] == pytest.approx(0, abs=1e-7)
        # Perfectly anti-correlated errors: the off-diagonal covariance is negative.
        assert record(outcome)["covariance_kj2_mol2"][0][1] < 0

    def test_the_same_reaction_used_twice_doubles_the_uncertainty_not_root_two(
        self, tmp_path
    ):
        params = atp()
        second = copy.deepcopy(params["reactions"][0])
        second["id"] = "second"
        params["reactions"].append(second)
        params["route_weights"]["second"] = 1
        outcome = estimate(params, tmp_path)
        assert out(outcome)["route"]["model_sd_kj_mol"] == pytest.approx(
            2 * out(outcome)["reactions"][0]["model_sd_kj_mol"], abs=1e-9
        )

    def test_a_regeneration_cycle_closes_the_nadh_pool_and_consumes_the_substrate(
        self, tmp_path
    ):
        params = atp()
        params["compounds"] = [
            {
                "id": k,
                "accession": "kegg:" + k,
                "activity": 1 if k == "C00001" else 0.001,
            }
            for k in (
                "C00003",
                "C00004",
                "C00022",
                "C00186",
                "C00031",
                "C00257",
                "C00001",
            )
        ]
        source = {
            "kind": "publication",
            "citation": "doi:10.1038/s41467-022-35228-y; LDH/GDH auxiliary redox chemistry",
        }
        params["reactions"] = [
            {
                "id": "ldh",
                "stoichiometry": {"C00022": -1, "C00004": -1, "C00186": 1, "C00003": 1},
                "source": source,
            },
            {
                "id": "gdh",
                "stoichiometry": {
                    "C00031": -1,
                    "C00001": -1,
                    "C00003": -1,
                    "C00257": 1,
                    "C00004": 1,
                },
                "source": source,
            },
        ]
        params["route_weights"] = {"ldh": 1, "gdh": 1}
        outcome = estimate(params, tmp_path)
        net = out(outcome)["route"]["net_stoichiometry"]
        assert out(outcome)["route"]["status"] == "estimated"
        assert "C00003" not in net and "C00004" not in net
        assert net["C00031"] == "-1" and net["C00257"] == "1"
        # Remove the reduced nicotinamide from one step and its electrons no longer balance.
        del params["reactions"][0]["stoichiometry"]["C00004"]
        assert (
            out(estimate(params, tmp_path, "unbalanced"))["reactions"][0]["status"]
            == "unbalanced"
        )


class TestWhatCannotBeEstimated:
    def test_a_missing_water_is_an_oxygen_residual_and_leaves_the_route_unresolved(
        self, tmp_path
    ):
        params = atp()
        del params["reactions"][0]["stoichiometry"]["water"]
        outcome = estimate(params, tmp_path)
        row = out(outcome)["reactions"][0]
        assert row["status"] == "unbalanced" and row["balance_residuals"]["O"] == "1"
        assert out(outcome)["route"]["dg_prime_kj_mol"] is None
        assert out(outcome)["route"]["status"] == "unresolved-steps"
        assert "has no estimate (unbalanced)" in " ".join(outcome.steps["thermo"].notes)

    def test_an_unknown_accession_is_unresolved_and_never_looked_up_online(
        self, tmp_path
    ):
        params = atp()
        params["compounds"][0]["accession"] = "kegg:C99999999"
        with patch("pooch.retrieve", side_effect=AssertionError("offline lookup")):
            outcome = estimate(params, tmp_path)
        assert out(outcome)["compounds"][0]["status"] == "unresolved-accession"
        assert out(outcome)["reactions"][0]["status"] == "unresolved-compounds"
        assert out(outcome)["reactions"][0]["unresolved_compounds"] == ["atp"]

    def test_missing_dissociation_data_stays_unresolved_with_the_reason(self, tmp_path):
        from equilibrator_cache.exceptions import MissingDissociationConstantsException

        with patch(
            "equilibrator_api.phased_reaction.PhasedReaction.transform",
            side_effect=MissingDissociationConstantsException("missing microspecies"),
        ):
            outcome = estimate(atp(), tmp_path)
        row = out(outcome)["reactions"][0]
        assert (
            row["status"] == "unresolved-transform" and row["dg0_prime_kj_mol"] is None
        )
        assert "missing microspecies" in " ".join(row["notes"])

    def test_a_direction_the_model_leaves_unconstrained_is_unresolved_not_guessed(
        self, tmp_path
    ):
        import numpy as np

        from alambiq.capabilities.investigate.science import thermodynamics

        model, identity = thermodynamics.local_model()
        with (
            patch.object(
                model.predictor,
                "get_reaction_prediction",
                return_value=(0, np.array([0]), np.array([1]), {}),
            ),
            patch.object(thermodynamics, "local_model", return_value=(model, identity)),
        ):
            outcome = estimate(atp(), tmp_path)
        assert out(outcome)["reactions"][0]["status"] == "unresolved-model"
        assert record(outcome)["covariance_kj2_mol2"][0][0] is None


class TestCompoundIdentity:
    def test_a_reference_structure_matches_within_a_proton_ensemble_and_not_otherwise(
        self, tmp_path
    ):
        params = atp()
        params["compounds"][3]["reference_smiles"] = "OP(=O)(O)O"
        assert out(estimate(params, tmp_path))["compounds"][3]["status"] == "resolved"
        params["compounds"][3]["reference_smiles"] = "CCO"
        assert (
            out(estimate(params, tmp_path, "b"))["compounds"][3]["status"]
            == "reference-identity-mismatch"
        )

    def test_a_stereochemical_reference_distinguishes_enantiomers(self, tmp_path):
        params = atp()
        params["compounds"].append(
            {
                "id": "lactate",
                "accession": "kegg:C00186",
                "activity": 0.001,
                "reference_smiles": "C[C@H](O)C(=O)O",
            }
        )
        assert out(estimate(params, tmp_path))["compounds"][-1]["status"] == "resolved"
        params["compounds"][-1]["reference_smiles"] = "C[C@@H](O)C(=O)O"
        assert (
            out(estimate(params, tmp_path, "b"))["compounds"][-1]["status"]
            == "reference-identity-mismatch"
        )


class TestConditions:
    def test_a_buffered_proton_with_null_activity_changes_nothing(self, tmp_path):
        params = atp()
        base = out(estimate(params, tmp_path))["reactions"][0]["dg_prime_kj_mol"]
        params["compounds"].append(
            {"id": "proton", "accession": "kegg:C00080", "activity": None}
        )
        params["reactions"][0]["stoichiometry"]["proton"] = 1
        assert out(estimate(params, tmp_path, "b"))["reactions"][0][
            "dg_prime_kj_mol"
        ] == pytest.approx(base, abs=1e-9)

    def test_changing_the_ph_changes_the_standard_energy_and_nothing_else_leaks(
        self, tmp_path
    ):
        base = out(estimate(atp(), tmp_path))
        params = atp()
        params["conditions"]["pH"] = 6.0
        changed = out(estimate(params, tmp_path, "b"))
        assert (
            abs(
                base["reactions"][0]["dg0_prime_kj_mol"]
                - changed["reactions"][0]["dg0_prime_kj_mol"]
            )
            > 1e-3
        )
        assert out(estimate(atp(), tmp_path, "c"))["reactions"] == base["reactions"]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda r: r["conditions"].update(temperature_K=310), "temperature_K"),
            (lambda r: r["conditions"].update(pH=float("nan")), "pH"),
            (lambda r: r["conditions"].update(pH=15), "pH"),
            (
                lambda r: r["conditions"].update(ionic_strength_M=0.6),
                "ionic_strength_M",
            ),
            (lambda r: r["compounds"][0].update(activity=0), "activity"),
            (lambda r: r["compounds"][0].update(activity=True), "expected a number"),
            (lambda r: r["compounds"][0].update(activity=None), "explicit activity"),
            (lambda r: r["compounds"][1].update(activity=2), "water"),
            (lambda r: r["route_weights"].update(missing=1), "Route weights"),
            (lambda r: r["compounds"][1].update(accession="kegg:C00002"), "accession"),
            (lambda r: r["compounds"][1].update(accession="kegg:C00305"), "pMg"),
            (
                lambda r: r["reactions"][0].update(stoichiometry={"atp": 1, "adp": 1}),
                "negative coefficients",
            ),
        ],
    )
    def test_a_request_outside_the_profile_is_refused_in_plain_words(
        self, tmp_path, change, message
    ):
        params = atp()
        change(params)
        outcome = run(params, tmp_path)
        assert outcome.steps["thermo"].status == "failed"
        assert message in outcome.steps["thermo"].error

    def test_a_proton_given_an_activity_is_refused_because_ph_sets_it(self, tmp_path):
        params = atp()
        params["compounds"].append(
            {"id": "proton", "accession": "kegg:C00080", "activity": 1e-7}
        )
        assert "pH" in run(params, tmp_path).steps["thermo"].error


class TestTheModel:
    def test_initialising_the_model_downloads_nothing(self):
        from alambiq.capabilities.investigate.science.thermodynamics import local_model

        with patch(
            "pooch.retrieve", side_effect=AssertionError("offline initialization")
        ):
            model, identity = local_model()
        assert model.get_compound("kegg:C00002") is not None
        assert identity["data_license"] == "CC-BY-4.0"

    def test_a_data_file_that_differs_from_the_pinned_one_is_refused(self):
        from alambiq.capabilities.investigate.science import thermodynamics

        with (
            patch.object(thermodynamics, "ASSETS", {"mini_compounds.sqlite": "0" * 64}),
            pytest.raises(ValueError, match="exact asset"),
        ):
            thermodynamics.local_model()

    def test_the_package_versions_that_ran_are_recorded_not_enforced(self):
        from alambiq.capabilities.investigate.science.thermodynamics import local_model

        _, identity = local_model()
        assert all(isinstance(v, str) and v for v in identity["packages"].values())


class TestFromARoute:
    def campaign(self, tmp_path, compounds=None, route_index=0):
        from tests.core.test_reaction_explore_routes import hydrolysis

        mapping = compounds or [
            {"smiles": "O", "accession": "kegg:C00001", "activity": 1},
            {"smiles": "CC(=O)O", "accession": "kegg:C00033", "activity": 0.001},
            {"smiles": "CO", "accession": "kegg:C00132", "activity": 0.001},
            {
                "smiles": "COC(C)=O",
                "accession": "local:unassigned-methyl-acetate",
                "activity": 0.001,
            },
        ]
        return {
            "id": "ester-route-energy",
            "steps": [
                {
                    "id": "routes",
                    "capability": "reaction.explore_routes",
                    "params": hydrolysis(),
                },
                {
                    "id": "thermo",
                    "capability": "reaction.estimate_thermodynamics",
                    "bind": {"route": "routes.record"},
                    "params": {
                        "id": "ester-route-energy",
                        "conditions": atp()["conditions"],
                        "assumption_note": "Illustrative millimolar activities.",
                        "route_index": route_index,
                        "route_compounds": mapping,
                    },
                },
            ],
        }

    def test_a_generated_route_becomes_a_conditioned_estimate_and_names_what_it_cannot_resolve(
        self, tmp_path
    ):
        pytest.importorskip("doranet")
        from alambiq.api import run as execute

        outcome = execute(self.campaign(tmp_path), run_dir=tmp_path / "chain")
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        result = outcome.outputs("thermo")
        # The methyl acetate accession is not in the model's database, so the one step is unresolved.
        assert (
            result["route"]["status"] == "unresolved-steps"
            and len(result["reactions"]) == 1
        )
        assert len(result["compounds"]) == 4
        stored = json.loads(Path(result["record"]).read_text())
        assert stored["route_source"]["route_id"].startswith("route-")

    def test_a_participant_without_an_aqueous_identity_is_refused(self, tmp_path):
        pytest.importorskip("doranet")
        from alambiq.api import run as execute

        mapping = [
            {"smiles": "O", "accession": "kegg:C00001", "activity": 1},
            {"smiles": "CC(=O)O", "accession": "kegg:C00033", "activity": 0.001},
            {"smiles": "CO", "accession": "kegg:C00132", "activity": 0.001},
        ]
        outcome = execute(self.campaign(tmp_path, mapping), run_dir=tmp_path / "chain")
        assert "every route participant" in outcome.steps["thermo"].error

    def test_a_route_index_that_does_not_exist_is_refused(self, tmp_path):
        pytest.importorskip("doranet")
        from alambiq.api import run as execute

        outcome = execute(
            self.campaign(tmp_path, route_index=7), run_dir=tmp_path / "chain"
        )
        assert "existing zero-based route_index" in outcome.steps["thermo"].error

    def test_both_ways_of_naming_equations_together_are_refused(self, tmp_path):
        params = atp()
        params["route"] = str(tmp_path / "x.json")
        params["route_index"] = 0
        params["route_compounds"] = [
            {"smiles": "O", "accession": "kegg:C00001", "activity": 1}
        ]
        assert "exactly one of the two groups" in run(params, tmp_path).steps["thermo"].error

    def test_neither_way_is_refused(self, tmp_path):
        params = {k: atp()[k] for k in ("id", "conditions", "assumption_note")}
        assert "exactly one of the two groups" in run(params, tmp_path).steps["thermo"].error
