"""Substitution free energies: the checks, the atom mapping, the analysis, and real runs with known answers.

Three groups of tests, from the cheapest:

* the request checks and the analysis arithmetic need no engine. The analysis combines two legs
  by subtraction, so synthetic legs with chosen estimates give ΔΔG, its standard errors and its
  verdicts exactly;
* the atom mapping between two residues is a rule over atom names and bonds, tested on
  residues written out by hand, whose mapped and unique atoms can be counted on paper;
* the real runs use the null substitution (the mutant state equals the wild type). Both states
  then have the same Hamiltonian, so every lambda window has the same energy in every
  configuration, MBAR returns exactly zero with a vanishing error, and the overlap between any
  two windows is 1/K for K windows (the iterative MBAR solver returns zero to within 1e-4 kcal/mol).
  These hold for the folded leg and the reference leg and test
  the pipeline (preparation, solvation, hybrid system, replica exchange, MBAR, gathering) for
  plumbing errors, not the force field. They run on 5 windows and 10 iterations of 0.04 ps, which
  test the machinery and nothing about convergence. Set ALAMBIQ_ENV_OPENFE to the Python of the
  pinned OpenFE environment to run them.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from alambiq.capabilities.investigate.science import mutation

ENV = os.environ.get("ALAMBIQ_ENV_OPENFE", "")
ENGINES = Path(__file__).resolve().parents[2] / "src/alambiq/engines"
STRUCTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/1L2Y.cif"


def load_engine():
    spec = importlib.util.spec_from_file_location("openmm_mutation", ENGINES / "openmm_mutation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["openmm_mutation"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------ requests


class TestRequestChecks:
    @pytest.mark.parametrize(
        ("value", "name"),
        [("V", "VAL"), ("val", "VAL"), (" Ile ", "ILE"), ("W", "TRP"), ("c", "CYS")],
    )
    def test_codes_of_substitutable_residues_become_three_letter_names(self, value, name):
        assert mutation.residue_name(value) == name

    @pytest.mark.parametrize("value", ["G", "P", "D", "E", "K", "R", "H", "GLY", "PRO", "ASP", "HIS", "X", ""])
    def test_glycine_proline_charged_residues_and_histidine_are_refused(self, value):
        with pytest.raises(ValueError, match="to is"):
            mutation.residue_name(value)

    PREPARE = {
        "structure": "x.pdb", "chain_id": "A", "residue_number": 4, "ph": 7.0, "folded_padding_nm": 1.0,
        "reference_padding_nm": 1.2, "ionic_strength_molar": 0.15, "seed": 1, "timeout_seconds": 3600,
    }  # fmt: skip

    def test_valid_preparation_settings_pass(self):
        mutation.check_preparation(**self.PREPARE)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"structure": "x.cif"}, "extension .pdb or .ent"),
            ({"chain_id": " "}, "chain_id is empty"),
            ({"ph": 15.0}, "ph requires"),
            ({"folded_padding_nm": 0.5}, "folded_padding_nm requires"),
            ({"reference_padding_nm": 4.0}, "reference_padding_nm requires"),
            ({"ionic_strength_molar": 1.5}, "ionic_strength_molar requires"),
            ({"seed": -1}, "seed requires"),
            ({"timeout_seconds": 10}, "timeout_seconds requires"),
            ({"ph": float("nan")}, "ph requires"),
        ],
    )
    def test_preparation_settings_that_cannot_run_are_refused_by_name(self, changes, message):
        with pytest.raises(ValueError, match=message):
            mutation.check_preparation(**{**self.PREPARE, **changes})

    SAMPLING = {
        "replicas": 3, "lambda_windows": 12, "temperature_kelvin": 300.0, "pressure_bar": 1.0, "equilibration_ps": 50.0,
        "production_ps": 1000.0, "time_per_iteration_ps": 1.0, "threads": 8, "timeout_seconds": 3600,
    }  # fmt: skip

    def test_valid_sampling_settings_pass(self):
        mutation.check_sampling(**self.SAMPLING)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"replicas": 0}, "replicas requires"),
            ({"replicas": 11}, "replicas requires"),
            ({"lambda_windows": 4}, "lambda_windows requires"),
            ({"lambda_windows": 42}, "lambda_windows requires"),
            ({"threads": 0}, "threads requires"),
            ({"timeout_seconds": 10}, "timeout_seconds requires"),
            ({"bootstraps": 1001}, "bootstraps requires"),
            ({"forward_reverse_samples": 1}, "forward_reverse_samples requires"),
            ({"temperature_kelvin": 200.0}, "temperature_kelvin requires"),
            ({"pressure_bar": 0.0}, "pressure_bar requires"),
            ({"production_ps": float("nan")}, "production_ps requires"),
            ({"time_per_iteration_ps": 0.003}, "whole number of 2 fs steps"),
            ({"production_ps": 1000.5}, "whole number of iterations"),
            ({"equilibration_ps": 50.5}, "whole number of iterations"),
        ],
    )
    def test_sampling_settings_that_cannot_run_are_refused_by_name(self, changes, message):
        with pytest.raises(ValueError, match=message):
            mutation.check_sampling(**{**self.SAMPLING, **changes})


# ------------------------------------------------------------------------------ the analysis


def replica(number, delta_g, error, overlap=0.1, effective=50.0, forward=None, reverse=None):
    matrix = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        matrix[i][i] = 1 - 2 * overlap
        if i + 1 < 3:
            matrix[i][i + 1] = matrix[i + 1][i] = overlap
    series = None
    if forward is not None:
        series = {
            "fractions": [0.1, 0.5, 1.0],
            "forward_kcal_mol": [0.0, forward, delta_g],
            "reverse_kcal_mol": [0.0, reverse, delta_g],
            "forward_error_kcal_mol": [0.1] * 3,
            "reverse_error_kcal_mol": [0.1] * 3,
        }
    return {
        "id": f"replica-{number}", "trajectory_sha256": "0" * 64, "delta_g_kcal_mol": delta_g,
        "mbar_error_kcal_mol": error, "overlap_matrix": matrix, "effective_samples": [effective] * 3,
        "forward_reverse": series, "sampling_seconds": 1.0, "atoms": 100,
    }  # fmt: skip


def leg(name, estimates, errors=None, mutant="VAL", temperature=300.0, **kwargs):
    errors = errors or [0.1] * len(estimates)
    return {
        "schema_version": "0.1", "leg": name, "wild_type_residue": "ILE", "mutant_residue": mutant,
        "temperature_kelvin": temperature, "inputs_sha256": {}, "method": {"platform": "CPU"},
        "replicas": [replica(i + 1, v, e, **kwargs) for i, (v, e) in enumerate(zip(estimates, errors, strict=True))],
    }  # fmt: skip


def analyze(tmp_path, folded, reference, checks=None):
    from alambiq.api import run as execute

    def paths(name, records):
        records = records if isinstance(records, list) else [records]
        out = []
        for i, record in enumerate(records):
            path = tmp_path / f"{name}{i}.json"
            path.write_text(json.dumps(record))
            out.append(str(path))
        return out

    params = {"folded": paths("folded", folded), "reference": paths("reference", reference)}
    if checks is not None:
        params["checks"] = checks
    return execute(
        {"id": "a", "steps": [{"id": "stab", "capability": "simulation.analyze_stability_change", "params": params}]},
        run_dir=tmp_path / "run",
    )


def analysis(tmp_path, folded, reference, checks=None):
    outcome = analyze(tmp_path, folded, reference, checks)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("stab"), outcome


class TestStabilityAnalysis:
    # Three independent preparations of each leg, one replica each.
    FOLDED = [leg("folded", [v]) for v in (-9.0, -9.5, -10.0)]
    REFERENCE = [leg("reference", [v]) for v in (-9.8, -10.0, -10.2)]

    def test_the_change_in_folding_free_energy_is_the_difference_of_the_leg_means(self, tmp_path):
        out, _ = analysis(tmp_path, self.FOLDED, self.REFERENCE)
        assert out["ddg_fold_kcal_mol"] == pytest.approx(-9.5 - (-10.0))
        assert out["mutation"] == "ILE>VAL" and out["temperature_kelvin"] == 300.0

    def test_the_standard_error_combines_the_spread_across_the_preparations_of_both_legs(self, tmp_path):
        out, _ = analysis(tmp_path, self.FOLDED, self.REFERENCE)
        # Sample standard deviations 0.5 and 0.2 of the preparation means, 3 preparations each.
        assert out["preparation_standard_error_kcal_mol"] == pytest.approx(math.sqrt(0.5**2 / 3 + 0.2**2 / 3))

    def test_the_mean_of_a_leg_weights_its_preparations_equally(self, tmp_path):
        two_replicas = leg("folded", [-9.0, -11.0])  # preparation mean -10
        one_replica = leg("folded", [-8.0])  # preparation mean -8
        out, _ = analysis(tmp_path, [two_replicas, one_replica], self.REFERENCE)
        assert out["legs"][0]["mean_delta_g_kcal_mol"] == pytest.approx(-9.0)
        assert out["legs"][0]["preparations"] == 2 and out["legs"][0]["replicas"] == 3
        assert out["legs"][0]["preparation_sd_kcal_mol"] == pytest.approx(math.sqrt(2))

    def test_the_mbar_error_combines_the_replica_errors_of_the_preparation_means(self, tmp_path):
        folded = [leg("folded", [-9.0, -9.2], errors=[0.1, 0.3]), leg("folded", [-9.4], errors=[0.2])]
        out, _ = analysis(tmp_path, folded, self.REFERENCE)
        # Error of the mean of a preparation: sqrt(sum of squared errors) / replicas; of the leg: sqrt(sum of squares) / preparations.
        first = math.sqrt(0.1**2 + 0.3**2) / 2
        leg_error = math.sqrt(first**2 + 0.2**2) / 2
        reference_error = math.sqrt(3 * 0.1**2) / 3
        assert out["mbar_error_kcal_mol"] == pytest.approx(math.sqrt(leg_error**2 + reference_error**2))

    def test_a_single_preparation_has_no_preparation_standard_error_and_the_notes_say_why(self, tmp_path):
        folded, reference = leg("folded", [-9.0, -9.4, -9.2]), leg("reference", [-10.0, -10.2, -9.8])
        out, outcome = analysis(tmp_path, folded, reference)
        assert out["preparation_standard_error_kcal_mol"] is None
        assert out["ddg_fold_kcal_mol"] == pytest.approx(-9.2 - (-10.0))
        assert out["legs"][0]["replica_sd_kcal_mol"] == pytest.approx(0.2)
        assert "share their starting coordinates" in " ".join(outcome.steps["stab"].notes)

    def test_each_leg_reports_its_diagnostics(self, tmp_path):
        folded = [leg("folded", [-9.0, -9.5], overlap=0.05, effective=20.0, forward=-9.2, reverse=-9.9)]
        out, _ = analysis(tmp_path, folded, self.REFERENCE)
        summary = {x["leg"]: x for x in out["legs"]}
        assert summary["folded"]["minimum_adjacent_overlap"] == pytest.approx(0.05)
        assert summary["folded"]["minimum_effective_samples"] == 20.0
        assert summary["folded"]["forward_reverse_gap_kcal_mol"] == pytest.approx(0.7)
        assert summary["reference"]["forward_reverse_gap_kcal_mol"] is None
        assert summary["folded"]["replica_sd_kcal_mol"] == pytest.approx(math.sqrt(0.5**2 / 2))

    def test_checks_become_verdicts_and_the_overall_status(self, tmp_path):
        folded = [leg("folded", [-9.0, -10.0], overlap=0.05, forward=-9.2, reverse=-9.9) for _ in range(2)]
        checks = {
            "minimum_adjacent_overlap": 0.03,
            "maximum_replica_sd_kcal_mol": 0.4,
            "maximum_forward_reverse_gap_kcal_mol": 1.0,
            "maximum_preparation_standard_error_kcal_mol": 1.0,
        }
        out, _ = analysis(tmp_path, folded, self.REFERENCE, checks)
        by = {(v["diagnostic"], v["subject"]): v["status"] for v in out["verdicts"]}
        assert by[("minimum_adjacent_overlap", "folded")] == "pass"
        assert by[("replica_sd_kcal_mol", "folded")] == "fail"
        assert by[("replica_sd_kcal_mol", "reference")] == "unassessed"
        assert by[("forward_reverse_gap_kcal_mol", "folded")] == "pass"
        assert by[("forward_reverse_gap_kcal_mol", "reference")] == "unassessed"
        # Two preparations with equal means have no spread: the standard error of the cycle is that of the reference alone.
        assert by[("preparation_standard_error_kcal_mol", "cycle")] == "pass"
        assert out["status"] == "fail"

    def test_without_checks_the_status_is_unassessed(self, tmp_path):
        out, _ = analysis(tmp_path, self.FOLDED, self.REFERENCE)
        assert out["verdicts"] == [] and out["status"] == "unassessed"

    def test_all_passing_checks_give_pass(self, tmp_path):
        both = {"minimum_adjacent_overlap": 0.03, "maximum_preparation_standard_error_kcal_mol": 1.0}
        out, _ = analysis(tmp_path, self.FOLDED, self.REFERENCE, both)
        assert out["status"] == "pass"

    @pytest.mark.parametrize(
        ("folded", "reference", "message"),
        [
            (leg("reference", [-9.0]), leg("reference", [-10.0]), "names the leg 'reference'"),
            (leg("folded", [-9.0]), leg("folded", [-10.0]), "names the leg 'folded'"),
            (leg("folded", [-9.0], mutant="ALA"), leg("reference", [-10.0]), "different substitutions"),
            ([leg("folded", [-9.0]), leg("folded", [-9.5], mutant="ALA")], leg("reference", [-10.0]), "different substitutions"),
            (leg("folded", [-9.0], temperature=310.0), leg("reference", [-10.0]), "differ in temperature"),
        ],
    )
    def test_legs_that_do_not_belong_together_are_refused(self, tmp_path, folded, reference, message):
        outcome = analyze(tmp_path, folded, reference)
        assert not outcome.ok
        assert message in outcome.steps["stab"].error

    def test_a_record_without_replicas_is_refused(self, tmp_path):
        empty = leg("folded", [-9.0])
        empty["replicas"] = []
        outcome = analyze(tmp_path, empty, leg("reference", [-10.0]))
        assert "no replicas" in outcome.steps["stab"].error

    def test_an_empty_list_of_records_is_refused(self, tmp_path):
        outcome = analyze(tmp_path, [], leg("reference", [-10.0]))
        assert "1 to 50 are required" in outcome.steps["stab"].error


# ------------------------------------------------------------------------------ atom mapping


def residue(spec: str):
    """Atoms of a residue from lines `NAME ELEMENT bonded,names`."""
    atoms = []
    for line in spec.strip().splitlines():
        name, element, *rest = line.split()
        atoms.append({"name": name, "element": element, "bonds": rest[0].split(",") if rest else []})
    table = {a["name"]: a for a in atoms}
    for a in atoms:
        for b in a["bonds"]:
            if a["name"] not in table[b]["bonds"]:
                table[b]["bonds"].append(a["name"])
    return atoms


BACKBONE = """
N N CA,H
H H
CA C C,HA,CB
HA H
C C O
O O
"""
ALA = residue(BACKBONE + "CB C HB1,HB2,HB3\nHB1 H\nHB2 H\nHB3 H\n".replace("CB C HB1", "CB C CA,HB1"))
VAL = residue(
    BACKBONE
    + "CB C CA,HB,CG1,CG2\nHB H\nCG1 C CB,HG11,HG12,HG13\nHG11 H\nHG12 H\nHG13 H\n"
    + "CG2 C CB,HG21,HG22,HG23\nHG21 H\nHG22 H\nHG23 H\n"
)
ILE = residue(
    BACKBONE
    + "CB C CA,HB,CG1,CG2\nHB H\nCG1 C CB,HG12,HG13,CD1\nHG12 H\nHG13 H\n"
    + "CD1 C HD11,HD12,HD13\nHD11 H\nHD12 H\nHD13 H\n"
    + "CG2 C CB,HG21,HG22,HG23\nHG21 H\nHG22 H\nHG23 H\n"
)
LEU = residue(
    BACKBONE
    + "CB C CA,HB2,HB3,CG\nHB2 H\nHB3 H\nCG C HG,CD1,CD2\nHG H\n"
    + "CD1 C HD11,HD12,HD13\nHD11 H\nHD12 H\nHD13 H\nCD2 C HD21,HD22,HD23\nHD21 H\nHD22 H\nHD23 H\n"
)
SER = residue(BACKBONE + "CB C CA,HB2,HB3,OG\nHB2 H\nHB3 H\nOG O HG\nHG H\n")
THR = residue(
    BACKBONE
    + "CB C CA,HB,OG1,CG2\nHB H\nOG1 O HG1\nHG1 H\nCG2 C HG21,HG22,HG23\nHG21 H\nHG22 H\nHG23 H\n"
)
RING = "CG C CB,CD1,CD2\nCD1 C HD1,CE1\nHD1 H\nCD2 C HD2,CE2\nHD2 H\nCE1 C HE1,CZ\nHE1 H\nCE2 C HE2,CZ\nHE2 H\n"
PHE = residue(BACKBONE + "CB C CA,HB2,HB3,CG\nHB2 H\nHB3 H\n" + RING + "CZ C HZ\nHZ H\n")
TYR = residue(BACKBONE + "CB C CA,HB2,HB3,CG\nHB2 H\nHB3 H\n" + RING + "CZ C OH\nOH O HH\nHH H\n")


@pytest.fixture(scope="module")
def engine():
    return load_engine()


class TestAtomMapping:
    def test_isoleucine_to_valine_maps_the_backbone_cb_and_both_gamma_carbons(self, engine):
        plan = engine.plan_mapping(ILE, VAL)
        assert len(plan["core"]) == 15
        assert {"N", "CA", "C", "O", "CB", "CG1", "CG2"} <= set(plan["core"])
        assert sorted(plan["old_unique"]) == ["CD1", "HD11", "HD12", "HD13"]
        assert plan["new_unique"] == ["HG13"]

    def test_the_reverse_substitution_swaps_the_unique_sets(self, engine):
        plan = engine.plan_mapping(VAL, ILE)
        assert len(plan["core"]) == 15
        assert plan["old_unique"] == ["HG13"]
        assert sorted(plan["new_unique"]) == ["CD1", "HD11", "HD12", "HD13"]

    def test_leucine_to_alanine_keeps_the_backbone_cb_and_two_of_its_hydrogens(self, engine):
        plan = engine.plan_mapping(LEU, ALA)
        assert len(plan["core"]) == 9
        assert set(plan["core"]) == {"N", "H", "CA", "HA", "C", "O", "CB", "HB2", "HB3"}
        assert plan["core"]["HB2"] == "HB1" and plan["core"]["HB3"] == "HB2"
        assert len(plan["old_unique"]) == 10 and plan["new_unique"] == ["HB3"]

    def test_serine_to_threonine_does_not_map_atoms_of_different_names(self, engine):
        plan = engine.plan_mapping(SER, THR)
        assert set(plan["core"]) == {"N", "H", "CA", "HA", "C", "O", "CB", "HB2"}
        assert sorted(plan["old_unique"]) == ["HB3", "HG", "OG"]
        assert len(plan["new_unique"]) == 6 and "OG1" in plan["new_unique"]

    def test_phenylalanine_to_tyrosine_maps_the_ring_and_leaves_the_hydroxyl_new(self, engine):
        plan = engine.plan_mapping(PHE, TYR)
        assert len(plan["core"]) == 19
        assert plan["old_unique"] == ["HZ"]
        assert sorted(plan["new_unique"]) == ["HH", "OH"]

    def test_an_atom_is_mapped_only_when_bonded_to_the_atom_it_was_reached_from_in_both_residues(self, engine):
        # CG sits on CB in the first residue and on CA in the second: not the same atom position.
        first = residue(BACKBONE + "CB C CA,CG\nCG C\n")
        second = residue(BACKBONE + "CB C CA\nCG C CA\n")
        assert "CG" not in engine.plan_mapping(first, second)["core"]

    def test_a_residue_without_backbone_atoms_cannot_be_mapped(self, engine):
        with pytest.raises(ValueError, match="backbone atoms"):
            engine.plan_mapping(VAL, residue("CB C\nCG C CB\n"))

    def test_an_atom_of_the_same_name_and_another_element_is_not_mapped(self, engine):
        odd = residue(BACKBONE + "CB C CA,CG\nCG N\n")
        plan = engine.plan_mapping(odd, residue(BACKBONE + "CB C CA,CG\nCG C\n"))
        assert "CG" not in plan["core"]


# ------------------------------------------------------------------------------ real engine


needs_engine = pytest.mark.skipif(not ENV, reason="needs ALAMBIQ_ENV_OPENFE (the OpenFE environment)")


def write_pdb(tmp_path_factory):
    gemmi = pytest.importorskip("gemmi")
    structure = gemmi.read_structure(str(STRUCTURE))
    structure.setup_entities()
    del structure[1:]
    path = tmp_path_factory.mktemp("structure") / "1L2Y_model1.pdb"
    structure.write_pdb(str(path))
    return path


def execute(steps, tmp_path, name="run"):
    from alambiq.api import run as execute_campaign

    return execute_campaign({"id": "m", "steps": steps}, run_dir=tmp_path / name)


def prepare_step(structure, **params):
    return {
        "id": "prep",
        "capability": "structure.prepare_mutation_inputs",
        "params": {"structure": str(structure), "chain_id": "A", "residue_number": 4, "to": "V", **params},
    }


@pytest.fixture(scope="module")
def pdb_file(tmp_path_factory):
    return write_pdb(tmp_path_factory)


def archive_names(path):
    with tarfile.open(path, "r:gz") as bundle:
        return sorted(bundle.getnames())


@needs_engine
class TestPreparation:
    @pytest.fixture(scope="class")
    def outcome(self, pdb_file, tmp_path_factory):
        outcome = execute([prepare_step(pdb_file)], tmp_path_factory.mktemp("prepare"))
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        return outcome

    def test_the_legs_map_fifteen_atoms_and_decouple_the_rest(self, outcome):
        out = outcome.outputs("prep")
        assert out["mutation"] == "ILE4VAL" and out["wild_type"] == "ILE" and out["mutant"] == "VAL"
        for leg in (out["folded"], out["reference"]):
            assert leg["core_atoms"] == 15
            assert sorted(leg["wild_type_unique_atoms"]) == ["CD1", "HD11", "HD12", "HD13"]
            assert leg["mutant_unique_atoms"] == ["HG13"]
            assert leg["net_charge"] == pytest.approx(0.0, abs=1e-6)

    def test_the_reference_leg_is_smaller_than_the_folded_leg(self, outcome):
        out = outcome.outputs("prep")
        assert out["reference"]["atoms"] < out["folded"]["atoms"]
        assert out["folded"]["padding_nm"] == 1.0 and out["reference"]["padding_nm"] == 1.2

    def test_each_leg_is_one_archive_of_its_three_files(self, outcome):
        out = outcome.outputs("prep")
        for key in ("folded_inputs", "reference_inputs"):
            assert archive_names(out[key]) == ["leg.json", "mutant.pdb", "wild_type.pdb"]

    def test_the_input_file_is_identified_by_its_digest(self, outcome, pdb_file):
        import hashlib

        assert outcome.outputs("prep")["structure_sha256"] == hashlib.sha256(pdb_file.read_bytes()).hexdigest()

    def test_the_same_seed_gives_the_same_atoms_in_the_same_order(self, pdb_file, tmp_path):
        def atoms(name):
            outcome = execute([prepare_step(pdb_file, seed=5)], tmp_path, name)
            assert outcome.ok
            with tarfile.open(outcome.outputs("prep")["folded_inputs"]) as bundle:
                lines = bundle.extractfile("wild_type.pdb").read().decode().splitlines()
            return [
                (line[12:26], tuple(float(line[i : i + 8]) for i in (30, 38, 46)))
                for line in lines
                if line.startswith(("ATOM", "HETATM"))
            ]

        one, two = atoms("one"), atoms("two")
        assert [name for name, _ in one] == [name for name, _ in two]
        moved = [max(abs(a - b) for a, b in zip(p, q, strict=True)) for (_, p), (_, q) in zip(one, two, strict=True)]
        # The relaxation sums forces on several threads, so the coordinates of the solvent
        # differ between runs; the seed fixes the atoms, their order and the box.
        assert sorted(moved)[len(moved) // 2] < 0.5

    def test_another_seed_places_it_differently(self, pdb_file, tmp_path):
        import hashlib

        digests = []
        for seed in (5, 6):
            outcome = execute([prepare_step(pdb_file, seed=seed)], tmp_path, f"s{seed}")
            with tarfile.open(outcome.outputs("prep")["folded_inputs"]) as bundle:
                digests.append(hashlib.sha256(bundle.extractfile("wild_type.pdb").read()).hexdigest())
        assert digests[0] != digests[1]

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"to": "G"}, "to is"),
            ({"to": "D"}, "to is"),
            ({"to": "I"}, "already ILE"),
            ({"residue_number": 8}, "substitutable residues"),
            ({"residue_number": 2}, "fewer than 2 residues on one side"),
            ({"residue_number": 99}, "no such residue"),
            ({"chain_id": "Z"}, "the file has the chains"),
        ],
    )
    def test_requests_that_cannot_be_prepared_are_refused_with_the_reason(self, pdb_file, tmp_path, params, message):
        outcome = execute([prepare_step(pdb_file, **params)], tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["prep"].error


def engine_call(function: str, cfg: dict) -> dict:
    """Call a function of the engine in the OpenFE environment and return its result."""
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(ENGINES)!r})\n"
        "import openmm_mutation as m\n"
        f"print('RESULT' + json.dumps(m.{function}(json.loads(sys.stdin.read()))))\n"
    )
    done = subprocess.run([ENV, "-c", script], input=json.dumps(cfg), capture_output=True, text=True, timeout=900)
    if done.returncode:
        raise ValueError(done.stderr.strip().splitlines()[-1])
    return json.loads(next(line for line in done.stdout.splitlines() if line.startswith("RESULT"))[6:])


@needs_engine
class TestRelaxation:
    SCRIPT = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(ENGINES)!r})\n"
        "import numpy as np, openmm\n"
        "from openmm import app, unit\n"
        "import openmm_mutation as m\n"
        "modeller = app.Modeller(app.Topology(), [] * unit.nanometer)\n"
        "modeller.addSolvent(m._forcefield(), model='tip3p', boxSize=openmm.Vec3(2.2, 2.2, 2.2) * unit.nanometer)\n"
        "positions = np.array(modeller.positions.value_in_unit(unit.nanometer))\n"
        "waters = [r for r in modeller.topology.residues() if r.name == 'HOH']\n"
        "first, second = [{a.name: a.index for a in w.atoms()} for w in waters[:2]]\n"
        "positions[second['O']] = positions[first['O']] + 0.02\n"
        "def worst(topology, xyz):\n"
        "    context = openmm.Context(m.create_system(topology), openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName('CPU'))\n"
        "    context.setPositions(xyz)\n"
        "    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)\n"
        "    return float(np.linalg.norm(forces, axis=1).max())\n"
        "before = worst(modeller.topology, positions * unit.nanometer)\n"
        "relaxed = m.relax(modeller.topology, positions * unit.nanometer)\n"
        "print('RESULT' + json.dumps([before, worst(modeller.topology, relaxed)]))\n"
    )

    def test_two_overlapping_waters_are_pulled_apart_before_the_sampler_starts(self):
        done = subprocess.run([ENV, "-c", self.SCRIPT], capture_output=True, text=True, timeout=900)
        assert done.returncode == 0, done.stderr[-500:]
        before, after = json.loads(next(line for line in done.stdout.splitlines() if line.startswith("RESULT"))[6:])
        assert before > 1e5
        assert after < 1e4


@needs_engine
class TestBoxWidth:
    SCRIPT = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(ENGINES)!r})\n"
        "import openmm_mutation as m\n"
        "modeller, _ = m.read_chain(sys.argv[1], 'A')\n"
        "modeller = m.complete_terminal_atoms(modeller)\n"
        "index = m.residue_index(modeller.topology, 4)\n"
        "piece, local = m.fragment(modeller, index)\n"
        "solvated, padding = m.solvate(piece, 0.3, 0.15, 1, 7.0, m.variants_for(piece.topology, {}))\n"
        "print('RESULT' + json.dumps([padding, min(m.box_widths(solvated.topology.getPeriodicBoxVectors())), m.MIN_BOX_WIDTH_NM]))\n"
    )

    def test_a_padding_that_gives_a_narrow_box_is_widened_until_the_cutoff_fits(self, pdb_file):
        done = subprocess.run([ENV, "-c", self.SCRIPT, str(pdb_file)], capture_output=True, text=True, timeout=900)
        assert done.returncode == 0, done.stderr[-500:]
        padding, width, minimum = json.loads(next(line for line in done.stdout.splitlines() if line.startswith("RESULT"))[6:])
        assert padding > 0.3
        assert width >= minimum

    def test_the_widths_of_a_cubic_box_are_its_edges(self):
        done = subprocess.run(
            [
                ENV,
                "-c",
                "import sys, json\n"
                f"sys.path.insert(0, {str(ENGINES)!r})\n"
                "import openmm_mutation as m\n"
                "from openmm import Vec3, unit\n"
                "v = [Vec3(2, 0, 0) * unit.nanometer, Vec3(0, 3, 0) * unit.nanometer, Vec3(0, 0, 4) * unit.nanometer]\n"
                "print('RESULT' + json.dumps(m.box_widths(v)))\n",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stderr[-500:]
        widths = json.loads(next(line for line in done.stdout.splitlines() if line.startswith("RESULT"))[6:])
        assert widths == pytest.approx([2.0, 3.0, 4.0])


@needs_engine
class TestEngineChecks:
    def cfg(self, pdb_file, tmp_path, **changes):
        return {
            "structure": str(pdb_file), "chain_id": "A", "residue_number": 4, "to": "VAL", "ph": 7.0,
            "folded_padding_nm": 1.0, "reference_padding_nm": 1.2, "ionic_strength_molar": 0.15, "seed": 1,
            "identity_check": False, "output": str(tmp_path / "out"), **changes,
        }  # fmt: skip

    def test_a_substitution_that_changes_the_net_charge_is_refused(self, pdb_file, tmp_path):
        with pytest.raises(ValueError, match="changes the net charge"):
            engine_call("prepare", self.cfg(pdb_file, tmp_path, to="ASP"))

    def test_a_chain_missing_heavy_atoms_is_refused_with_the_residues(self, pdb_file, tmp_path):
        lines = [ln for ln in pdb_file.read_text().splitlines() if not (ln.startswith("ATOM") and ln[12:16].strip() == "CG" and ln[22:26].strip() == "6")]
        broken = tmp_path / "broken.pdb"
        broken.write_text("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="lacks heavy atoms of TRP6"):
            engine_call("prepare", self.cfg(broken, tmp_path))


def run_step(step_id, inputs, **params):
    return {
        "id": step_id,
        "capability": "simulation.run_mutation_free_energy",
        "bind": {"inputs": inputs},
        "params": {
            "replicas": 2, "lambda_windows": 5, "equilibration_ps": 0.04, "production_ps": 0.4,
            "time_per_iteration_ps": 0.04, "platform": "CPU", "threads": 8, "bootstraps": 0,
            "forward_reverse_samples": 2, **params,
        },
    }  # fmt: skip


@needs_engine
class TestNullSubstitution:
    """Both states of each leg are the wild type, so every answer is known."""

    @pytest.fixture(scope="class")
    def outcome(self, pdb_file, tmp_path_factory):
        steps = [
            prepare_step(pdb_file, identity_check=True, to="L"),
            run_step("folded", "prep.folded_inputs"),
            run_step("reference", "prep.reference_inputs"),
            {
                "id": "stab",
                "capability": "simulation.analyze_stability_change",
                "bind": {"folded.0": "folded.leg_record", "reference.0": "reference.leg_record"},
                "params": {"checks": {"minimum_adjacent_overlap": 0.1}},
            },
        ]
        outcome = execute(steps, tmp_path_factory.mktemp("null"))
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        return outcome

    @pytest.mark.parametrize("leg", ["folded", "reference"])
    def test_a_null_substitution_costs_nothing_in_every_replica(self, outcome, leg):
        out = outcome.outputs(leg)
        assert len(out["repeats"]) == 2
        for repeat in out["repeats"]:
            assert repeat["delta_g_kcal_mol"] == pytest.approx(0.0, abs=1e-4)
            assert repeat["mbar_error_kcal_mol"] == pytest.approx(0.0, abs=1e-4)
        assert out["mean_delta_g_kcal_mol"] == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("leg", ["folded", "reference"])
    def test_identical_windows_overlap_completely(self, outcome, leg):
        for repeat in outcome.outputs(leg)["repeats"]:
            assert repeat["minimum_adjacent_overlap"] == pytest.approx(1 / 5, abs=1e-3)

    def test_the_replicas_are_distinct_runs(self, outcome):
        record = json.loads(Path(outcome.outputs("folded")["leg_record"]).read_text())
        digests = {r["trajectory_sha256"] for r in record["replicas"]}
        assert len(digests) == 2

    def test_the_change_in_folding_free_energy_of_the_null_substitution_is_zero(self, outcome):
        out = outcome.outputs("stab")
        assert out["ddg_fold_kcal_mol"] == pytest.approx(0.0, abs=1e-4)
        assert out["mutation"] == "ILE>ILE"
        assert outcome.outputs("prep")["mutation"] == "ILE4ILE"
        assert out["status"] == "pass"
        assert out["preparation_standard_error_kcal_mol"] is None

    def test_the_run_states_its_platform_and_the_versions(self, outcome):
        out = outcome.outputs("folded")
        assert out["platform"] == "CPU" and out["precision"] == "native"
        assert {"openmm", "openfe", "openmmtools"} <= set(out["versions"])

    def test_a_short_run_says_so_and_cites_the_convergence_record(self, outcome):
        note = " ".join(outcome.steps["folded"].notes)
        assert "below 2500 ps" in note and "1.3 kcal/mol across independent preparations" in note

    def test_native_records_are_kept_when_asked_for(self, outcome):
        path = outcome.outputs("folded")["native_records"]
        assert path is not None and any(name.endswith(".nc") for name in archive_names(path))


@needs_engine
class TestRunOptions:
    @pytest.fixture(scope="class")
    def outcome(self, pdb_file, tmp_path_factory):
        steps = [
            prepare_step(pdb_file, identity_check=True),
            run_step("leg", "prep.reference_inputs", replicas=1, precision="double", keep_native_records=False),
        ]
        outcome = execute(steps, tmp_path_factory.mktemp("options"))
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        return outcome

    def test_without_native_records_none_are_kept(self, outcome):
        assert outcome.outputs("leg")["native_records"] is None

    def test_the_precision_applies_to_cuda_only_and_the_cpu_reports_native(self, outcome):
        out = outcome.outputs("leg")
        assert out["platform"] == "CPU" and out["precision"] == "native"

    def test_a_single_replica_has_no_spread(self, outcome):
        out = outcome.outputs("leg")
        assert len(out["repeats"]) == 1 and out["replica_sd_kcal_mol"] is None


@needs_engine
class TestRunRefusals:
    def test_an_input_that_is_not_a_leg_archive_is_refused(self, tmp_path):
        bogus = tmp_path / "leg.tar.gz"
        with tarfile.open(bogus, "w:gz") as bundle:
            note = tmp_path / "note.txt"
            note.write_text("x")
            bundle.add(note, arcname="note.txt")
        outcome = execute(
            [{"id": "r", "capability": "simulation.run_mutation_free_energy", "params": {"inputs": str(bogus)}}], tmp_path
        )
        assert not outcome.ok and "leg.json, mutant.pdb and wild_type.pdb" in outcome.steps["r"].error

    def test_a_file_that_is_not_an_archive_is_refused(self, tmp_path):
        text = tmp_path / "leg.txt"
        text.write_text("not an archive")
        outcome = execute(
            [{"id": "r", "capability": "simulation.run_mutation_free_energy", "params": {"inputs": str(text)}}], tmp_path
        )
        assert not outcome.ok and "not a leg archive" in outcome.steps["r"].error
