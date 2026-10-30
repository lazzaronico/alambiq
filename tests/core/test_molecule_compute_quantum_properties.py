"""molecule.compute_quantum_properties optimizes a molecule and computes its properties."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ENV = os.environ.get("ALAMBIQ_ENV_MOLECULAR_PROPERTIES", "")
HARTREE_TO_CAL_MOL = 627509.4740631
HC_EV_NM = 1239.841984332


def installed() -> bool:
    return bool(ENV) and Path(ENV).is_file()


engine_only = pytest.mark.skipif(
    not installed(),
    reason="set ALAMBIQ_ENV_MOLECULAR_PROPERTIES to run the engine tests",
)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_MOLECULAR_PROPERTIES", ENV or sys.executable)


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {
        "id": "props",
        "capability": "molecule.compute_quantum_properties",
        "params": params,
    }
    return execute({"id": "q", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def refusal(params: dict, tmp_path: Path) -> str:
    outcome = run(params, tmp_path)
    assert not outcome.ok
    return " ".join(x.error or "" for x in outcome.steps.values())


@pytest.mark.usefixtures("engine")
class TestRequestsAreRefusedBeforeAnythingRuns:
    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"smiles": ""}, "smiles must be 1 to 500 characters"),
            ({"smiles": "C C"}, "smiles must be 1 to 500 characters"),
            ({"smiles": "C" * 501}, "smiles must be 1 to 500 characters"),
            ({"smiles": "O", "properties": []}, "properties must list each"),
            (
                {"smiles": "O", "properties": ["orbitals", "orbitals"]},
                "properties must list each",
            ),
            ({"smiles": "O", "excited_states": 0}, "excited_states must be"),
            ({"smiles": "O", "excited_states": 11}, "excited_states must be"),
            ({"smiles": "O", "temperature_k": 0}, "temperature_k must be between"),
            ({"smiles": "O", "temperature_k": float("nan")}, "temperature_k must be"),
            ({"smiles": "O", "max_steps": 0}, "max_steps must be an integer"),
            ({"smiles": "O", "max_steps": 301}, "max_steps must be an integer"),
            ({"smiles": "O", "multiplicity": 0}, "multiplicity must be an integer"),
            ({"smiles": "O", "multiplicity": 8}, "multiplicity must be an integer"),
            ({"smiles": "O", "seed": -1}, "seed must be an integer"),
            ({"smiles": "O", "threads": 0}, "threads must be an integer"),
        ],
    )
    def test_a_setting_out_of_range_is_refused(self, tmp_path, params, message):
        assert message in refusal(params, tmp_path)


@engine_only
@pytest.mark.usefixtures("engine")
class TestTheMoleculeIsChecked:
    @pytest.mark.parametrize(
        ("smiles", "message"),
        [
            ("not a molecule((", "smiles must be"),
            ("C(C", "is not a SMILES string that RDKit can read"),
            (
                "[CH3]",
                "unpaired electrons (1), so the molecule is open-shell; set multiplicity to 2",
            ),
            ("[Na+].[Cl-]", "elements Na are not supported"),
            ("C" * 13, "13 heavy atoms; at most 12 are accepted"),
        ],
    )
    def test_a_molecule_the_method_cannot_treat_is_refused_with_the_reason(
        self, tmp_path, smiles, message
    ):
        assert message in refusal({"smiles": smiles}, tmp_path)

    def test_a_multiplicity_that_the_electron_count_cannot_have_is_refused(
        self, tmp_path
    ):
        message = refusal({"smiles": "O", "multiplicity": 2}, tmp_path)
        assert "multiplicity of 2 is impossible for 10 electrons" in message
        message = refusal({"smiles": "[OH]", "multiplicity": 3}, tmp_path)
        assert "multiplicity of 3 is impossible for 9 electrons" in message

    def test_a_geometry_that_does_not_converge_returns_no_properties(self, tmp_path):
        message = refusal({"smiles": "O", "max_steps": 1}, tmp_path)
        assert "did not converge in 1 steps" in message


@engine_only
@pytest.mark.usefixtures("engine")
class TestWater:
    """Water at B3LYP/6-31G*, for which published values exist."""

    @pytest.fixture(scope="class")
    def water(self, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_MOLECULAR_PROPERTIES"] = ENV
        tmp = tmp_path_factory.mktemp("water")
        return run_ok(
            {
                "smiles": "O",
                "properties": ["orbitals", "thermochemistry", "excitations"],
            },
            tmp,
        ).outputs("props")

    def test_the_energy_is_the_published_b3lyp_6_31g_star_value(self, water):
        assert water["energy_hartree"] == pytest.approx(-76.4089, abs=5e-4)

    def test_the_gap_is_the_lumo_minus_the_homo(self, water):
        o = water["orbitals"]
        assert o["gap_ev"] == pytest.approx(o["lumo_ev"] - o["homo_ev"])
        assert o["homo_ev"] == pytest.approx(o["highest_occupied_ev"][-1])
        assert o["lumo_ev"] == pytest.approx(o["lowest_virtual_ev"][0])
        assert o["occupied_orbitals"] == 5

    def test_the_heat_capacity_and_zero_point_energy_match_experiment_and_literature(
        self, water
    ):
        t = water["thermochemistry"]
        assert t["cv_cal_mol_k"] == pytest.approx(5.97, abs=0.1)
        assert t["cp_cal_mol_k"] - t["cv_cal_mol_k"] == pytest.approx(1.987, abs=0.01)
        assert t["zero_point_energy_kcal_mol"] == pytest.approx(13.3, abs=0.3)

    def test_the_vibrations_are_three_real_modes_and_none_imaginary(self, water):
        t = water["thermochemistry"]
        assert (t["vibrational_modes"], t["imaginary_modes"]) == (3, 0)
        assert all(1500 < f < 4000 for f in t["lowest_frequencies_cm1"])

    def test_the_symmetry_number_of_water_is_two(self, water):
        assert water["thermochemistry"]["symmetry_number"] == 2

    def test_the_thermodynamic_quantities_are_consistent(self, water):
        t = water["thermochemistry"]
        s = t["entropy_cal_mol_k"] / HARTREE_TO_CAL_MOL
        assert t["gibbs_energy_hartree"] == pytest.approx(
            t["enthalpy_hartree"] - t["temperature_k"] * s, abs=1e-6
        )
        assert t["energy_0k_hartree"] == pytest.approx(
            water["energy_hartree"] + t["zero_point_energy_hartree"], abs=1e-9
        )

    def test_the_excitations_are_ordered_and_their_wavelengths_follow_their_energies(
        self, water
    ):
        states = water["excitations"]
        assert [s["state"] for s in states] == [1, 2, 3]
        assert [s["energy_ev"] for s in states] == sorted(
            s["energy_ev"] for s in states
        )
        assert all(s["oscillator_strength"] >= 0 for s in states)
        for s in states:
            assert s["wavelength_nm"] == pytest.approx(HC_EV_NM / s["energy_ev"])

    def test_the_geometry_is_an_xyz_file_of_three_atoms(self, water):
        lines = Path(water["geometry"]).read_text().splitlines()
        assert lines[0] == "3"
        assert [line.split()[0] for line in lines[2:]] == ["O", "H", "H"]

    def test_the_record_keeps_the_settings_and_the_numbers(self, water):
        record = json.loads(Path(water["record"]).read_text())
        assert record["functional"] == "b3lyp" and record["basis"] == "6-31G*"
        assert record["cartesian_basis"] is True and record["grid_level"] == 4
        assert record["energy_hartree"] == pytest.approx(water["energy_hartree"])


@engine_only
@pytest.mark.usefixtures("engine")
class TestOptions:
    def test_only_the_requested_properties_are_computed(self, tmp_path):
        out = run_ok({"smiles": "C", "properties": ["orbitals"]}, tmp_path).outputs(
            "props"
        )
        assert out["orbitals"] is not None
        assert out["thermochemistry"] is None and out["excitations"] == []
        out = run_ok(
            {"smiles": "C", "properties": ["thermochemistry"]}, tmp_path, "t"
        ).outputs("props")
        assert out["orbitals"] is None and out["thermochemistry"] is not None

    def test_methane_has_the_tetrahedral_symmetry_number_and_nine_modes(self, tmp_path):
        t = run_ok(
            {"smiles": "C", "properties": ["thermochemistry"]}, tmp_path
        ).outputs("props")["thermochemistry"]
        assert t["symmetry_number"] == 12 and t["vibrational_modes"] == 9

    def test_ethylene_has_d2h_symmetry_and_the_requested_number_of_excited_states(
        self, tmp_path
    ):
        out = run_ok(
            {
                "smiles": "C=C",
                "properties": ["thermochemistry", "excitations"],
                "excited_states": 5,
            },
            tmp_path,
        ).outputs("props")
        assert out["thermochemistry"]["symmetry_number"] == 4
        assert len(out["excitations"]) == 5

    @pytest.mark.parametrize(
        ("smiles", "number"), [("O=C=O", 2), ("c1ccccc1", 12), ("CC(=O)O", 1)]
    )
    def test_symmetry_number_follows_the_point_group(self, tmp_path, smiles, number):
        t = run_ok(
            {"smiles": smiles, "properties": ["thermochemistry"]}, tmp_path
        ).outputs("props")["thermochemistry"]
        assert t["symmetry_number"] == number

    def test_heat_capacity_and_entropy_rise_with_temperature(self, tmp_path):
        values = {}
        for kelvin in (100.0, 500.0):
            t = run_ok(
                {
                    "smiles": "C=C",
                    "properties": ["thermochemistry"],
                    "temperature_k": kelvin,
                },
                tmp_path,
                f"t{int(kelvin)}",
            ).outputs("props")["thermochemistry"]
            values[kelvin] = t
        assert values[500.0]["cv_cal_mol_k"] > values[100.0]["cv_cal_mol_k"]
        assert values[500.0]["entropy_cal_mol_k"] > values[100.0]["entropy_cal_mol_k"]
        assert values[500.0]["temperature_k"] == 500.0

    def test_another_functional_and_another_basis_change_the_energy(self, tmp_path):
        base = run_ok({"smiles": "O"}, tmp_path, "a").outputs("props")["energy_hartree"]
        pbe = run_ok({"smiles": "O", "functional": "pbe"}, tmp_path, "b").outputs(
            "props"
        )
        svp = run_ok({"smiles": "O", "basis": "def2-SVP"}, tmp_path, "c").outputs(
            "props"
        )
        assert pbe["energy_hartree"] != pytest.approx(base, abs=1e-3)
        assert svp["energy_hartree"] != pytest.approx(base, abs=1e-3)
        assert (pbe["functional"], svp["basis"]) == ("pbe", "def2-SVP")

    def test_a_charged_closed_shell_ion_is_accepted(self, tmp_path):
        out = run_ok({"smiles": "[NH4+]"}, tmp_path).outputs("props")
        assert out["charge"] == 1 and out["formula"].startswith("H4N")

    def test_the_same_seed_gives_the_same_energy(self, tmp_path):
        a = run_ok({"smiles": "CCO", "seed": 7, "threads": 2}, tmp_path, "a")
        b = run_ok({"smiles": "CCO", "seed": 7, "threads": 4}, tmp_path, "b")
        assert a.outputs("props")["energy_hartree"] == pytest.approx(
            b.outputs("props")["energy_hartree"], abs=1e-7
        )

    def test_acetic_acid_converges_which_the_default_grid_does_not(self, tmp_path):
        out = run_ok({"smiles": "CC(=O)O", "max_steps": 30}, tmp_path).outputs("props")
        assert out["optimization_steps"] <= 30
        assert out["grid_level"] == 4

    def test_the_report_lists_each_requested_property(self, tmp_path):
        from alambiq.capabilities.investigate.molecule_compute_quantum_properties_report import (
            render,
        )

        outcome = run_ok(
            {
                "smiles": "O",
                "properties": ["orbitals", "thermochemistry", "excitations"],
            },
            tmp_path,
        )
        html = render(outcome.steps["props"].result)
        for text in (
            "H2O",
            "Highest occupied",
            "Zero-point energy",
            "Oscillator strength",
        ):
            assert text in html


@engine_only
@pytest.mark.usefixtures("engine")
class TestOpenShellMolecules:
    @pytest.fixture(scope="class")
    def radical(self, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_MOLECULAR_PROPERTIES"] = ENV
        tmp = tmp_path_factory.mktemp("hydroxyl")
        return run_ok(
            {
                "smiles": "[OH]",
                "multiplicity": 2,
                "properties": ["orbitals", "charges"],
            },
            tmp,
        ).outputs("props")

    def test_the_hydroxyl_radical_is_a_doublet_with_a_nearly_pure_spin_state(
        self, radical
    ):
        assert radical["multiplicity"] == 2
        assert radical["expected_spin_squared"] == pytest.approx(0.75)
        assert radical["spin_squared"] == pytest.approx(0.75, abs=0.02)

    def test_the_spin_sits_on_the_oxygen_and_sums_to_one_unpaired_electron(
        self, radical
    ):
        spins = {
            a["element"]: a["spin_population"] for a in radical["charges"]["mulliken"]
        }
        assert spins["O"] > 0.9
        assert sum(spins.values()) == pytest.approx(1.0, abs=1e-6)

    def test_each_spin_has_its_own_frontier_orbitals(self, radical):
        o = radical["orbitals"]
        assert o["alpha_homo_ev"] is not None and o["beta_lumo_ev"] is not None
        assert o["homo_ev"] == pytest.approx(max(o["alpha_homo_ev"], o["beta_homo_ev"]))
        assert o["lumo_ev"] == pytest.approx(min(o["alpha_lumo_ev"], o["beta_lumo_ev"]))

    def test_triplet_oxygen_has_two_unpaired_electrons_shared_by_the_atoms(
        self, tmp_path
    ):
        out = run_ok(
            {"smiles": "O=O", "multiplicity": 3, "properties": ["charges"]}, tmp_path
        ).outputs("props")
        assert out["spin_squared"] == pytest.approx(2.0, abs=0.05)
        assert [
            a["spin_population"] for a in out["charges"]["mulliken"]
        ] == pytest.approx([1.0, 1.0], abs=0.02)

    def test_the_methyl_radical_is_planar_and_has_no_dipole(self, tmp_path):
        out = run_ok(
            {"smiles": "[CH3]", "multiplicity": 2, "properties": ["charges"]}, tmp_path
        ).outputs("props")
        assert out["charges"]["dipole_debye"] == pytest.approx(0.0, abs=0.05)

    def test_the_closed_shell_results_carry_no_spin_quantities(self, tmp_path):
        out = run_ok(
            {"smiles": "O", "properties": ["orbitals", "charges"]}, tmp_path
        ).outputs("props")
        assert out["spin_squared"] is None and out["orbitals"]["alpha_homo_ev"] is None
        assert all(a["spin_population"] is None for a in out["charges"]["mulliken"])


@engine_only
@pytest.mark.usefixtures("engine")
class TestChargesAndTheDipole:
    def test_the_dipole_of_water_is_the_b3lyp_value_and_lies_along_the_symmetry_axis(
        self, tmp_path
    ):
        c = run_ok({"smiles": "O", "properties": ["charges"]}, tmp_path).outputs(
            "props"
        )["charges"]
        assert c["dipole_debye"] == pytest.approx(2.1, abs=0.15)
        assert c["dipole_origin"] == "center of mass"
        assert sum(x * x for x in c["dipole_components_debye"]) ** 0.5 == pytest.approx(
            c["dipole_debye"]
        )

    def test_the_charges_follow_the_electronegativity_and_sum_to_the_net_charge(
        self, tmp_path
    ):
        c = run_ok({"smiles": "O", "properties": ["charges"]}, tmp_path).outputs(
            "props"
        )["charges"]["mulliken"]
        assert [a["element"] for a in c] == ["O", "H", "H"]
        assert c[0]["charge"] < 0 < c[1]["charge"]
        assert c[1]["charge"] == pytest.approx(c[2]["charge"], abs=1e-3)
        assert sum(a["charge"] for a in c) == pytest.approx(0.0, abs=1e-6)

    def test_the_charges_of_an_ion_sum_to_its_charge(self, tmp_path):
        c = run_ok({"smiles": "[NH4+]", "properties": ["charges"]}, tmp_path).outputs(
            "props"
        )["charges"]["mulliken"]
        assert sum(a["charge"] for a in c) == pytest.approx(1.0, abs=1e-6)


@engine_only
@pytest.mark.usefixtures("engine")
class TestWaterAsAContinuum:
    @pytest.fixture(scope="class")
    def solvated(self, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_MOLECULAR_PROPERTIES"] = ENV
        tmp = tmp_path_factory.mktemp("solvated")
        gas = run_ok(
            {"smiles": "O", "properties": ["orbitals", "charges", "excitations"]},
            tmp,
            "gas",
        ).outputs("props")
        water = run_ok(
            {
                "smiles": "O",
                "solvent": "water",
                "properties": ["orbitals", "charges", "excitations", "thermochemistry"],
            },
            tmp,
            "water",
        ).outputs("props")
        return gas, water

    def test_the_polarization_energy_of_water_is_close_to_its_experimental_solvation_energy(
        self, solvated
    ):
        _, water = solvated
        assert water["solvent_polarization_energy_kcal_mol"] == pytest.approx(
            -6.32, abs=1.0
        )

    def test_the_polarization_energy_is_the_energy_difference_at_one_geometry(
        self, solvated
    ):
        _, water = solvated
        difference = water["energy_hartree"] - water["gas_phase_energy_hartree"]
        assert water["solvent_polarization_energy_kcal_mol"] == pytest.approx(
            difference * 627.5094740631
        )

    def test_the_continuum_raises_the_dipole_and_the_gap(self, solvated):
        gas, water = solvated
        assert water["charges"]["dipole_debye"] > gas["charges"]["dipole_debye"] + 0.1
        assert water["orbitals"]["gap_ev"] > gas["orbitals"]["gap_ev"]

    def test_the_excitations_are_computed_in_the_continuum(self, solvated):
        gas, water = solvated
        assert water["excitations"][0]["energy_ev"] != pytest.approx(
            gas["excitations"][0]["energy_ev"], abs=0.1
        )

    def test_the_thermochemistry_runs_in_the_continuum_without_imaginary_modes(
        self, solvated
    ):
        _, water = solvated
        assert water["thermochemistry"]["imaginary_modes"] == 0
        assert water["thermochemistry"]["cv_cal_mol_k"] == pytest.approx(5.97, abs=0.3)

    def test_a_gas_phase_result_has_no_polarization_energy(self, solvated):
        gas, _ = solvated
        assert gas["solvent"] == "none"
        assert gas["solvent_polarization_energy_kcal_mol"] is None
        assert gas["gas_phase_energy_hartree"] is None


class TestTheResultFromTheEngineIsRead:
    """With a stand-in engine, so the notes and the fields are checked without PySCF."""

    @pytest.fixture
    def stand_in(self, monkeypatch, engine):
        from alambiq.core import envs

        def make(**changes):
            found = {
                "canonical_smiles": "[OH]",
                "formula": "HO",
                "charge": 0,
                "multiplicity": 2,
                "solvent": "none",
                "heavy_atoms": 1,
                "atoms": 2,
                "conformers_embedded": 10,
                "force_field_energy_kcal_mol": 0.0,
                "optimization_steps": 3,
                "energy_hartree": -75.7,
                "geometry_xyz": "2\nOH\nO 0 0 0\nH 0 0 0.97\n",
                "seconds": {"optimization": 1.0, "total": 1.0},
                "versions": {"pyscf": "2.14.0"},
                "spin_squared": 0.75,
                "expected_spin_squared": 0.75,
                **changes,
            }
            monkeypatch.setattr(envs, "run_in_environment", lambda *a, **k: found)

        return make

    def test_a_pure_spin_state_carries_no_note(self, stand_in, tmp_path):
        stand_in()
        outcome = run_ok({"smiles": "[OH]", "multiplicity": 2}, tmp_path)
        assert outcome.steps["props"].notes == []

    def test_spin_contamination_of_more_than_ten_percent_carries_a_note(
        self, stand_in, tmp_path
    ):
        stand_in(spin_squared=0.9)
        outcome = run_ok({"smiles": "[OH]", "multiplicity": 2}, tmp_path)
        (note,) = outcome.steps["props"].notes
        assert "0.900 against 0.750" in note and "spin contaminated" in note

    def test_imaginary_frequencies_carry_a_note_naming_the_count(
        self, stand_in, tmp_path
    ):
        stand_in(
            thermochemistry={
                "temperature_k": 298.15,
                "pressure_pa": 101325.0,
                "zero_point_energy_hartree": 0.01,
                "zero_point_energy_kcal_mol": 6.3,
                "energy_0k_hartree": -75.69,
                "internal_energy_hartree": -75.68,
                "enthalpy_hartree": -75.68,
                "gibbs_energy_hartree": -75.7,
                "entropy_cal_mol_k": 43.0,
                "cv_cal_mol_k": 5.0,
                "cp_cal_mol_k": 7.0,
                "symmetry_number": 1,
                "vibrational_modes": 2,
                "imaginary_modes": 2,
                "lowest_frequencies_cm1": [100.0, 200.0],
            }
        )
        outcome = run_ok(
            {"smiles": "[OH]", "multiplicity": 2, "properties": ["thermochemistry"]},
            tmp_path,
        )
        (note,) = outcome.steps["props"].notes
        assert "2 imaginary frequencies" in note and "saddle point" in note
