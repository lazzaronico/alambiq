"""structure.prepare_protein_ligand_system on the modelled Trp-cage with ethanol.

The protein and the ligand come from `structure.prepare_reactive_coordinates` run on the
synthetic Trp-cage case, so the answers are known without a model in the loop:

* every heavy atom of the observed residues, and of the ligand, is exactly where the
  input put it after relaxation and solvation, which the capability's own check also
  insists on;
* the two modelled residues have moved by the amount the capability reports and no more,
  recomputed here from the input and the exported state;
* the composition (waters, ions, net charge, atom count, the ligand's place in the
  topology) follows from the written files, and both relaxation stages lowered the energy;
* the vacuum variant has no box, no solvent and the protein's own charge;
* each way the request can disagree with the files is refused by name;
* the cluster cut by `structure.prepare_finite_environment` from the solvated cube is modelled
  from the ligand's place in the topology, so the three steps fit together.

The preparation runs in the pinned OpenFF environment (ALAMBIQ_ENV_OPENFF).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gemmi")
Chem = pytest.importorskip("rdkit.Chem")

from tests.core.test_structure_prepare_reactive_coordinates import (  # noqa: E402
    files,  # noqa: F401
    ok,
    pdb_atoms,
    recipe,
)
from tests.core.test_structure_prepare_reactive_coordinates import (  # noqa: E402
    run as prepare_coordinates,
)

ENV = os.environ.get("ALAMBIQ_ENV_OPENFF", "")

pytestmark = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_OPENFF to prepare the system",
)

SOLVENT = {"HOH", "WAT"}
IONS = {"NA", "CL", "Na+", "Cl-"}


@pytest.fixture(scope="module")
def modelled(files, tmp_path_factory):  # noqa: F811
    outcome = prepare_coordinates(recipe(files), tmp_path_factory.mktemp("modelled"))
    return outcome, ok(outcome)


def system_params(modelled, **changes):
    params = {
        "protein": modelled[1]["protein_pdb"],
        "ligands": [
            {
                "chain": "S",
                "residue_name": "ETH",
                "pose": modelled[1]["substrates"][0]["sdf"],
                "formal_charge": 0,
            }
        ],
        "modeled_residues": [{"chain": "A", "number": 1}, {"chain": "A", "number": 2}],
        "padding_nm": 0.9,
        "ionic_strength_molar": 0.15,
        "minimization_max_iterations": 2000,
        "threads": 2,
    }
    params.update(changes)
    return params


def run_system(params, tmp_path, name="run"):
    from alambiq.api import run as execute

    step = {
        "id": "sys",
        "capability": "structure.prepare_protein_ligand_system",
        "params": params,
    }
    return execute({"id": "sys", "steps": [step]}, run_dir=tmp_path / name)


def state_positions(path):
    text = Path(path).read_text()
    rows = re.findall(r'<Position x="([^"]+)" y="([^"]+)" z="([^"]+)"', text)
    return np.array(rows, dtype=float)


def topology_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            rows.append(
                {
                    "name": line[12:16].strip(),
                    "residue": line[17:20].strip(),
                    "number": int(line[22:26]),
                    "element": line[76:78].strip(),
                }
            )
    return rows


@pytest.fixture(scope="module")
def solvated(modelled, tmp_path_factory):
    outcome = run_system(system_params(modelled), tmp_path_factory.mktemp("solvated"))
    return outcome, ok_system(outcome)


def ok_system(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("sys")


@pytest.fixture(scope="module")
def vacuum(modelled, tmp_path_factory):
    outcome = run_system(
        system_params(modelled, solvent="none"), tmp_path_factory.mktemp("vacuum")
    )
    return outcome, ok_system(outcome)


class TestHeldAtoms:
    def test_every_observed_protein_heavy_atom_is_exactly_where_it_was(
        self,
        solvated,
        modelled,
    ):
        placed = pdb_atoms(modelled[1]["protein_pdb"])
        rows = topology_rows(solvated[1]["topology_pdb"])
        positions = state_positions(solvated[1]["state_xml"])
        checked = 0
        for index, row in enumerate(rows):
            if row["element"] == "H" or row["residue"] in SOLVENT | IONS:
                continue
            if 3 <= row["number"] <= 20:
                expected = placed[(row["number"], row["name"])] / 10.0
                assert positions[index] == pytest.approx(expected, abs=1e-6)
                checked += 1
        assert checked == 154 - sum(1 for key in placed if key[0] in (1, 2))

    def test_the_ligand_heavy_atoms_are_where_the_pose_put_them(
        self,
        solvated,
        modelled,
    ):
        molecule = next(
            iter(
                Chem.SDMolSupplier(modelled[1]["substrates"][0]["sdf"], removeHs=False)
            )
        )
        site = solvated[1]["ligand_sites"][0]
        positions = state_positions(solvated[1]["state_xml"])
        for atom in molecule.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            expected = np.array(molecule.GetConformer().GetAtomPosition(atom.GetIdx()))
            got = positions[site["first_atom"] - 1 + atom.GetIdx()]
            assert got == pytest.approx(expected / 10.0, abs=1e-5)

    def test_the_modelled_residues_moved_by_the_reported_amount(
        self,
        solvated,
        modelled,
    ):
        placed = pdb_atoms(modelled[1]["protein_pdb"])
        rows = topology_rows(solvated[1]["topology_pdb"])
        positions = state_positions(solvated[1]["state_xml"])
        moves = [
            np.linalg.norm(positions[i] * 10.0 - placed[(r["number"], r["name"])])
            for i, r in enumerate(rows)
            if r["element"] != "H"
            and r["residue"] not in SOLVENT | IONS
            and r["number"] in (1, 2)
        ]
        assert len(moves) == solvated[1]["modeled_heavy_atoms_relaxed"]
        assert max(moves) == pytest.approx(
            solvated[1]["maximum_modeled_displacement_angstrom"], abs=2e-3
        )

    def test_the_count_of_held_atoms_is_the_observed_protein_plus_the_ligand(
        self, solvated
    ):
        out = solvated[1]
        assert (
            out["observed_heavy_atoms_held"]
            == 154 - out["modeled_heavy_atoms_relaxed"] + 3
        )


class TestComposition:
    def test_the_counts_follow_from_the_written_topology(self, solvated):
        out = solvated[1]
        rows = topology_rows(out["topology_pdb"])
        waters = sum(1 for r in rows if r["residue"] in SOLVENT and r["name"] == "O")
        sodium = sum(1 for r in rows if r["residue"] in {"NA", "Na+"})
        chloride = sum(1 for r in rows if r["residue"] in {"CL", "Cl-"})
        assert len(rows) == out["atom_count"]
        assert (out["water_molecules"], out["sodium_ions"], out["chloride_ions"]) == (
            waters,
            sodium,
            chloride,
        )
        assert out["periodic"] is True and out["box_volume_nm3"] > 0
        assert abs(out["net_charge_e"]) < 0.01

    def test_the_ligand_follows_the_protein_in_the_topology(self, solvated):
        out = solvated[1]
        (site,) = out["ligand_sites"]
        rows = topology_rows(out["topology_pdb"])
        assert site["residue_name"] == "ETH" and site["residue_number"] == 21
        assert site["atom_count"] == 9 and site["first_atom"] == out["solute_atoms"] - 8
        block = rows[site["first_atom"] - 1 : site["first_atom"] - 1 + 9]
        assert {r["residue"] for r in block} == {"ETH"}
        assert abs(site["charge_sum_e"]) < 1e-6 and site["formal_charge"] == 0

    def test_both_relaxation_stages_lowered_the_energy(self, solvated):
        out = solvated[1]
        assert out["vacuum_energy_after_kj_mol"] < out["vacuum_energy_before_kj_mol"]
        assert out["solvent_energy_after_kj_mol"] < out["solvent_energy_before_kj_mol"]
        assert np.isfinite(out["rms_force_after_kj_mol_nm"])

    def test_a_dodecahedron_cell_is_smaller_than_the_cube_for_the_same_padding(
        self, solvated, modelled, tmp_path
    ):
        outcome = run_system(
            system_params(
                modelled,
                box_shape="dodecahedron",
                minimizer_tolerance_kj_mol_nm=5.0,
                seed=7,
            ),
            tmp_path,
        )
        out = ok_system(outcome)
        assert 0 < out["box_volume_nm3"] < solvated[1]["box_volume_nm3"]
        assert abs(out["net_charge_e"]) < 0.01
        assert out["solute_atoms"] == solvated[1]["solute_atoms"]


class TestVacuum:
    def test_there_is_no_box_solvent_or_ions_and_the_charge_is_the_proteins(
        self, vacuum
    ):
        out = vacuum[1]
        assert out["periodic"] is False and out["box_volume_nm3"] is None
        assert (out["water_molecules"], out["sodium_ions"], out["chloride_ions"]) == (
            0,
            0,
            0,
        )
        # Trp-cage: Lys8, Arg16 and the N terminus against Asp9 and the C terminus.
        assert out["net_charge_e"] == pytest.approx(1.0, abs=1e-3)
        assert out["atom_count"] == out["solute_atoms"]

    def test_stiffer_springs_hold_the_modelled_residues_closer_to_where_they_were_placed(
        self, vacuum, modelled, tmp_path
    ):
        moved = {}
        for k in (0.0, 100.0, 10000.0):
            outcome = run_system(
                system_params(
                    modelled, solvent="none", modeled_restraint_k_kj_mol_nm2=k
                ),
                tmp_path,
                name=f"k{k:g}",
            )
            moved[k] = ok_system(outcome)["maximum_modeled_displacement_angstrom"]
        assert moved[0.0] > moved[100.0] > moved[10000.0] >= 0.0
        # Hydrogens are added by PDBFixer, which is not bit-reproducible, so the same
        # request gives displacements that agree closely but not exactly.
        assert moved[100.0] == pytest.approx(
            vacuum[1]["maximum_modeled_displacement_angstrom"], abs=0.1
        )


class TestRefusals:
    def fails(self, modelled, tmp_path, message, **changes):
        outcome = run_system(system_params(modelled, **changes), tmp_path)
        assert not outcome.ok
        assert message in " ".join(s.error or "" for s in outcome.steps.values())

    def test_a_ligand_chain_the_protein_uses_is_refused(self, modelled, tmp_path):
        ligands = system_params(modelled)["ligands"]
        ligands[0]["chain"] = "A"
        self.fails(modelled, tmp_path, "already used", ligands=ligands)

    def test_a_wrong_declared_charge_is_refused(self, modelled, tmp_path):
        ligands = system_params(modelled)["ligands"]
        ligands[0]["formal_charge"] = -1
        self.fails(modelled, tmp_path, "formal charge 0, but -1", ligands=ligands)

    def test_a_modelled_residue_that_is_not_in_the_protein_is_refused(
        self,
        modelled,
        tmp_path,
    ):
        self.fails(
            modelled,
            tmp_path,
            "not in the protein",
            modeled_residues=[{"chain": "A", "number": 99}],
        )

    def test_a_modelled_chain_that_is_not_in_the_protein_is_refused(
        self,
        modelled,
        tmp_path,
    ):
        self.fails(
            modelled,
            tmp_path,
            "which is not in the protein file",
            modeled_residues=[{"chain": "Z", "number": 1}],
        )

    def test_a_residue_name_of_the_wrong_length_is_refused(self, modelled, tmp_path):
        ligands = system_params(modelled)["ligands"]
        ligands[0]["residue_name"] = "ETHANOL"
        self.fails(
            modelled,
            tmp_path,
            "exactly 3 letters or digits are required",
            ligands=ligands,
        )

    def test_settings_outside_their_range_are_refused(self, modelled, tmp_path):
        self.fails(modelled, tmp_path, "padding_nm is 9.0", padding_nm=9.0)
        self.fails(modelled, tmp_path, "1 to 8 are required", ligands=[])


class TestChain:
    def test_the_solvated_cube_is_cut_into_a_droplet_around_the_ligand(
        self, solvated, tmp_path
    ):
        from alambiq.api import run as execute

        out = solvated[1]
        (site,) = out["ligand_sites"]
        step = {
            "id": "cut",
            "capability": "structure.prepare_finite_environment",
            "params": {
                "amber_topology": out["amber_topology"],
                "state": out["state_xml"],
                "solute_residues": list(range(1, 22)),
                "anchor_residue": site["residue_number"],
                "protected_atoms": [site["first_atom"]],
                "solvent_radius_angstrom": 12.0,
                "inner_solute_radius_angstrom": 10.0,
            },
        }
        outcome = execute({"id": "cut", "steps": [step]}, run_dir=tmp_path / "cut")
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        cut = outcome.outputs("cut")
        assert cut["solute_atoms"] == out["solute_atoms"]
        assert cut["atoms"] <= out["atom_count"]
