"""structure.prepare_finite_environment on a prepared protein-ligand complex, then QM/MM on its ligand.

The periodic system is the Trp-cage and alaninate complex that
`structure.prepare_predicted_complex_for_dynamics` builds. Every property of the droplet that
can be written down from the files is checked independently of the capability:

* every kept solvent molecule is whole and has an atom within the sphere's radius of its centre;
* the atom counts, the net charge (summed from the system's own charges) and the
  restrained residues follow from the topology and the coordinates;
* the boundary system holds the two named potentials, with one wall per kept molecule;
* the cluster runs through `simulation.evaluate_qmmm_state` with its boundary, so the three
  capabilities (preparation, finite cut, quantum evaluation) fit together.

The cut runs in the OpenFF environment (ALAMBIQ_ENV_OPENFF); the evaluation also needs
ALAMBIQ_ENV_QMMM.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from tests.core.test_structure_prepare_predicted_complex_for_dynamics import (  # noqa: E402
    ENV,
    complex_files,  # noqa: F401
)
from tests.core.test_structure_prepare_predicted_complex_for_dynamics import (  # noqa: E402
    run as prepare,
)

ENV_QMMM = os.environ.get("ALAMBIQ_ENV_QMMM", "")
SOLVENT = {"HOH", "WAT"}
IONS = {"NA", "CL", "Na+", "Cl-"}

pytestmark = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_OPENFF to run the finite environment",
)


def atoms_of(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            rows.append(
                {
                    "name": line[12:16].strip(),
                    "residue": line[17:20].strip(),
                    "chain": line[21],
                    "number": int(line[22:26]),
                    "xyz": np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    ),
                    "element": line[76:78].strip(),
                }
            )
    return rows


def charges_of(system_xml: str) -> list[float]:
    return [
        float(q)
        for q in re.findall(r'<Particle[^>]* q="([^"]+)"', Path(system_xml).read_text())
    ]


@pytest.fixture(scope="module")
def chain(complex_files, tmp_path_factory):  # noqa: F811
    from tests.core.test_structure_prepare_predicted_complex_for_dynamics import request

    os.environ["ALAMBIQ_ENV_OPENFF"] = ENV
    directory = tmp_path_factory.mktemp("finite")
    prepared = prepare(
        request(complex_files, padding_nm=1.5, box_shape="cube"), directory, "prep"
    )
    assert prepared.ok, [s.error for s in prepared.steps.values() if s.error]
    out = prepared.outputs("prep")
    atoms = atoms_of(out["topology_pdb"])
    ligand = [i + 1 for i, a in enumerate(atoms) if a["residue"] == "LIG"]
    protein_atoms = ligand[0] - 1
    from alambiq.api import run

    step = {
        "id": "cut",
        "capability": "structure.prepare_finite_environment",
        "params": {
            "amber_topology": out["amber_topology"],
            "state": out["state_xml"],
            "solute_residues": list(range(1, 22)),
            "anchor_residue": 21,
            "protected_atoms": ligand,
            "solvent_radius_angstrom": 18.0,
            "inner_solute_radius_angstrom": 8.0,
            "position_coefficient_kj_mol_nm2": 2092.0,
            "wall_coefficient_kj_mol_nm2": 4184.0,
        },
    }
    cut = run({"id": "cut", "steps": [step]}, run_dir=directory / "cut")
    assert cut.ok, [s.error for s in cut.steps.values() if s.error]
    return {
        "directory": directory,
        "prepared": out,
        "cut": cut.outputs("cut"),
        "outcome": cut,
        "ligand": ligand,
        "protein_atoms": protein_atoms,
    }


class TestTheDroplet:
    def test_the_solute_is_kept_whole_and_the_solvent_is_whole_molecules(self, chain):
        atoms = atoms_of(chain["cut"]["topology_pdb"])
        by_residue: dict[tuple, list] = {}
        for a in atoms:
            by_residue.setdefault((a["chain"], a["number"], a["residue"]), []).append(a)
        waters = [v for k, v in by_residue.items() if k[2] in SOLVENT]
        ions = [v for k, v in by_residue.items() if k[2] in IONS]
        assert waters and all(len(w) == 3 for w in waters)
        assert all(len(i) == 1 for i in ions)
        assert chain["cut"]["solvent_and_ion_molecules"] == len(waters) + len(ions)
        solute = len(atoms) - 3 * len(waters) - len(ions)
        assert chain["cut"]["solute_atoms"] == solute == chain["protein_atoms"] + 12
        assert chain["cut"]["atoms"] == len(atoms) < chain["cut"]["source_atoms"]
        assert chain["cut"]["solute_residues"] == 21

    def test_every_kept_molecule_reaches_into_the_sphere(self, chain):
        atoms = atoms_of(chain["cut"]["topology_pdb"])
        mapping = json.loads(Path(chain["cut"]["source_atom_map"]).read_text())
        centre = atoms[mapping["center_atom_one_based"] - 1]["xyz"]
        assert centre == pytest.approx([64.0, 64.0, 64.0], abs=1e-3)
        molecules: dict[tuple, list] = {}
        for a in atoms:
            if a["residue"] in SOLVENT | IONS:
                molecules.setdefault((a["chain"], a["number"]), []).append(a["xyz"])
        assert molecules
        for key, points in molecules.items():
            nearest = min(np.linalg.norm(np.array(points) - centre, axis=1))
            assert nearest <= 18.0 + 1e-3, key

    def test_the_net_charge_is_the_sum_of_the_systems_charges(self, chain):
        charges = charges_of(chain["cut"]["system_xml"])
        assert len(charges) == chain["cut"]["atoms"]
        assert chain["cut"]["net_charge_e"] == pytest.approx(sum(charges), abs=1e-6)

    def test_the_restrained_atoms_are_the_heavy_atoms_of_the_outer_solute_residues(
        self, chain
    ):
        atoms = atoms_of(chain["cut"]["topology_pdb"])
        mapping = json.loads(Path(chain["cut"]["source_atom_map"]).read_text())
        centre = atoms[mapping["center_atom_one_based"] - 1]["xyz"]
        expected = 0
        residues: dict[tuple, list] = {}
        for a in atoms:
            if a["residue"] not in SOLVENT | IONS:
                residues.setdefault((a["chain"], a["number"]), []).append(a)
        for members in residues.values():
            distances = [np.linalg.norm(a["xyz"] - centre) for a in members]
            if min(distances) > 8.0:
                expected += sum(a["element"] != "H" for a in members)
        assert chain["cut"]["restrained_heavy_atoms"] == expected > 0

    def test_the_protected_atoms_are_the_ligand_and_stay_free(self, chain):
        atoms = atoms_of(chain["cut"]["topology_pdb"])
        mapping = json.loads(Path(chain["cut"]["source_atom_map"]).read_text())
        protected = mapping["protected_atoms_one_based"]
        assert [atoms[i - 1]["residue"] for i in protected] == ["LIG"] * 12
        assert len(mapping["source_atom_indices_one_based"]) == chain["cut"]["atoms"]
        assert mapping["source_atom_indices_one_based"] == sorted(
            mapping["source_atom_indices_one_based"]
        )

    def test_the_boundary_system_has_a_restraint_and_one_wall_per_kept_molecule(
        self, chain
    ):
        text = Path(chain["cut"]["external_system_xml"]).read_text()
        assert "Outer-solute positional restraint" in text
        assert "Complete solvent/ion mass-center wall" in text
        assert (
            text.count("<Group ") == chain["cut"]["solvent_and_ion_molecules"]
            or text.count("<Group") >= chain["cut"]["solvent_and_ion_molecules"]
        )

    def test_forces_on_the_source_positions_are_finite_and_the_manifest_records_the_cut(
        self, chain
    ):
        assert (
            0
            < chain["cut"]["rms_force_kj_mol_nm"]
            < chain["cut"]["max_force_kj_mol_nm"]
        )
        manifest = json.loads(Path(chain["cut"]["manifest"]).read_text())
        assert manifest["parameters"]["solvent_radius_angstrom"] == 18.0
        assert (
            manifest["physical_hamiltonian_sha256"]
            and len(manifest["physical_hamiltonian_sha256"]) == 64
        )
        assert chain["cut"]["inverted_alpha_residues"] == []

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"solute_residues": []}, "nonempty ascending list"),
            ({"solute_residues": [2, 1]}, "nonempty ascending list"),
            ({"anchor_residue": 99}, "which is not in solute_residues"),
            ({"protected_atoms": [0]}, "nonempty ascending list"),
            (
                {"solvent_radius_angstrom": 5.0},
                "solvent_radius_angstrom is 5.0; a finite value from 10 to 50 is required",
            ),
            (
                {"inner_solute_radius_angstrom": 30.0},
                "is not smaller than solvent_radius_angstrom",
            ),
            ({"threads": 0}, "threads is 0; an integer from 1 to 64 is required"),
        ],
    )
    def test_a_declaration_that_cannot_be_cut_is_refused(
        self, chain, tmp_path, changes, message
    ):
        from alambiq.api import run

        out = chain["prepared"]
        params = {
            "amber_topology": out["amber_topology"],
            "state": out["state_xml"],
            "solute_residues": list(range(1, 22)),
            "anchor_residue": 21,
            "protected_atoms": chain["ligand"],
            **changes,
        }
        step = {
            "id": "cut",
            "capability": "structure.prepare_finite_environment",
            "params": params,
        }
        outcome = run({"id": "cut", "steps": [step]}, run_dir=tmp_path / "bad")
        assert not outcome.ok
        assert message in outcome.steps["cut"].error

    def test_a_solute_that_leaves_out_a_molecular_residue_is_refused(
        self, chain, tmp_path
    ):
        from alambiq.api import run

        out = chain["prepared"]
        step = {
            "id": "cut",
            "capability": "structure.prepare_finite_environment",
            "params": {
                "amber_topology": out["amber_topology"],
                "state": out["state_xml"],
                "solute_residues": list(range(1, 21)),
                "anchor_residue": 1,
                "protected_atoms": [1],
            },
        }
        outcome = run({"id": "cut", "steps": [step]}, run_dir=tmp_path / "bad2")
        assert not outcome.ok
        assert (
            "remaining residues must be complete water or monatomic ions"
            in outcome.steps["cut"].error
        )


@pytest.mark.skipif(
    not (ENV_QMMM and Path(ENV_QMMM).is_file()),
    reason="set ALAMBIQ_ENV_QMMM to evaluate the ligand quantum mechanically",
)
class TestTheDropletInQuantumMechanics:
    def test_the_ligand_in_its_droplet_has_a_finite_converged_energy_and_the_boundary_adds_none(
        self, chain, tmp_path
    ):
        from alambiq.api import run

        os.environ["ALAMBIQ_ENV_QMMM"] = ENV_QMMM
        cut = chain["cut"]
        step = {
            "id": "qm",
            "capability": "simulation.evaluate_qmmm_state",
            "params": {
                "system": cut["system_xml"],
                "state": cut["state_xml"],
                "topology": cut["topology_pdb"],
                "external_system": cut["external_system_xml"],
                "quantum": {"atoms": chain["ligand"], "charge": -1},
                "grid": 3,
                "threads": 8,
            },
        }
        outcome = run({"id": "qm", "steps": [step]}, run_dir=tmp_path / "qm")
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        out = outcome.outputs("qm")
        assert out["converged"] and out["electrons_expected"] == 48
        assert np.isfinite(out["total_energy_kj_mol"])
        assert out["physical_atoms"] == cut["atoms"]
        # The springs sit at their reference positions, so only the wall costs energy:
        # k_wall times the squared excess of each kept molecule's mass centre beyond the
        # radius, which this recomputes from the written files.
        masses = [
            float(m)
            for m in re.findall(
                r'<Particle mass="([^"]+)"', Path(cut["system_xml"]).read_text()
            )
        ]
        atoms = atoms_of(cut["topology_pdb"])
        molecules: dict[tuple, list[int]] = {}
        for index, a in enumerate(atoms):
            if a["residue"] in SOLVENT | IONS:
                molecules.setdefault((a["chain"], a["number"]), []).append(index)
        wall = 0.0
        for members in molecules.values():
            weights = np.array([masses[i] for i in members])
            centre = (
                np.array([atoms[i]["xyz"] for i in members]) * weights[:, None]
            ).sum(0) / weights.sum()
            excess = max(0.0, np.linalg.norm(centre / 10 - np.array([6.4] * 3)) - 1.8)
            wall += 4184.0 * excess**2
        assert out["external_energy_kj_mol"] == pytest.approx(wall, rel=5e-3, abs=1e-3)
