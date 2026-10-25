"""structure.prepare_predicted_complex_for_dynamics on a complex built from known parts.

The "prediction" here is Trp-cage's deposited heavy atoms with L-alaninate beside it,
written in the layout Boltz uses, and the ligand's chemistry and pose come from the
same functions the prediction capability uses on a real Boltz output. That makes
the answers known without a model in the loop:

* the protein is read, checked and rewritten atom for atom, with the names the force
  field expects (histidine as HIE, disulfide cysteines as CYX), and each way the
  prediction can be unusable (an unsupported residue, a missing atom, a wrong
  sequence, an ambiguous disulfide) is refused by name;
* the ligand must be the one intended: a pose of the opposite handedness, a
  pose from another structure, and a charge other than the declared one are all
  refused;
* the prepared system has the composition its own topology shows, a net charge of zero,
  AshGC ligand charges that sum to the ligand's formal charge, a lower energy than it
  started with, and runs in `simulation.run_molecular_dynamics`, whose OpenMM is a
  different version from the one that built it.

The preparation runs in the pinned OpenFF environment (ALAMBIQ_ENV_OPENFF); the
dynamics check also needs the OpenMM environment (ALAMBIQ_ENV_OPENMM).
"""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
gemmi = pytest.importorskip("gemmi")
rdkit = pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

import alambiq.capabilities.investigate.science.predicted_complex as pc  # noqa: E402
from alambiq.engines import boltz_predict  # noqa: E402

STRUCTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/1L2Y.cif"
ENV = os.environ.get("ALAMBIQ_ENV_OPENFF", "")
ENV_MD = os.environ.get("ALAMBIQ_ENV_OPENMM", "")
ALANINATE = "C[C@H](N)C(=O)[O-]"
SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"
CIF_COLUMNS = (
    "group_PDB id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id "
    "label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy "
    "B_iso_or_equiv auth_seq_id auth_comp_id auth_asym_id auth_atom_id pdbx_PDB_model_num"
).split()


def cif_text(residues, ligand=None, second_ligand=None) -> str:
    """Write protein residues (chain, number, name, [(atom, element, xyz)]) as mmCIF."""
    lines = ["data_prediction", "loop_", *[f"_atom_site.{c}" for c in CIF_COLUMNS]]
    serial = 0
    for chain, number, name, atoms in residues:
        for atom, element, (x, y, z) in atoms:
            serial += 1
            lines.append(
                f"ATOM {serial} {element} {atom} . {name} {chain} 1 {number} ? "
                f"{x:.5f} {y:.5f} {z:.5f} 1 50.0 {number} {name} {chain} {atom} 1"
            )
    for chain, entity, atoms in (("L", 2, ligand), ("M", 3, second_ligand)):
        for atom, element, (x, y, z) in atoms or []:
            serial += 1
            lines.append(
                f"HETATM {serial} {element} {atom} . LIG1 {chain} {entity} 1 ? "
                f"{x:.5f} {y:.5f} {z:.5f} 1 50.0 1 LIG1 {chain} {atom} 1"
            )
    return "\n".join(lines) + "\n"


def trpcage_residues():
    structure = gemmi.read_structure(str(STRUCTURE))
    out = []
    for residue in structure[0]["A"]:
        atoms = [
            (a.name, a.element.name.upper(), (a.pos.x, a.pos.y, a.pos.z))
            for a in residue
            # Boltz writes no terminal OXT; the deposited structure has one.
            if a.element.name not in {"H", "D"} and a.name != "OXT"
        ]
        out.append(("A", residue.seqid.num, residue.name, atoms))
    return out


def ligand_template():
    molecule = Chem.MolFromSmiles(ALANINATE)
    for atom in molecule.GetAtoms():
        atom.SetProp("name", f"{atom.GetSymbol()}{atom.GetIdx() + 1}")
    assert AllChem.EmbedMolecule(molecule, randomSeed=11) == 0
    return molecule


@pytest.fixture(scope="module")
def complex_files(tmp_path_factory):
    """The predicted mmCIF, the ligand pose and chemistry, and a mirrored pose."""
    directory = tmp_path_factory.mktemp("predicted")
    protein = trpcage_residues()
    edge = max(x for _, _, _, atoms in protein for _, _, (x, _, _) in atoms)
    centre = np.mean([xyz for _, _, _, atoms in protein for _, _, xyz in atoms], axis=0)
    molecule = ligand_template()
    coordinates = molecule.GetConformer().GetPositions()
    coordinates = (
        coordinates - coordinates.mean(axis=0) + [edge + 5.0, centre[1], centre[2]]
    )
    ligand = [
        (a.GetProp("name"), a.GetSymbol().upper(), tuple(coordinates[a.GetIdx()]))
        for a in molecule.GetAtoms()
    ]
    cif = directory / "prediction.cif"
    cif.write_text(cif_text(protein, ligand))
    template = Chem.Mol(molecule)
    template.RemoveAllConformers()
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    native = directory / "input.pkl"
    native.write_bytes(pickle.dumps({"LIG1": template}))
    chemistry = boltz_predict._chemistry(native, {"id": "L", "smiles": ALANINATE})
    pose = directory / "pose.sdf"
    boltz_predict._positioned(chemistry, cif, pose)
    chemistry_file = directory / "chemistry.json"
    chemistry_file.write_text(json.dumps(chemistry))
    # The mirror image: the same atoms with x reflected, in its own structure file.
    mirrored = [(n, e, (-x, y, z)) for n, e, (x, y, z) in ligand]
    mirror_cif = directory / "mirror.cif"
    mirror_cif.write_text(cif_text(protein, mirrored))
    mirror_pose = directory / "mirror.sdf"
    verdict = boltz_predict._positioned(chemistry, mirror_cif, mirror_pose)
    assert verdict["stereochemistry_matches_input"] is False
    return {
        "directory": directory,
        "cif": cif,
        "pose": pose,
        "chemistry": chemistry_file,
        "mirror_cif": mirror_cif,
        "mirror_pose": mirror_pose,
        "protein": protein,
        "ligand": ligand,
    }


class TestReadingAPrediction:
    def write(self, tmp_path, residues, ligand=None):
        path = tmp_path / "p.cif"
        path.write_text(cif_text(residues, ligand))
        return path

    def test_chains_and_sequences_are_read_from_the_file(self, complex_files):
        chains, ligand_chain, ligand_atoms, sequences = pc.read_prediction(
            complex_files["cif"], None
        )
        assert sequences == [SEQUENCE]
        assert [name for name, _ in chains] == ["A"] and len(chains[0][1]) == 20
        assert ligand_chain == "L" and len(ligand_atoms) == 6

    def test_a_declared_sequence_is_checked(self, complex_files):
        pc.read_prediction(complex_files["cif"], [SEQUENCE.lower()])
        with pytest.raises(ValueError, match="sequence mismatch for chain 0"):
            pc.read_prediction(complex_files["cif"], ["A" * 20])
        with pytest.raises(ValueError, match="chain count does not match"):
            pc.read_prediction(complex_files["cif"], [SEQUENCE, SEQUENCE])

    def test_a_residue_that_is_not_a_standard_amino_acid_is_named_and_refused(
        self, complex_files, tmp_path
    ):
        protein = list(complex_files["protein"])
        protein.append(("A", 21, "ZN", [("ZN", "ZN", (1.0, 2.0, 3.0))]))
        path = self.write(tmp_path, protein, complex_files["ligand"])
        with pytest.raises(ValueError, match="unsupported residues: A:21:ZN"):
            pc.read_prediction(path, None)

    @pytest.mark.parametrize("count", [0, 2])
    def test_exactly_one_ligand_is_required(self, complex_files, tmp_path, count):
        ligand = complex_files["ligand"]
        path = tmp_path / "p.cif"
        path.write_text(
            cif_text(
                complex_files["protein"],
                ligand if count else None,
                ligand if count == 2 else None,
            )
        )
        with pytest.raises(ValueError, match="exactly one LIG1 residue"):
            pc.read_prediction(path, None)

    def test_a_missing_heavy_atom_is_named(self, complex_files, tmp_path):
        protein = [list(r) for r in complex_files["protein"]]
        protein[4][3] = [a for a in protein[4][3] if a[0] != "CD"]
        path = self.write(
            tmp_path, [tuple(r) for r in protein], complex_files["ligand"]
        )
        with pytest.raises(
            ValueError, match=r"heavy-atom mismatch at A:5; missing=\['CD'\]"
        ):
            pc.read_prediction(path, None)

    def test_an_extra_atom_such_as_the_terminal_oxygen_is_refused(
        self, complex_files, tmp_path
    ):
        protein = [list(r) for r in complex_files["protein"]]
        protein[-1][3] = [*protein[-1][3], ("OXT", "O", (0.0, 0.0, 0.0))]
        path = self.write(
            tmp_path, [tuple(r) for r in protein], complex_files["ligand"]
        )
        with pytest.raises(ValueError, match=r"extra=\['OXT'\]"):
            pc.read_prediction(path, None)

    def test_residue_numbers_must_run_from_one_without_gaps(
        self, complex_files, tmp_path
    ):
        protein = [
            (c, n + (n > 10), name, atoms)
            for c, n, name, atoms in complex_files["protein"]
        ]
        path = self.write(tmp_path, protein, complex_files["ligand"])
        with pytest.raises(ValueError, match="consecutive starting at 1"):
            pc.read_prediction(path, None)

    def test_a_ligand_atom_name_that_repeats_is_refused(self, complex_files, tmp_path):
        ligand = list(complex_files["ligand"])
        ligand[1] = (ligand[0][0], *ligand[1][1:])
        path = self.write(tmp_path, complex_files["protein"], ligand)
        with pytest.raises(ValueError, match="ligand atom names must be unique"):
            pc.read_prediction(path, None)


class TestDisulfidesAndTheProteinFile:
    def cysteines(self, distance, extra=()):
        """Cys-Gly-Cys with the sulfurs a chosen distance apart."""
        residues = []
        for number, (name, sulfur) in enumerate(
            [("CYS", (0.0, 0.0, 0.0)), ("GLY", None), ("CYS", (distance, 0.0, 0.0))], 1
        ):
            atoms = [
                (a, a[0], (10.0 * number + i, float(number), 0.0))
                for i, a in enumerate(sorted(pc.EXPECTED_HEAVY_ATOMS[name]))
                if a != "SG"
            ]
            if sulfur is not None:
                atoms.append(("SG", "S", sulfur))
            residues.append(("A", number, name, atoms))
        return [*residues, *extra]

    def chains(self, tmp_path, residues):
        path = tmp_path / "c.cif"
        path.write_text(cif_text(residues, [("C1", "C", (50.0, 50.0, 50.0))]))
        return pc.read_prediction(path, None)[0]

    def test_sulfurs_within_two_and_a_half_angstrom_form_a_disulfide(self, tmp_path):
        bonds = pc.find_disulfides(self.chains(tmp_path, self.cysteines(2.05)))
        assert len(bonds) == 1
        left, right, distance = bonds[0]
        assert (left[:2], right[:2]) == (("A", 1), ("A", 3))
        assert distance == pytest.approx(2.05)

    def test_sulfurs_beyond_three_and_a_half_angstrom_are_free_thiols(self, tmp_path):
        assert pc.find_disulfides(self.chains(tmp_path, self.cysteines(3.6))) == []

    @pytest.mark.parametrize("distance", [2.6, 3.0, 3.5])
    def test_a_distance_between_the_two_is_ambiguous_and_refused(
        self, tmp_path, distance
    ):
        with pytest.raises(ValueError, match="Ambiguous cysteine SG geometry"):
            pc.find_disulfides(self.chains(tmp_path, self.cysteines(distance)))

    def test_one_cysteine_cannot_make_two_disulfides(self, tmp_path):
        residues = self.cysteines(2.0)
        third = [
            (a, e, (x, y + 2.0, z)) for a, e, (x, y, z) in residues[2][3] if a == "SG"
        ]
        # A fourth cysteine whose sulfur is 2 A from the first, sharing it with residue 3.
        extra = (
            "A",
            4,
            "CYS",
            [
                (a, a[0], (80.0 + i, 1.0, 0.0))
                for i, a in enumerate(sorted(pc.EXPECTED_HEAVY_ATOMS["CYS"]))
                if a != "SG"
            ]
            + [("SG", "S", (1.0, 1.7, 0.0))],
        )
        assert third  # the third residue's sulfur exists
        with pytest.raises(ValueError, match="multiple disulfides"):
            pc.find_disulfides(self.chains(tmp_path, self.cysteines(2.0, [extra])))

    def test_the_protein_pdb_has_each_heavy_atom_once_with_the_force_field_names(
        self, complex_files, tmp_path
    ):
        chains = pc.read_prediction(complex_files["cif"], None)[0]
        target = tmp_path / "protein.pdb"
        pc.write_protein_pdb(chains, [], target)
        lines = [
            line for line in target.read_text().splitlines() if line.startswith("ATOM")
        ]
        expected = sum(len(a) for _, _, _, a in complex_files["protein"])
        assert len(lines) == expected == 153
        assert target.read_text().splitlines()[-1] == "END"
        assert "TER" in target.read_text().splitlines()
        positions = np.array(
            [
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                for line in lines
            ]
        )
        source = np.array(
            [xyz for _, _, _, atoms in complex_files["protein"] for _, _, xyz in atoms]
        )
        assert positions == pytest.approx(source, abs=6e-4)

    def test_disulfide_cysteines_become_cyx_and_histidine_becomes_hie(self, tmp_path):
        residues = self.cysteines(2.05)
        histidine = (
            "A",
            4,
            "HIS",
            [
                (a, a[0], (90.0 + i, 0.0, 0.0))
                for i, a in enumerate(sorted(pc.EXPECTED_HEAVY_ATOMS["HIS"]))
            ],
        )
        chains = self.chains(tmp_path, [*residues, histidine])
        bonds = pc.find_disulfides(chains)
        target = tmp_path / "p.pdb"
        pc.write_protein_pdb(chains, bonds, target)
        names = Counter(
            line[17:20]
            for line in target.read_text().splitlines()
            if line.startswith("ATOM")
        )
        assert set(names) == {"CYX", "GLY", "HIE"}
        free = Counter(
            line[17:20]
            for line in (tmp_path / "p.pdb").read_text().splitlines()
            if line.startswith("ATOM")
        )
        assert free["CYX"] == 6 * 2

    def test_four_character_names_fill_the_field_and_two_letter_elements_shift_left(
        self,
    ):
        assert pc.pdb_atom_name("CA", "C") == " CA "
        assert pc.pdb_atom_name("HD21", "H") == "HD21"
        assert pc.pdb_atom_name("FE", "FE") == "FE  "


pytestmark_engine = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_OPENFF to run the preparation",
)


def run(params, tmp_path, name="run"):
    from alambiq.api import run as execute

    step = {
        "id": "prep",
        "capability": "structure.prepare_predicted_complex_for_dynamics",
        "params": params,
    }
    return execute({"id": "prep", "steps": [step]}, run_dir=tmp_path / name)


def request(files, **changes):
    return {
        "structure": str(files["cif"]),
        "ligand_pose": str(files["pose"]),
        "ligand_chemistry": str(files["chemistry"]),
        "padding_nm": 0.8,
        "minimize_iterations": 300,
        **changes,
    }


def error(outcome) -> str:
    assert not outcome.ok
    return outcome.steps["prep"].error


@pytestmark_engine
class TestRefusals:
    def test_a_pose_from_another_structure_is_refused(self, complex_files, tmp_path):
        outcome = run(
            request(complex_files, ligand_pose=str(complex_files["mirror_pose"])),
            tmp_path,
        )
        assert "differs from the digest of structure" in error(outcome)

    def test_a_pose_with_the_wrong_handedness_is_refused(self, complex_files, tmp_path):
        outcome = run(
            request(
                complex_files,
                structure=str(complex_files["mirror_cif"]),
                ligand_pose=str(complex_files["mirror_pose"]),
            ),
            tmp_path,
        )
        assert "different stereochemistry from the intended SMILES" in error(outcome)

    def test_a_declared_charge_that_differs_from_the_ligands_is_refused(
        self, complex_files, tmp_path
    ):
        outcome = run(request(complex_files, ligand_formal_charge=0), tmp_path)
        assert "formal charge -1, but 0 was declared" in error(outcome)

    def test_a_declared_sequence_that_differs_from_the_structure_is_refused(
        self, complex_files, tmp_path
    ):
        outcome = run(request(complex_files, protein_sequences=["A" * 20]), tmp_path)
        assert "sequence mismatch for chain 0" in error(outcome)

    def test_ligand_atoms_that_are_not_in_the_chemistry_are_refused(
        self, complex_files, tmp_path
    ):
        chemistry = json.loads(complex_files["chemistry"].read_text())
        chemistry["atoms"][0]["name"] = "Q9"
        changed = tmp_path / "other-chemistry.json"
        changed.write_text(json.dumps(chemistry))
        outcome = run(request(complex_files, ligand_chemistry=str(changed)), tmp_path)
        assert "differ from those in ligand_chemistry" in error(outcome)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            (
                {"padding_nm": 0.1},
                "padding_nm is 0.1; a value from 0.5 to 3.0 is required",
            ),
            ({"ionic_strength_molar": 2.0}, "ionic_strength_molar is 2.0"),
            ({"minimize_iterations": -1}, "minimize_iterations is -1"),
            ({"seed": 0}, "seed is 0"),
            ({"threads": 0}, "threads is 0"),
        ],
    )
    def test_settings_outside_their_limits_are_refused_by_name(
        self, complex_files, tmp_path, changes, message
    ):
        assert message in error(run(request(complex_files, **changes), tmp_path))


@pytestmark_engine
class TestPreparedComplex:
    @pytest.fixture(scope="class")
    def outcome(self, complex_files, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_OPENFF"] = ENV
        return run(
            request(
                complex_files,
                protein_sequences=[SEQUENCE],
                ligand_formal_charge=-1,
                box_shape="octahedron",
            ),
            tmp_path_factory.mktemp("prepared"),
        )

    @pytest.fixture(scope="class")
    def out(self, outcome):
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        return outcome.outputs("prep")

    def topology_counts(self, out) -> Counter:
        residues = {}
        for line in Path(out["topology_pdb"]).read_text().splitlines():
            if line.startswith(("ATOM", "HETATM")):
                residues[(line[21], line[22:27])] = line[17:20].strip()
        return Counter(residues.values())

    def test_the_composition_is_what_the_topology_shows(self, out):
        counts = self.topology_counts(out)
        atoms = [
            line
            for line in Path(out["topology_pdb"]).read_text().splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]
        assert len(atoms) == out["atom_count"]
        assert counts["LIG"] == 1
        assert counts["HOH"] == out["water_molecules"] > 100
        assert counts["NA"] + counts["Na+"] == out["sodium_ions"]
        assert counts["CL"] + counts["Cl-"] == out["chloride_ions"]
        protein = sum(
            v
            for k, v in counts.items()
            if k not in {"LIG", "HOH", "NA", "CL", "Na+", "Cl-"}
        )
        assert protein == out["protein_residues"] == 20
        ligand_atoms = sum(1 for line in atoms if line[17:20].strip() == "LIG")
        assert ligand_atoms == 12 and out["ligand_heavy_atoms"] == 6

    def test_the_system_is_neutral_and_the_ligand_charges_sum_to_its_charge(self, out):
        assert abs(out["net_charge_e"]) < 1e-3
        assert out["ligand_formal_charge"] == -1
        assert out["ligand_charge_sum_e"] == pytest.approx(-1.0, abs=1e-6)
        parameters = json.loads(Path(out["ligand_parameters"]).read_text())
        assert sum(parameters["partial_charges_e"]) == pytest.approx(-1.0, abs=1e-6)
        assert parameters["atom_count"] == 12
        assert sorted(parameters["atom_elements"]) == sorted(
            [6, 6, 6, 7, 8, 8] + [1] * 6
        )

    def test_ions_balance_the_solute_charge_and_salt(self, out):
        # Trp-cage is +1 and alaninate -1, so the solute is neutral and ions come in pairs.
        assert out["sodium_ions"] == out["chloride_ions"]

    def test_minimisation_lowered_the_energy(self, out):
        assert (
            out["potential_energy_after_kj_mol"] < out["potential_energy_before_kj_mol"]
        )
        assert 0 < out["rms_force_after_kj_mol_nm"] < out["max_force_after_kj_mol_nm"]
        assert out["box_volume_nm3"] > 0

    def test_the_protein_is_untouched_in_its_chirality_and_planarity(self, out):
        assert out["inverted_alpha_residues"] == []
        assert out["max_peptide_planarity_deviation_degrees"] < 30

    def test_the_force_field_and_charge_model_are_recorded_with_their_digests(
        self, out
    ):
        field = out["force_field"]
        assert field["ligand"] == "openff_unconstrained-2.3.0.offxml"
        assert field["charges"] == "openff-gnn-am1bcc-1.0.0.pt"
        assert len(field["charge_model_sha256"]) == 64
        assert set(field["force_field_sha256"]) == {
            "openff_unconstrained-2.3.0.offxml",
            "openff-2.3.0.offxml",
        }

    def test_the_manifest_records_the_choices_that_were_made(self, out):
        manifest = json.loads(Path(out["manifest"]).read_text())
        assert manifest["box_shape"] == "octahedron" and manifest["padding_nm"] == 0.8
        assert manifest["ionic_strength_molar"] == 0.15 and manifest["seed"] == 1
        assert manifest["versions"]["openff-toolkit"].startswith("0.")
        assert manifest["hydrogen_variants"][0] is None
        assert "approximate" in manifest["salt_note"]

    def test_the_amber_topology_describes_the_same_atoms_as_the_system(self, out):
        lines = Path(out["amber_topology"]).read_text().splitlines()
        start = lines.index("%FLAG POINTERS")
        # The first of the pointers is the number of atoms.
        assert int(lines[start + 2].split()[0]) == out["atom_count"]

    def test_the_protein_file_is_the_one_read_from_the_prediction(
        self, out, complex_files
    ):
        text = Path(out["protein_pdb"]).read_text()
        assert text.count("\nATOM") + text.startswith("ATOM") == 153

    @pytest.mark.skipif(
        not (ENV_MD and Path(ENV_MD).is_file()),
        reason="set ALAMBIQ_ENV_OPENMM to run the prepared system in dynamics",
    )
    def test_the_system_runs_in_the_dynamics_capability(self, out, tmp_path):
        from alambiq.api import run as execute

        os.environ["ALAMBIQ_ENV_OPENMM"] = ENV_MD
        step = {
            "id": "md",
            "capability": "simulation.run_molecular_dynamics",
            "params": {
                "system": out["system_xml"],
                "state": out["state_xml"],
                "topology": out["topology_pdb"],
                "ensemble": "nvt",
                "thermalize_ps": 0.2,
                "readiness_window_ps": 0,
                "production_ps": 0.2,
                "report_interval": 20,
                "timestep_fs": 2.0,
            },
        }
        outcome = execute({"id": "md", "steps": [step]}, run_dir=tmp_path / "md")
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        result = outcome.outputs("md")
        assert result["particles"] == out["atom_count"]
        assert result["frames"] == 5 and result["production_run"]
        assert 150 < result["mean_temperature_kelvin"] < 450
