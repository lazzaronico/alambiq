"""structure.prepare_reactive_coordinates on Trp-cage with a donor, an analogue and a substrate.

The target is Trp-cage with its first two residues unresolved and an analogue (three heavy
atoms) in a site beside it; the donor is the complete Trp-cage; the substrate is ethanol
deposited elsewhere. Because the donor is the same structure the target was cut from,
and the analogue is the substrate moved by a known rigid motion, the answers are known:

* the donor's first two residues superpose on the target with zero RMSD, so the modelled
  atoms are the donor's atoms exactly, and the new peptide bond has the donor's length;
* every observed atom is where it was deposited;
* the substrate's heavy atoms land on the analogue's, with zero RMSD, and with the analogue
  stretched the RMSD is the least-squares value computed here from the Kabsch solution;
* a water within the exclusion distance of a modelled residue is removed and a far one is kept;
* each way the sources can disagree with the recipe is refused by name.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
gemmi = pytest.importorskip("gemmi")
Chem = pytest.importorskip("rdkit.Chem")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/1L2Y.cif"
SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"
SOURCE = np.array(
    [[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [2.07, 1.33, 0.0]]
)  # ethanol C, C, O


def one_model():
    structure = gemmi.read_structure(str(FIXTURE))
    structure.setup_entities()
    del structure[1:]
    structure.remove_hydrogens()
    return structure


def add_residue(structure, chain_name, name, number, names, elements, xyz, subchain):
    residue = gemmi.Residue()
    residue.name, residue.seqid, residue.het_flag = name, gemmi.SeqId(number, " "), "H"
    residue.entity_type = (
        gemmi.EntityType.Water if name == "HOH" else gemmi.EntityType.NonPolymer
    )
    residue.subchain = subchain
    for n, e, p in zip(names, elements, xyz, strict=True):
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos, atom.occ = (
            n,
            gemmi.Element(e),
            gemmi.Position(*p),
            1.0,
        )
        residue.add_atom(atom)
    next(c for c in structure[0] if c.name == chain_name).add_residue(residue)


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(axis=1).mean()))


def write_target(path: Path, analogue: np.ndarray, waters=()):
    target = one_model()
    chain = target[0]["A"]
    del chain[0]
    del chain[0]
    add_residue(
        target, "A", "ANL", 101, ["C1", "C2", "O1"], ["C", "C", "O"], analogue, "B"
    )
    subchains = ["B"]
    for number, position in enumerate(waters, 201):
        add_residue(target, "A", "HOH", number, ["O"], ["O"], [position], f"W{number}")
        subchains.append(f"W{number}")
    target.setup_entities()
    for assembly in target.assemblies:
        for generator in assembly.generators:
            generator.subchains = [*generator.subchains, *subchains]
    target.make_mmcif_document().write_file(str(path))
    return target


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reactive")
    donor = one_model()
    donor.make_mmcif_document().write_file(str(directory / "donor.cif"))
    full = np.array([a.pos.tolist() for r in donor[0]["A"] for a in r])
    base = np.array([full[:, 0].max() + 6.0, full[:, 1].mean(), full[:, 2].mean()])
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    analogue = SOURCE @ quarter.T + base
    write_target(directory / "target.cif", analogue)
    source = gemmi.Structure()
    source.name = "SRC"
    model = gemmi.Model(1)
    model.add_chain(gemmi.Chain("A"))
    source.add_model(model)
    add_residue(source, "A", "SUB", 1, ["C1", "C2", "O1"], ["C", "C", "O"], SOURCE, "B")
    source.setup_entities()
    document = source.make_mmcif_document()
    document.sole_block().set_mmcif_category(
        "_chem_comp_bond.",
        {
            "comp_id": ["SUB", "SUB"],
            "atom_id_1": ["C1", "C2"],
            "atom_id_2": ["C2", "O1"],
            "value_order": ["SING", "SING"],
        },
    )
    document.write_file(str(directory / "source.cif"))
    Chem.MolToMolFile(Chem.MolFromSmiles("CCO"), str(directory / "ethanol.mol"))
    return {
        "directory": directory,
        "analogue": analogue,
        "base": base,
        "donor_atoms": donor[0]["A"],
    }


def recipe(files, **changes):
    params = {
        "structure": str(files["directory"] / "target.cif"),
        "donor_structure": str(files["directory"] / "donor.cif"),
        "substrate_source": str(files["directory"] / "source.cif"),
        "molecule": str(files["directory"] / "ethanol.mol"),
        "target": {
            "assembly_id": "1",
            "model_number": 1,
            "chains": [
                {
                    "source_chain": "A1",
                    "id": "A",
                    "sequence": SEQUENCE,
                    "missing_residues": [1, 2],
                }
            ],
        },
        "donor": {"model_number": 1, "chains": ["A"]},
        "substrate": {
            "model_number": 1,
            "chain": "A",
            "residue_number": 1,
            "component": "SUB",
            "source_form": "ethanol, constructed for this test",
            "formal_charge": 0,
            "source_atom_names": ["C1", "C2", "O1"],
            "anchor_component": "ANL",
            "anchor_atom_names": ["C1", "C2", "O1"],
        },
    }
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(params.get(key), dict):
            params[key] = {**params[key], **value}
        else:
            params[key] = value
    return params


def run(params, tmp_path, name="run"):
    from alambiq.api import run as execute

    step = {
        "id": "rc",
        "capability": "structure.prepare_reactive_coordinates",
        "params": params,
    }
    return execute({"id": "rc", "steps": [step]}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("rc")


def pdb_atoms(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (int(line[22:26]), line[12:16].strip())
            rows[key] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    return rows


@pytest.fixture(scope="module")
def built(files, tmp_path_factory):
    outcome = run(recipe(files), tmp_path_factory.mktemp("built"))
    return outcome, ok(outcome)


class TestModelledTermini:
    def test_the_two_missing_residues_come_from_the_donor_with_a_perfect_fit(
        self, built
    ):
        (terminus,) = built[1]["modeled_termini"]
        assert terminus["terminus"] == "N" and terminus["modeled_residues"] == [1, 2]
        assert terminus["donor_chain"] == "A" and terminus["candidates_considered"] == 1
        assert terminus["backbone_fit_rmsd_angstrom"] < 1e-5

    def test_the_modelled_atoms_are_the_donors_atoms(self, built, files):
        atoms = pdb_atoms(built[1]["protein_pdb"])
        for residue in list(files["donor_atoms"])[:2]:
            for atom in residue:
                assert atoms[(residue.seqid.num, atom.name)] == pytest.approx(
                    atom.pos.tolist(), abs=1e-3
                )

    def test_the_new_peptide_bond_has_the_length_the_donor_has(self, built, files):
        residues = list(files["donor_atoms"])
        carbon = next(a for a in residues[1] if a.name == "C").pos
        nitrogen = next(a for a in residues[2] if a.name == "N").pos
        (terminus,) = built[1]["modeled_termini"]
        assert terminus["junction_bond_angstrom"] == pytest.approx(
            carbon.dist(nitrogen), abs=1e-5
        )
        assert 1.2 <= terminus["junction_bond_angstrom"] <= 1.5

    def test_every_observed_atom_is_where_it_was_deposited(self, built, files):
        atoms = pdb_atoms(built[1]["protein_pdb"])
        for residue in list(files["donor_atoms"])[2:]:
            for atom in residue:
                assert atoms[(residue.seqid.num, atom.name)] == pytest.approx(
                    atom.pos.tolist(), abs=1e-3
                )

    def test_the_model_has_all_residues_and_heavy_atoms_and_its_sequence_is_written(
        self, built
    ):
        out = built[1]
        assert out["protein_residues"] == 20 and out["protein_heavy_atoms"] == 154
        assert Path(out["sequence_fasta"]).read_text() == f">A\n{SEQUENCE}\n"

    def test_a_terminus_no_donor_can_join_within_the_stated_bond_length_is_refused(
        self, files, tmp_path
    ):
        outcome = run(
            recipe(
                files,
                terminal_model={
                    "junction_min_angstrom": 0.5,
                    "junction_max_angstrom": 1.0,
                },
            ),
            tmp_path,
        )
        assert (
            "No donor chain meets the terminal-placement criteria"
            in outcome.steps["rc"].error
        )

    def test_a_donor_whose_sequence_differs_is_refused(self, files, tmp_path):
        wrong = copy.deepcopy(recipe(files))
        wrong["target"]["chains"][0]["sequence"] = "A" + SEQUENCE[1:]
        outcome = run(wrong, tmp_path)
        assert not outcome.ok


class TestWaters:
    def test_a_water_next_to_a_modelled_residue_is_removed_and_a_far_one_kept(
        self, files, tmp_path
    ):
        donor_atoms = list(files["donor_atoms"])
        near = (
            np.array(next(a for a in donor_atoms[0] if a.name == "CA").pos.tolist())
            + [1.0, 0.0, 0.0]
        ).tolist()
        far = (files["base"] + [30.0, 0.0, 0.0]).tolist()
        target = tmp_path / "wet.cif"
        write_target(target, files["analogue"], waters=[near, far])
        outcome = run(recipe(files, structure=str(target)), tmp_path)
        out = ok(outcome)
        assert out["removed_waters"] == 1 and out["water_count"] == 1
        assert "1 water(s) within 2.5" in " ".join(outcome.steps["rc"].notes)


class TestSubstrate:
    def test_the_substrate_lands_on_the_analogue_atom_for_atom(self, built, files):
        (substrate,) = built[1]["substrates"]
        molecule = Chem.SDMolSupplier(substrate["sdf"], removeHs=False)[0]
        found = molecule.GetConformer().GetPositions()[:3]
        assert found == pytest.approx(files["analogue"], abs=2e-3)
        assert substrate["analogue_rmsd_angstrom"] < 1e-5

    def test_the_substrate_keeps_its_chemistry_and_gains_its_hydrogens(self, built):
        (substrate,) = built[1]["substrates"]
        assert substrate["heavy_atoms"] == 3 and substrate["atoms"] == 9
        assert (
            substrate["canonical_smiles"] == "CCO" and substrate["formal_charge"] == 0
        )

    def test_nothing_clashes_with_a_substrate_six_angstrom_from_the_protein(
        self, built
    ):
        assert built[1]["substrates"][0]["protein_overlap_score_angstrom2"] == 0.0

    def test_a_stretched_analogue_gives_the_least_squares_rmsd_of_the_two_shapes(
        self, files, tmp_path
    ):
        stretched = files["analogue"].copy()
        stretched[2] += [0.3, 0.0, 0.0]
        target = tmp_path / "stretched.cif"
        write_target(target, stretched)
        out = ok(run(recipe(files, structure=str(target)), tmp_path))
        expected = kabsch_rmsd(SOURCE, stretched)
        assert out["substrates"][0]["analogue_rmsd_angstrom"] == pytest.approx(
            expected, rel=1e-3
        )
        assert expected > 0.05

    def test_a_substrate_placed_on_top_of_the_protein_reports_the_overlap(
        self, files, tmp_path
    ):
        donor_atoms = list(files["donor_atoms"])
        centre = np.mean([a.pos.tolist() for r in donor_atoms for a in r], axis=0)
        moved = SOURCE - SOURCE.mean(axis=0) + centre
        target = tmp_path / "buried.cif"
        write_target(target, moved)
        out = ok(run(recipe(files, structure=str(target)), tmp_path))
        assert out["substrates"][0]["protein_overlap_score_angstrom2"] > 0

    def test_the_display_structure_holds_the_polymer_and_the_substrate(self, built):
        structure = gemmi.read_structure(
            built[1]["display_structure"], format=gemmi.CoorFormat.Mmcif
        )
        names = {c.name: [r.name for r in c] for c in structure[0]}
        assert len(names["A"]) == 20 and names["LA"] == ["SUB"]

    def test_the_manifest_records_the_recipe_and_versions(self, built):
        manifest = json.loads(Path(built[1]["manifest"]).read_text())
        assert manifest["parameters"]["substrate"]["component"] == "SUB"
        assert (
            manifest["versions"]
            if "versions" in manifest
            else manifest["environment_versions"]["gemmi"]
        )
        assert manifest["terminal_templates"][0]["modeled_residues"] == [1, 2]


class TestRefusals:
    def fails(self, files, tmp_path, message, **changes):
        outcome = run(recipe(files, **changes), tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["rc"].error

    def test_residues_that_are_missing_but_not_declared_are_refused(
        self, files, tmp_path
    ):
        self.fails(
            files,
            tmp_path,
            "the declared missing residues must be terminal groups",
            target={
                "chains": [
                    {**recipe(files)["target"]["chains"][0], "missing_residues": [1]}
                ]
            },
        )

    def test_an_unknown_assembly_is_refused(self, files, tmp_path):
        self.fails(
            files,
            tmp_path,
            "biological assemblies of the target structure; exactly 1 is required",
            target={"assembly_id": "9"},
        )

    def test_an_unknown_model_is_refused(self, files, tmp_path):
        self.fails(
            files,
            tmp_path,
            "matches 0 deposited models; exactly 1 is required",
            target={"model_number": 7},
        )

    def test_a_donor_chain_that_is_not_there_is_refused(self, files, tmp_path):
        self.fails(
            files,
            tmp_path,
            "The donor model has 0 chains named Z; exactly 1 is required",
            donor={"chains": ["Z"]},
        )

    def test_a_substrate_with_the_wrong_charge_is_refused(self, files, tmp_path):
        self.fails(
            files,
            tmp_path,
            "the charge must equal substrate.formal_charge and the stereochemistry must be complete",
            substrate={"formal_charge": 1},
        )

    def test_a_deposited_ligand_that_is_not_where_the_recipe_says_is_refused(
        self, files, tmp_path
    ):
        self.fails(
            files,
            tmp_path,
            "match substrate chain",
            substrate={"residue_number": 7},
        )

    def test_mapped_atom_names_the_residue_lacks_are_refused(self, files, tmp_path):
        self.fails(
            files,
            tmp_path,
            "must contain exactly the mapped heavy atoms",
            substrate={"source_atom_names": ["C1", "C2", "X9"]},
        )

    def test_an_analogue_that_is_not_in_the_chain_is_refused(self, files, tmp_path):
        self.fails(
            files,
            tmp_path,
            "exactly 1 analogue anchor is required",
            substrate={"anchor_component": "NOP"},
        )

    def test_a_molecule_with_another_atom_count_is_refused(self, files, tmp_path):
        other = files["directory"] / "propanol.mol"
        Chem.MolToMolFile(Chem.MolFromSmiles("CCCO"), str(other))
        self.fails(files, tmp_path, "matching the source map", molecule=str(other))

    def test_a_histidine_without_a_declared_state_is_refused(self, files, tmp_path):
        # Declare a histidine at a position that is not one.
        chain = {**recipe(files)["target"]["chains"][0], "histidines": {"3": "HIE"}}
        self.fails(
            files,
            tmp_path,
            "name a position that is not a histidine of the sequence",
            target={"chains": [chain]},
        )

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            (
                {"terminal_model": {"anchor_residues": 0}},
                "an integer from 1 to 10 adjacent backbone anchor residues is required",
            ),
            (
                {
                    "terminal_model": {
                        "junction_min_angstrom": 2.0,
                        "junction_max_angstrom": 1.0,
                    }
                },
                "bounds must be ordered",
            ),
            (
                {"terminal_model": {"max_fit_rmsd_angstrom": -1.0}},
                "positive finite angstrom values up to 20 are required",
            ),
            (
                {"substrate": {"hydrogen_max_iterations": 0}},
                "an integer from 1 to 10,000 is required",
            ),
            (
                {"substrate": {"source_atom_names": ["C1", "C2", "O1", "C3"]}},
                "same molecule",
            ),
            (
                {"substrate": {"source_form": ""}},
                "a single line of 1 to 500 characters is required",
            ),
            (
                {"donor": {"chains": []}},
                "donor.chains has 0 entries; 1 to 16 are required",
            ),
        ],
    )
    def test_a_recipe_that_cannot_be_followed_is_refused_in_plain_words(
        self, files, tmp_path, changes, message
    ):
        self.fails(files, tmp_path, message, **changes)
