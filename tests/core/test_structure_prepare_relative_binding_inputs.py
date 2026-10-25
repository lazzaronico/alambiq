"""structure.prepare_relative_binding_inputs on T4 lysozyme L99A with its deposited benzene.

PDB entry 181L holds benzene in the engineered cavity of T4 lysozyme, 162 of 164
residues modelled. That gives checks with known answers:

* every deposited heavy atom of the protein is where it was deposited, and the two missing
  terminal residues are added only when allowed, naming which they are;
* the six ring carbons of every placed congener sit on the six deposited ring positions
  (the anchor exactly, a substituted ring up to its symmetry);
* the partial charges of a neutral molecule sum to zero, and a symmetric ring has equal
  charges on equivalent atoms;
* the reported clash distance equals the shortest distance from a substituent atom to a
  protein heavy atom, computed here from the files written;
* the atom mappings join like to like, with the ring carbons on the same positions.

The preparation runs in the OpenFE environment (ALAMBIQ_ENV_OPENFE).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
gemmi = pytest.importorskip("gemmi")
Chem = pytest.importorskip("rdkit.Chem")

ENV = os.environ.get("ALAMBIQ_ENV_OPENFE", "")
STRUCTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/181L.cif"
RING = ["C1", "C2", "C3", "C4", "C5", "C6"]
LIGANDS = [
    {"id": "benzene", "smiles": "c1ccccc1", "embedding_seed": 1},
    {"id": "toluene", "smiles": "Cc1ccccc1", "embedding_seed": 2},
    {"id": "ethylbenzene", "smiles": "CCc1ccccc1", "embedding_seed": 3},
]
EDGES = [
    {"id": "bt", "a": "benzene", "b": "toluene"},
    {"id": "be", "a": "benzene", "b": "ethylbenzene"},
]

pytestmark = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_OPENFE to run the binding-input preparation",
)


def run(params, tmp_path, name="run"):
    from alambiq.api import run as execute

    step = {
        "id": "prep",
        "capability": "structure.prepare_relative_binding_inputs",
        "params": params,
    }
    return execute({"id": "prep", "steps": [step]}, run_dir=tmp_path / name)


def request(**changes):
    return {
        "structure": str(STRUCTURE),
        "structure_format": "mmcif",
        "chain_id": "A",
        "anchor_residue_name": "BNZ",
        "anchor_atom_names": RING,
        "ligands": LIGANDS,
        "edges": EDGES,
        "complete_terminals": True,
        **changes,
    }


def error(outcome) -> str:
    assert not outcome.ok
    return outcome.steps["prep"].error


def deposited():
    """The deposited protein heavy atoms of chain A and the deposited ring, from the file."""
    model = gemmi.read_structure(str(STRUCTURE))[0]
    protein, ring = [], None
    for residue in model["A"]:
        if residue.name == "BNZ":
            ring = {a.name: np.array([a.pos.x, a.pos.y, a.pos.z]) for a in residue}
        elif residue.het_flag != "H":
            protein.append(
                {
                    a.name: np.array([a.pos.x, a.pos.y, a.pos.z])
                    for a in residue
                    if a.element.name != "H"
                }
            )
    return protein, ring


def pdb_heavy(path: str) -> list[dict[str, np.ndarray]]:
    residues: dict[tuple, dict] = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[76:78].strip() != "H":
            key = (line[21], line[22:27])
            residues.setdefault(key, {})[line[12:16].strip()] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    return list(residues.values())


class TestRefusingWhatItWouldHaveToGuess:
    def test_missing_terminal_residues_are_modelled_only_when_allowed(self, tmp_path):
        message = error(run(request(complete_terminals=False), tmp_path))
        assert "missing terminal residues" in message and "['ASN', 'LEU']" in message
        assert "set complete_terminals" in message

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"chain_id": "Z"}, "no chain Z"),
            ({"anchor_residue_name": "XYZ"}, "exactly one anchor ring is needed"),
            (
                {"anchor_atom_names": ["C1", "C2", "C3", "C4", "C5", "C7"]},
                "are not the declared",
            ),
            ({"anchor_atom_names": RING[:5]}, "6 distinct ring carbon atom names"),
            ({"anchor_atom_names": [*RING[:5], "C1"]}, "6 distinct ring carbon"),
            ({"ph": 15.0}, "ph is 15.0; a value from 0 to 14 is required"),
            ({"seed": 0}, "seed is 0"),
            ({"threads": 0}, "threads is 0"),
        ],
    )
    def test_a_recipe_the_structure_cannot_satisfy_is_refused_in_plain_words(
        self, tmp_path, changes, message
    ):
        outcome = run(request(**changes), tmp_path)
        assert message in error(outcome)

    def test_the_first_ligand_must_be_benzene(self, tmp_path):
        ligands = [{**LIGANDS[1], "id": "first"}, LIGANDS[0]]
        outcome = run(
            request(ligands=ligands, edges=[{"id": "e", "a": "first", "b": "benzene"}]),
            tmp_path,
        )
        assert "must be benzene" in error(outcome)

    @pytest.mark.parametrize(
        ("smiles", "message"),
        [
            ("c1ccc2ccccc2c1", "exactly one ring, six aromatic carbons"),
            ("C1CCCCC1", "exactly one ring, six aromatic carbons"),
            ("c1ccncc1", "exactly one ring, six aromatic carbons"),
            ("[O-]c1ccccc1", "one connected neutral molecule"),
            ("Cc1ccccc1.Cc1ccccc1", "one connected neutral molecule"),
            ("Cc1cc(C)ccc1", "heavy-atom substituents on exactly one ring position"),
            ("CC(F)c1ccccc1", "leaves stereochemistry unspecified"),
        ],
    )
    def test_a_congener_that_is_not_a_single_substituted_benzene_is_refused(
        self, tmp_path, smiles, message
    ):
        ligands = [LIGANDS[0], {"id": "x", "smiles": smiles, "embedding_seed": 1}]
        outcome = run(
            request(ligands=ligands, edges=[{"id": "e", "a": "benzene", "b": "x"}]),
            tmp_path,
        )
        assert message in error(outcome)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"ligands": [LIGANDS[0]]}, "the benzene anchor and 1 to 19 congeners"),
            (
                {"ligands": [LIGANDS[0], {**LIGANDS[1], "id": "benzene"}]},
                "distinct and start with a letter",
            ),
            (
                {"ligands": [LIGANDS[0], {**LIGANDS[1], "id": "1bad"}]},
                "distinct and start with a letter",
            ),
            (
                {"ligands": [LIGANDS[0], {**LIGANDS[1], "embedding_seed": 0}]},
                "embedding_seed of ligand",
            ),
            ({"edges": []}, "1 to 50 are required"),
            (
                {"edges": [{"id": "e", "a": "benzene", "b": "nothing"}]},
                "two different ligands",
            ),
            (
                {"edges": [{"id": "e", "a": "benzene", "b": "benzene"}]},
                "two different ligands",
            ),
            (
                {"edges": [EDGES[0], {**EDGES[1], "id": "bt"}]},
                "distinct ids",
            ),
        ],
    )
    def test_a_series_that_is_not_well_formed_is_refused(
        self, tmp_path, changes, message
    ):
        assert message in error(run(request(**changes), tmp_path))


class TestPreparedInputs:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_OPENFE"] = ENV
        return run(request(), tmp_path_factory.mktemp("binding"))

    @pytest.fixture(scope="class")
    def out(self, outcome):
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        return outcome.outputs("prep")

    def test_the_missing_terminal_residues_are_named_and_counted(self, out, outcome):
        assert out["deposited_residues"] == 162 and out["protein_residues"] == 164
        assert out["modeled_terminal_residues"] == [
            {"insertion_index": 162, "names": ["ASN", "LEU"]}
        ]
        assert "ASN, LEU at insertion point 162" in " ".join(
            outcome.steps["prep"].notes
        )

    def test_every_deposited_heavy_atom_is_where_it_was_deposited(self, out):
        source, _ = deposited()
        written = pdb_heavy(out["protein_pdb"])
        assert len(written) == 164
        for before, after in zip(source, written[:162], strict=True):
            assert set(before) <= set(after)
            for name, xyz in before.items():
                assert after[name] == pytest.approx(xyz, abs=6e-4), name

    def test_the_anchor_ring_is_the_deposited_ring_atom_for_atom(self, out):
        _, ring = deposited()
        posed = Chem.SDMolSupplier(out["ligands"][0]["sdf"], removeHs=False)[0]
        positions = posed.GetConformer().GetPositions()
        carbons = [a.GetIdx() for a in posed.GetAtoms() if a.GetAtomicNum() == 6]
        for index, name in zip(carbons, RING, strict=True):
            assert positions[index] == pytest.approx(ring[name], abs=1e-3)

    @pytest.mark.parametrize("which", [1, 2])
    def test_a_substituted_ring_lies_on_the_deposited_ring_positions(self, out, which):
        _, ring = deposited()
        posed = Chem.SDMolSupplier(out["ligands"][which]["sdf"], removeHs=False)[0]
        ring_atoms = posed.GetRingInfo().AtomRings()[0]
        found = posed.GetConformer().GetPositions()[list(ring_atoms)]
        for xyz in found:
            nearest = min(np.linalg.norm(xyz - r) for r in ring.values())
            assert nearest < 1e-3

    def test_the_ligands_keep_their_smiles_and_are_neutral_with_charges_in_the_sdf(
        self, out
    ):
        for row, spec in zip(out["ligands"], LIGANDS, strict=True):
            posed = Chem.SDMolSupplier(row["sdf"], removeHs=False)[0]
            assert Chem.MolToSmiles(Chem.RemoveHs(posed)) == Chem.MolToSmiles(
                Chem.MolFromSmiles(spec["smiles"])
            )
            charges = np.array(
                posed.GetProp("atom.dprop.PartialCharge").split(), dtype=float
            )
            assert len(charges) == posed.GetNumAtoms() == row["atoms"]
            assert charges.sum() == pytest.approx(0.0, abs=1e-6)
            assert row["charge_sum_e"] == pytest.approx(0.0, abs=1e-6)

    def test_equivalent_atoms_of_benzene_carry_equal_charges(self, out):
        posed = Chem.SDMolSupplier(out["ligands"][0]["sdf"], removeHs=False)[0]
        charges = np.array(
            posed.GetProp("atom.dprop.PartialCharge").split(), dtype=float
        )
        carbons = [
            c
            for a, c in zip(posed.GetAtoms(), charges, strict=True)
            if a.GetAtomicNum() == 6
        ]
        hydrogens = [
            c
            for a, c in zip(posed.GetAtoms(), charges, strict=True)
            if a.GetAtomicNum() == 1
        ]
        assert (
            max(carbons) - min(carbons) < 1e-6
            and max(hydrogens) - min(hydrogens) < 1e-6
        )
        assert carbons[0] < 0 < hydrogens[0]

    def test_the_reported_clash_distance_is_the_shortest_substituent_to_protein_distance(
        self, out
    ):
        protein = np.array(
            [xyz for res in pdb_heavy(out["protein_pdb"]) for xyz in res.values()]
        )
        for row in out["ligands"][1:]:
            posed = Chem.SDMolSupplier(row["sdf"], removeHs=False)[0]
            ring = set(posed.GetRingInfo().AtomRings()[0])
            added = posed.GetConformer().GetPositions()[
                [
                    a.GetIdx()
                    for a in posed.GetAtoms()
                    if a.GetAtomicNum() > 1 and a.GetIdx() not in ring
                ]
            ]
            distances = np.linalg.norm(added[:, None] - protein[None], axis=2)
            assert row["minimum_substituent_distance_angstrom"] == pytest.approx(
                distances.min(), abs=2e-3
            )
            shortfall = np.maximum(0, 2.2 - distances)
            assert row["overlap_penalty_angstrom2"] == pytest.approx(
                float((shortfall**2).sum()), abs=1e-3
            )

    def test_a_placement_that_clashes_is_reported_in_a_note(self, out, outcome):
        notes = " ".join(outcome.steps["prep"].notes)
        for row in out["ligands"][1:]:
            if row["overlap_penalty_angstrom2"] > 0:
                assert f"{row['id']}: the best placement still has" in notes
            else:
                assert f"{row['id']}: the best placement" not in notes

    def test_the_first_edge_maps_like_to_like_with_the_ring_on_the_ring(self, out):
        edge = out["edges"][0]
        a = Chem.SDMolSupplier(out["ligands"][0]["sdf"], removeHs=False)[0]
        b = Chem.SDMolSupplier(out["ligands"][1]["sdf"], removeHs=False)[0]
        # Benzene has 12 atoms and toluene 15; the substituted hydrogen has no partner.
        assert edge["mapped_atoms"] == 11
        for i, j in edge["atom_mapping"]:
            assert (
                a.GetAtomWithIdx(i).GetAtomicNum() == b.GetAtomWithIdx(j).GetAtomicNum()
            )
            assert (
                a.GetAtomWithIdx(i).GetIsAromatic()
                == b.GetAtomWithIdx(j).GetIsAromatic()
            )
        ring_pairs = [
            (i, j)
            for i, j in edge["atom_mapping"]
            if a.GetAtomWithIdx(i).GetAtomicNum() == 6
        ]
        assert len(ring_pairs) == 6
        for i, j in ring_pairs:
            assert (
                np.linalg.norm(
                    a.GetConformer().GetAtomPosition(i)
                    - b.GetConformer().GetAtomPosition(j)
                )
                < 1e-3
            )
        assert edge["rms_mapped_distance_angstrom"] < 0.05

    def test_the_manifest_records_the_choices_and_the_versions(self, out):
        manifest = json.loads(Path(out["manifest"]).read_text())
        assert manifest["ph"] == 7.0 and manifest["seed"] == 1
        assert (
            manifest["hydrogen_variants"] and len(manifest["hydrogen_variants"]) == 164
        )
        assert len(manifest["ligands"]["toluene"]["candidate_poses"]) == 12
        assert manifest["versions"]["openff-toolkit"].startswith("0.")
        assert len(manifest["force_field"]["charge_model_sha256"]) == 64

    def test_the_same_inputs_give_the_same_ligands_and_heavy_atoms_but_hydrogens_may_shift(
        self, out, tmp_path
    ):
        again = run(request(), tmp_path)
        assert again.ok
        other = again.outputs("prep")
        for first, second in zip(out["ligands"], other["ligands"], strict=True):
            assert Path(first["sdf"]).read_text() == Path(second["sdf"]).read_text()
        before = pdb_heavy(out["protein_pdb"])
        after = pdb_heavy(other["protein_pdb"])
        for x, y in zip(before, after, strict=True):
            assert x.keys() == y.keys()
            for name in x:
                assert (x[name] == y[name]).all(), name
        # The hydrogens are placed by an energy minimisation that is not bit-reproducible
        # (differences of a few hundredths of an angstrom were seen, 0.063 at most). The
        # same atoms must appear in the same order; positions are held to a sanity bound
        # of 0.5 angstrom, which is not a reproducibility claim.
        lines = lambda o: [  # noqa: E731
            line
            for line in Path(o["protein_pdb"]).read_text().splitlines()
            if line.startswith("ATOM")
        ]
        for a, b in zip(lines(out), lines(other), strict=True):
            assert a[:30] == b[:30]
        moved = [
            max(abs(float(a[c : c + 8]) - float(b[c : c + 8])) for c in (30, 38, 46))
            for a, b in zip(lines(out), lines(other), strict=True)
        ]
        assert max(moved) < 0.5
