"""structure.assign_protonation: the request checks, the file reading and the comparison of
heavy atoms are tested on text written here; the protonation itself runs in the protonation
environment on fixtures with known residues (set ALAMBIQ_ENV_PROTONATION to its Python).

* T4 lysozyme (181L) has one histidine, His31, with a predicted pKa just above pH 7, a
  C-terminal lysine without OXT, two chloride ions, one benzene and one 2-mercaptoethanol.
* Human deoxyhemoglobin (1A3N) has 4 chains, 38 histidines and the heme groups.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/structures"
ENV = os.environ.get("ALAMBIQ_ENV_PROTONATION", "")


def engine():
    spec = importlib.util.spec_from_file_location(
        "pdb2pqr_protonate", ROOT / "src/alambiq/engines/pdb2pqr_protonate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atom(serial, name, residue, number, xyz, *, record="ATOM  ", chain="A", altloc=" ", element=None, insertion=" "):
    element = element or name[0]
    padded = f" {name:<3}" if len(name) < 4 and len(element) == 1 else f"{name:<4}"
    x, y, z = xyz
    return (
        f"{record}{serial:5d} {padded}{altloc}{residue:>3} {chain}{number:4d}{insertion}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {element:>2}"
    )


def write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


class TestReadingTheInput:
    def test_only_the_first_model_is_read(self, tmp_path):
        lines = [
            atom(1, "N", "ALA", 1, (0, 0, 0)),
            atom(2, "CA", "ALA", 1, (1.4, 0, 0)),
            "ENDMDL",
            atom(3, "N", "GLY", 2, (5, 0, 0)),
        ]
        read = engine().read_input(write(tmp_path / "a.pdb", lines))
        assert len(read["lines"]) == 2

    def test_water_hydrogens_and_heteroatoms_are_left_out_and_counted(self, tmp_path):
        lines = [
            atom(1, "N", "ALA", 1, (0, 0, 0)),
            atom(2, "H", "ALA", 1, (0, 1, 0), element="H"),
            atom(3, "O", "HOH", 9, (3, 0, 0), record="HETATM"),
            atom(4, "C1", "BNZ", 10, (6, 0, 0), record="HETATM"),
            atom(5, "C2", "BNZ", 10, (7, 0, 0), record="HETATM"),
        ]
        read = engine().read_input(write(tmp_path / "a.pdb", lines))
        assert len(read["lines"]) == 1
        assert read["waters"] == 1
        assert read["hydrogens"] == 1
        assert read["heteroatoms"] == {"BNZ": 2}

    def test_the_first_alternate_location_of_an_atom_is_kept(self, tmp_path):
        lines = [
            atom(1, "CA", "ALA", 1, (0, 0, 0), altloc="B"),
            atom(2, "CA", "ALA", 1, (0.5, 0, 0), altloc="A"),
        ]
        read = engine().read_input(write(tmp_path / "a.pdb", lines))
        assert len(read["lines"]) == 1
        assert read["alternates"] == 1
        assert read["lines"][0][16] == " "
        assert read["lines"][0][30:38].strip() == "0.000"

    def test_an_insertion_code_is_found(self, tmp_path):
        lines = [atom(1, "N", "ALA", 5, (0, 0, 0), insertion="A")]
        read = engine().read_input(write(tmp_path / "a.pdb", lines))
        assert read["insertion_codes"] == ["A:ALA5A"]


class TestComparingHeavyAtoms:
    def test_only_the_residues_flagged_as_flipped_are_turned_residues(self):
        flips = {"A|25|ASN": True, "A|40|GLN": False, "B|68|HIS": True}
        assert engine().turned_residues(flips) == [
            {"chain": "A", "number": 25, "name": "ASN"},
            {"chain": "B", "number": 68, "name": "HIS"},
        ]

    def test_no_flags_give_no_turned_residues(self):
        assert engine().turned_residues({}) == []

    def test_atoms_present_only_in_the_output_are_the_added_ones(self):
        before = {("A", 3, "C"): ((0.0, 0.0, 0.0), "LYS")}
        after = {("A", 3, "C"): ((0.0, 0.0, 0.0), "LYS"), ("A", 3, "OXT"): ((1.0, 0.0, 0.0), "LYS")}
        assert engine().added_heavy_atoms(before, after) == [
            {"chain": "A", "number": 3, "name": "LYS", "atoms": ["OXT"]}
        ]


SUMMARY = """\
noise before
SUMMARY OF THIS PREDICTION
       Group      pKa  model-pKa   ligand atom-type
   ASP  21 A     3.36       3.80
   HIS  68 B     6.00       6.50
   CYS   6 A    99.99       9.00
   N+    1 A     8.07       8.00
   C-   76 A     3.16       3.20

--------------------------------
   LYS 99 A 10.0 10.5
"""


class TestTitrationTable:
    def test_rows_of_the_summary_are_read_up_to_the_blank_line(self):
        rows = engine().parse_titration(SUMMARY)
        assert [(r["group"], r["chain"], r["number"], r["pka"], r["model_pka"]) for r in rows] == [
            ("ASP", "A", 21, 3.36, 3.8),
            ("HIS", "B", 68, 6.0, 6.5),
            ("CYS", "A", 6, 99.99, 9.0),
            ("N+", "A", 1, 8.07, 8.0),
            ("C-", "A", 76, 3.16, 3.2),
        ]

    def test_a_log_without_the_summary_gives_no_rows(self):
        assert engine().parse_titration("nothing here") == []

    def test_a_group_is_protonated_when_its_pka_is_above_the_ph(self):
        rows = engine().parse_titration(SUMMARY)
        states = engine().titration_states(rows, {("B", 68): "HIS"}, {("B", 68): "HIP"}, 6.5)
        by_group = {s["group"]: s for s in states}
        assert by_group["HIS"]["protonated_at_ph"] is False
        assert by_group["HIS"]["name_out"] == "HIP"
        assert by_group["N+"]["protonated_at_ph"] is True


class TestRequestChecks:
    @pytest.fixture(autouse=True)
    def interpreter(self, monkeypatch):
        """The request is refused before the engine starts, so any interpreter serves."""
        monkeypatch.setenv("ALAMBIQ_ENV_PROTONATION", ENV or sys.executable)

    def run(self, tmp_path, params, text="ATOM\n", name="a.pdb"):
        from alambiq.api import run as execute

        path = tmp_path / name
        path.write_text(text)
        outcome = execute(
            {
                "id": "c",
                "steps": [
                    {
                        "id": "p",
                        "capability": "structure.assign_protonation",
                        "params": {"structure": str(path), **params},
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        assert not outcome.ok
        return " ".join(s.error for s in outcome.steps.values() if s.error)

    def test_a_ph_outside_0_to_14_is_refused(self, tmp_path):
        assert "ph is 15" in self.run(tmp_path, {"ph": 15})

    def test_a_file_that_is_not_a_pdb_file_is_refused(self, tmp_path):
        assert "not a PDB file" in self.run(tmp_path, {}, name="a.cif")

    def test_a_short_timeout_is_refused(self, tmp_path):
        assert "timeout_seconds is 5" in self.run(tmp_path, {"timeout_seconds": 5})

    def test_the_optimization_switch_is_accepted_in_a_refused_request(self, tmp_path):
        message = self.run(tmp_path, {"optimize_hydrogen_network": False, "ph": -1})
        assert "ph is -1" in message


def pdb_of(name: str, tmp_path: Path) -> Path:
    gemmi = pytest.importorskip("gemmi")
    structure = gemmi.read_structure(str(FIXTURES / f"{name}.cif"))
    structure.setup_entities()
    path = tmp_path / f"{name}.pdb"
    structure.write_pdb(str(path))
    return path


def protein_only(name: str, tmp_path: Path) -> Path:
    """The amino acid atoms of a fixture as one chain, as the preparation capabilities read it."""
    lines = [line for line in pdb_of(name, tmp_path).read_text().splitlines() if line.startswith("ATOM")]
    path = tmp_path / f"{name}-protein.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def protonate(tmp_path: Path, name: str, **params) -> dict:
    from alambiq.api import run as execute

    outcome = execute(
        {
            "id": "c",
            "steps": [
                {
                    "id": "p",
                    "capability": "structure.assign_protonation",
                    "params": {"structure": str(pdb_of(name, tmp_path)), **params},
                }
            ],
        },
        run_dir=tmp_path / "run",
    )
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("p")


needs_engine = pytest.mark.skipif(
    not ENV, reason="needs ALAMBIQ_ENV_PROTONATION (the protonation environment)"
)


@needs_engine
class TestProtonation:
    @pytest.fixture(scope="class")
    def lysozyme(self, tmp_path_factory):
        return protonate(tmp_path_factory.mktemp("t4l"), "181L")

    def test_the_histidine_is_charged_when_its_pka_is_above_the_ph(self, lysozyme):
        his = next(g for g in lysozyme["groups"] if g["group"] == "HIS")
        assert his["pka"] > 7.0 and his["protonated"] and his["name_out"] == "HIP"

    def test_a_higher_ph_makes_the_histidine_neutral(self, tmp_path):
        his = next(g for g in protonate(tmp_path, "181L", ph=9.0)["groups"] if g["group"] == "HIS")
        assert not his["protonated"] and his["name_out"] in ("HID", "HIE")

    def test_the_missing_terminal_oxygen_is_reported_as_added(self, lysozyme):
        added = {(a["chain"], a["number"]): a["atoms"] for a in lysozyme["added_heavy_atoms"]}
        assert added == {("A", 162): ["OXT"]}

    def test_water_ligands_and_ions_are_counted_as_left_out(self, lysozyme):
        assert lysozyme["dropped_waters"] > 0
        assert set(lysozyme["dropped_heteroatoms"]) == {"CL", "HED", "BNZ"}

    def test_the_protonated_file_has_hydrogens_and_no_heteroatoms(self, lysozyme):
        lines = Path(lysozyme["protonated_structure"]).read_text().splitlines()
        assert not any(line.startswith("HETATM") for line in lines)
        hydrogens = [line for line in lines if line.startswith("ATOM") and line[76:78].strip() == "H"]
        assert len(hydrogens) == lysozyme["hydrogens"] > 1000

    def test_the_hash_names_the_file_that_was_read(self, lysozyme, tmp_path):
        import hashlib

        path = pdb_of("181L", tmp_path)
        assert lysozyme["structure_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_a_cysteine_in_a_disulfide_is_named_cyx_and_listed_apart(self, tmp_path):
        out = protonate(tmp_path, "1A3N")
        assert out["disulfide_cysteines"] == []
        assert all(g["group"] != "CYS" or g["pka"] < 99 for g in out["groups"])


@needs_engine
class TestHydrogenNetwork:
    @pytest.fixture(scope="class")
    def hemoglobin(self, tmp_path_factory):
        return protonate(tmp_path_factory.mktemp("hb"), "1A3N")

    def test_an_asparagine_amide_turns(self, hemoglobin):
        assert any(t["name"] == "ASN" and t["chain"] == "C" and t["number"] == 78 for t in hemoglobin["turned"])

    def test_a_turned_residue_has_its_two_amide_atoms_exchanged(self, hemoglobin, tmp_path):
        """The flag agrees with the coordinates: the oxygen sits where the nitrogen was."""
        import math

        engine_module = engine()
        before = engine_module.heavy_atoms(pdb_of("1A3N", tmp_path).read_text().splitlines())
        after = engine_module.heavy_atoms(Path(hemoglobin["protonated_structure"]).read_text().splitlines())
        o, n = ("C", 78, "OD1"), ("C", 78, "ND2")
        assert math.dist(before[o][0], after[n][0]) < 0.6
        assert math.dist(before[n][0], after[o][0]) < 0.6

    def test_the_heme_is_left_out_and_noted(self, hemoglobin):
        assert hemoglobin["dropped_heteroatoms"] == {"HEM": 172}

    def test_every_histidine_gets_a_tautomer_or_a_charge(self, hemoglobin):
        names = {g["name_out"] for g in hemoglobin["groups"] if g["group"] == "HIS"}
        assert names <= {"HID", "HIE", "HIP"}

    def test_without_the_optimization_nothing_turns(self, tmp_path):
        out = protonate(tmp_path, "1A3N", optimize_hydrogen_network=False)
        assert out["turned"] == []


OPENMM = os.environ.get("ALAMBIQ_ENV_OPENFE", "")


@pytest.mark.skipif(
    not (ENV and OPENMM), reason="needs ALAMBIQ_ENV_PROTONATION and ALAMBIQ_ENV_OPENFE"
)
class TestUseInOpenMM:
    def test_amber14_builds_a_system_from_the_protonated_structure(self, tmp_path):
        """Every residue and hydrogen name of the output is one the force field knows."""
        import subprocess

        out = protonate(tmp_path, "181L")
        script = (
            "import sys, openmm.app as app;"
            "pdb = app.PDBFile(sys.argv[1]);"
            "ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml');"
            "system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds);"
            "print(system.getNumParticles())"
        )
        done = subprocess.run(
            [OPENMM, "-c", script, out["protonated_structure"]], capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr[-500:]
        assert int(done.stdout.split()[-1]) == out["atoms"]


class TestVariantsFromAProtonatedStructure:
    def text(self, rows):
        lines = [atom(i, "CA", name, number, (float(i), 0, 0), chain=chain) for i, (chain, number, name) in enumerate(rows, 1)]
        return "\n".join(lines) + "\nEND\n"

    def test_titratable_residues_of_the_chain_give_their_variant_by_number(self, tmp_path):
        from alambiq.capabilities.investigate.science.protonation import (
            variants_from_structure,
        )

        path = tmp_path / "p.pdb"
        path.write_text(self.text([("A", 3, "HIP"), ("A", 4, "LEU"), ("A", 5, "ASH"), ("B", 3, "HID")]))
        assert variants_from_structure(path, "A") == {3: "HIP", 5: "ASH"}

    def test_a_chain_that_is_absent_is_refused(self, tmp_path):
        from alambiq.capabilities.investigate.science.protonation import (
            variants_from_structure,
        )

        path = tmp_path / "p.pdb"
        path.write_text(self.text([("A", 3, "HIP")]))
        with pytest.raises(ValueError, match="no chain 'Z'"):
            variants_from_structure(path, "Z")

    def test_an_insertion_code_is_refused(self, tmp_path):
        from alambiq.capabilities.investigate.science.protonation import (
            variants_from_structure,
        )

        path = tmp_path / "p.pdb"
        path.write_text(atom(1, "CA", "HIP", 3, (0, 0, 0), insertion="A") + "\nEND\n")
        with pytest.raises(ValueError, match="insertion code"):
            variants_from_structure(path, "A")


OPENMM_ENV = os.environ.get("ALAMBIQ_ENV_OPENMM", "")


def variants_of(manifest_path: str) -> dict:
    import json

    manifest = json.loads(Path(manifest_path).read_text())

    def find(node):
        if isinstance(node, dict):
            if "residue_variants" in node:
                return node["residue_variants"]
            for value in node.values():
                found = find(value)
                if found is not None:
                    return found
        return None

    return find(manifest)


@pytest.mark.skipif(
    not (ENV and OPENMM_ENV),
    reason="needs ALAMBIQ_ENV_PROTONATION and ALAMBIQ_ENV_OPENMM (with PDBFixer for the terminal atom)",
)
class TestProteinSystemUsesTheProtonation:
    def prepare(self, tmp_path, **params):
        from alambiq.api import run as execute

        outcome = execute(
            {
                "id": "c",
                "steps": [
                    {
                        "id": "p",
                        "capability": "structure.prepare_protein_system",
                        "params": {
                            "structure": str(protein_only("181L", tmp_path)),
                            "chain_id": "A",
                            "padding_nm": 1.0,
                            "heavy_atom_completion": "pdbfixer-standard",
                            "minimization_max_iterations": 20,
                            **params,
                        },
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        return outcome.outputs("p")

    def test_the_states_of_the_file_become_the_variants_of_the_system(self, tmp_path):
        (tmp_path / "pq").mkdir()
        protonated = protonate(tmp_path / "pq", "181L", ph=5.0)["protonated_structure"]
        out = self.prepare(tmp_path, protonated_structure=protonated)
        assert variants_of(out["manifest"])[30] == "HIP"

    def test_a_variant_named_in_the_request_wins_over_the_file(self, tmp_path):
        (tmp_path / "pq").mkdir()
        protonated = protonate(tmp_path / "pq", "181L", ph=5.0)["protonated_structure"]
        out = self.prepare(
            tmp_path,
            protonated_structure=protonated,
            residue_variants=[{"residue": "31", "variant": "HIE"}],
        )
        assert variants_of(out["manifest"])[30] == "HIE"


OPENFE_ENV = os.environ.get("ALAMBIQ_ENV_OPENFE", "")


@pytest.mark.skipif(not (ENV and OPENFE_ENV), reason="needs ALAMBIQ_ENV_PROTONATION and ALAMBIQ_ENV_OPENFE")
class TestMutationInputsUseTheProtonation:
    def legs(self, tmp_path, **params):
        import json
        import tarfile

        from alambiq.api import run as execute

        outcome = execute(
            {
                "id": "c",
                "steps": [
                    {
                        "id": "p",
                        "capability": "structure.prepare_mutation_inputs",
                        "params": {
                            "structure": str(protein_only("181L", tmp_path)),
                            "chain_id": "A",
                            "residue_number": 84,
                            "to": "ALA",
                            "reference_padding_nm": 1.0,
                            "folded_padding_nm": 1.0,
                            **params,
                        },
                    }
                ],
            },
            run_dir=tmp_path / "run",
        )
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        out = outcome.outputs("p")
        with tarfile.open(out["folded_inputs"]) as bundle:
            record = json.load(bundle.extractfile("leg.json"))
            wild_type = bundle.extractfile("wild_type.pdb").read().decode()
        return record, wild_type

    @staticmethod
    def ring_hydrogens(pdb_text: str, number: int) -> set[str]:
        """OpenMM writes every histidine as HIS; the state shows in the hydrogens HD1 and HE2."""
        return {
            line[12:16].strip()
            for line in pdb_text.splitlines()
            if line.startswith("ATOM") and int(line[22:26]) == number and line[12:16].strip() in ("HD1", "HE2")
        }

    def test_a_histidine_takes_the_state_of_the_protonated_structure(self, tmp_path):
        (tmp_path / "pq").mkdir()
        protonated = protonate(tmp_path / "pq", "181L", ph=5.0)["protonated_structure"]
        record, wild = self.legs(tmp_path, protonated_structure=protonated)
        assert record["residue_variants"]["31"] == "HIP"
        assert self.ring_hydrogens(wild, 31) == {"HD1", "HE2"}

    def test_without_it_the_histidine_is_neutral_at_ph_7(self, tmp_path):
        record, wild = self.legs(tmp_path)
        assert record["residue_variants"] == {}
        assert len(self.ring_hydrogens(wild, 31)) == 1
