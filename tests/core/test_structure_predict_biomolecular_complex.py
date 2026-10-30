"""structure.predict_biomolecular_complex: the checks, and the real Boltz-2 on a tiny case.

Boltz-2 is a trained model, so there is no analytic answer for what it predicts, and
this file does not claim one: whether its structures agree with experiment is a
property of the model that these tests do not measure. What can be known exactly is
everything around the model, and that is what is checked:

* the inputs Boltz is given are the inputs the request describes (the YAML it reads,
  the alignment it is handed, normalised so its parser accepts it);
* what comes back is complete and coherent: as many structures as asked for, one
  confidence record each, a number in the ranges those numbers must lie in;
* the ligand's chemistry survives: its SMILES, charge, stereochemistry and atom
  count are those of the request, and the SDF positions equal the predicted
  structure's coordinates for the same atom names, atom for atom;
* each unavailable device is refused rather than replaced.

The real runs use two diffusion steps and one recycling pass, a setting the capability
itself calls unconverged, to test the pipeline in under a minute. Set
ALAMBIQ_ENV_BOLTZ to the Python of the pinned Boltz environment, and
ALAMBIQ_ENV_BOLTZ_MPS to that of the patched one for the Apple GPU case.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from alambiq.capabilities.characterize.science import boltz

ENV = os.environ.get("ALAMBIQ_ENV_BOLTZ", "")
ENV_MPS = os.environ.get("ALAMBIQ_ENV_BOLTZ_MPS", "")
SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"
ALANINATE = "C[C@H](N)C(=O)[O-]"
TINY = {"sampling_steps": 2, "recycling_steps": 1}


class TestAlignmentNormalisation:
    QUERY = "ACDEFGHIK"

    def test_wrapped_rows_are_joined_and_the_row_count_is_returned(self):
        text = f">q\n{self.QUERY[:4]}\n{self.QUERY[4:]}\n>hit one\nAC-EFGHIK\n"
        normalised, rows = boltz.normalize_a3m(text, self.QUERY)
        assert rows == 2
        assert normalised == f">q\n{self.QUERY}\n>hit one\nAC-EFGHIK\n"

    def test_lowercase_insertions_are_kept_and_do_not_count_as_columns(self):
        text = f">q\n{self.QUERY}\n>hit\nACDeeFGHIK-\n"
        # Nine query columns: eight letters plus a gap, with two inserted residues.
        normalised, rows = boltz.normalize_a3m(text, self.QUERY)
        assert rows == 2 and "ACDeeFGHIK-" in normalised

    def test_comments_and_blank_lines_are_ignored(self):
        text = f"# a comment\n\n>q\n{self.QUERY}\n\n"
        assert boltz.normalize_a3m(text, self.QUERY)[1] == 1

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            (">q\nACDEFGHIL\n", "first A3M row must be exactly"),
            ("", "first A3M row must be exactly"),
            ("ACDEFGHIK\n>q\n", "comes before its header"),
            (">\nACDEFGHIK\n", "needs a name"),
            (">q\nACDEFGHIK\n>h\nAC.EFGHIK\n", "dot padding"),
            (">q\nACDEFGHIK\n>h\nACDEFGHI\n", "same number of query columns"),
            (">q\nACDEFGHIK\n>h\nACDEFGHIB\n", "dot padding is not accepted"),
        ],
    )
    def test_alignments_boltz_would_misread_are_refused(self, text, message):
        with pytest.raises(ValueError, match=message):
            boltz.normalize_a3m(text, self.QUERY)


class TestChecks:
    def chain(self, **changes):
        return {"id": "A", "sequence": SEQUENCE, "msa": None, **changes}

    def test_a_valid_complex_passes(self):
        boltz.check_chains(
            [self.chain(), self.chain(id="B_2")], {"id": "L", "smiles": ALANINATE}
        )

    @pytest.mark.parametrize(
        ("proteins", "ligand", "message"),
        [
            ([], None, "Supply 1–8 protein chains"),
            (
                [{"id": f"C{i}", "sequence": SEQUENCE, "msa": None} for i in range(9)],
                None,
                "Supply 1–8",
            ),
            (
                [{"id": "1A", "sequence": SEQUENCE, "msa": None}],
                None,
                "Chain ids must be unique",
            ),
            (
                [{"id": "lysozyme", "sequence": SEQUENCE, "msa": None}],
                None,
                "up to 5 characters",
            ),
            (
                [
                    {"id": "A", "sequence": SEQUENCE, "msa": None},
                    {"id": "A", "sequence": SEQUENCE, "msa": None},
                ],
                None,
                "Chain ids must be unique",
            ),
            (
                [{"id": "A", "sequence": "ACDEB", "msa": None}],
                None,
                "20 amino acids and X",
            ),
            ([{"id": "A", "sequence": "", "msa": None}], None, "1–2000 letters"),
            (
                [{"id": "A", "sequence": "A" * 2001, "msa": None}],
                None,
                "1–2000 letters",
            ),
            (
                [
                    {"id": "A", "sequence": "A" * 1500, "msa": None},
                    {"id": "B", "sequence": "A" * 600, "msa": None},
                ],
                None,
                "At most 2000 protein residues in all",
            ),
            (
                [{"id": "A", "sequence": SEQUENCE, "msa": None}],
                {"id": "A", "smiles": "CCO"},
                "Chain ids must be unique",
            ),
            (
                [{"id": "A", "sequence": SEQUENCE, "msa": None}],
                {"id": "L", "smiles": "C C"},
                "no whitespace",
            ),
            (
                [{"id": "A", "sequence": SEQUENCE, "msa": None}],
                {"id": "L", "smiles": ""},
                "no whitespace",
            ),
        ],
    )
    def test_chains_that_cannot_be_run_are_refused_in_plain_words(
        self, proteins, ligand, message
    ):
        with pytest.raises(ValueError, match=message):
            boltz.check_chains(proteins, ligand)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("seed", -1),
            ("seed", 2**32),
            ("sampling_steps", 1),
            ("sampling_steps", 1001),
            ("diffusion_samples", 0),
            ("diffusion_samples", 21),
            ("recycling_steps", 0),
            ("max_msa_seqs", 0),
            ("num_subsampled_msa", 8193),
            ("threads", 0),
            ("affinity_sampling_steps", 1),
            ("affinity_samples", 21),
        ],
    )
    def test_settings_outside_their_limits_are_refused_by_name(self, name, value):
        with pytest.raises(ValueError, match=f"{name} must be an integer from"):
            boltz.check_settings(**{name: value})

    def test_affinity_without_a_ligand_is_refused(self):
        boltz.check_affinity(True, {"id": "L", "smiles": ALANINATE})
        boltz.check_affinity(False, None)
        with pytest.raises(ValueError, match="affinity needs a ligand"):
            boltz.check_affinity(True, None)

    def test_the_affinity_pass_adds_to_the_ceiling(self):
        base = boltz.timeout_seconds("gpu", 1, 200, 3)
        assert boltz.timeout_seconds("gpu", 1, 200, 3, 5, 200) == 6 * base
        assert boltz.timeout_seconds("gpu", 1, 200, 3, 5, 400) == 11 * base

    def test_a_longer_run_gets_a_longer_ceiling_and_the_cpu_more_than_a_gpu(self):
        base = boltz.timeout_seconds("gpu", 1, 200, 3)
        assert boltz.timeout_seconds("gpu", 5, 200, 3) == 5 * base
        assert boltz.timeout_seconds("gpu", 1, 400, 3) == 2 * base
        assert boltz.timeout_seconds("gpu", 1, 200, 6) == 2 * base
        assert boltz.timeout_seconds("cpu", 1, 200, 3) > base
        # Fewer steps than normal never shrink the ceiling below the base.
        assert boltz.timeout_seconds("gpu", 1, 2, 1) == base


class TestAffinityInputsAndOutputs:
    """What the engine hands Boltz for affinity and how it reads the answer."""

    def cfg(self, **changes):
        return {
            "proteins": [{"id": "A", "sequence": SEQUENCE, "msa": None}],
            "ligand": {"id": "L", "smiles": ALANINATE},
            "model_root": "/models",
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "recycling_steps": 3,
            "accelerator": "cpu",
            "seed": 0,
            "affinity": True,
            "affinity_sampling_steps": 50,
            "affinity_samples": 2,
            "affinity_mw_correction": True,
            **changes,
        }

    def test_the_input_names_the_ligand_as_the_binder(self):
        from alambiq.engines import boltz_predict

        document = boltz_predict._document(self.cfg())
        assert document["properties"] == [{"affinity": {"binder": "L"}}]
        assert "properties" not in boltz_predict._document(self.cfg(affinity=False))

    def test_the_command_carries_the_affinity_settings_only_when_asked(self):
        from alambiq.engines import boltz_predict

        command = boltz_predict._command(self.cfg(), Path("in.yaml"), Path("out"))
        joined = " ".join(command)
        assert "--sampling_steps_affinity 50" in joined
        assert "--diffusion_samples_affinity 2" in joined
        assert "--affinity_mw_correction" in command
        plain = boltz_predict._command(self.cfg(affinity=False), Path("in.yaml"), Path("out"))
        assert not any("affinity" in part for part in plain)

    def test_the_affinity_file_is_read_with_both_models(self, tmp_path):
        from alambiq.engines import boltz_predict

        path = tmp_path / "affinity_input.json"
        path.write_text(
            json.dumps(
                {
                    "affinity_pred_value": -1.5,
                    "affinity_probability_binary": 0.8,
                    "affinity_pred_value1": -1.2,
                    "affinity_probability_binary1": 0.7,
                    "affinity_pred_value2": -1.8,
                    "affinity_probability_binary2": 0.9,
                }
            )
        )
        found = boltz_predict._affinity(path)
        assert found["log10_ic50_um"] == -1.5 and found["binder_probability"] == 0.8
        assert [m["log10_ic50_um"] for m in found["models"]] == [-1.2, -1.8]

    def test_a_probability_outside_zero_to_one_is_refused(self, tmp_path):
        from alambiq.engines import boltz_predict

        path = tmp_path / "affinity_input.json"
        path.write_text(
            json.dumps({"affinity_pred_value": 0.1, "affinity_probability_binary": 1.2})
        )
        with pytest.raises(ValueError, match="outside 0 to 1"):
            boltz_predict._affinity(path)

    def test_a_ligand_above_128_atoms_is_refused_before_the_run(self):
        pytest.importorskip("rdkit")
        from alambiq.engines import boltz_predict

        assert boltz_predict._affinity_atoms(ALANINATE) == 6
        with pytest.raises(ValueError, match="at most 128 atoms"):
            boltz_predict._affinity_atoms("C" * 129)


class TestLigandChemistryOnSyntheticPoses:
    """The engine's ligand export, with poses built by hand so the answers are known.

    A pose embedded by RDKit has the handedness of its SMILES; reflecting every x
    coordinate gives the mirror image. The export must keep the bonds either way,
    write the coordinates it was given, and say which handedness the pose has.
    """

    @pytest.fixture(scope="class")
    def synthetic(self, tmp_path_factory):
        pytest.importorskip("rdkit")
        pytest.importorskip("gemmi")
        import pickle

        from rdkit import Chem
        from rdkit.Chem import AllChem

        directory = tmp_path_factory.mktemp("ligand")
        molecule = Chem.MolFromSmiles(ALANINATE)
        for atom in molecule.GetAtoms():
            atom.SetProp("name", f"{atom.GetSymbol()}{atom.GetIdx() + 1}")
        molecule = Chem.AddHs(molecule, addCoords=False)
        heavy = Chem.RemoveHs(molecule)
        for atom in heavy.GetAtoms():
            atom.SetProp("name", f"{atom.GetSymbol()}{atom.GetIdx() + 1}")
        assert AllChem.EmbedMolecule(heavy, randomSeed=7) == 0
        native = heavy.GetConformer().GetPositions()
        native_file = directory / "input.pkl"
        template = Chem.Mol(heavy)
        template.RemoveAllConformers()
        # Boltz pickles its molecules with their properties, which hold the atom names.
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        native_file.write_bytes(pickle.dumps({"LIG1": template}))
        return directory, heavy, native, native_file

    def write_model(self, directory, heavy, xyz, name):
        rows = [
            "data_pose",
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.pdbx_PDB_model_num",
        ]
        for atom in heavy.GetAtoms():
            x, y, z = xyz[atom.GetIdx()]
            rows.append(
                f"HETATM {atom.GetIdx() + 1} {atom.GetSymbol()} {atom.GetProp('name')} . "
                f"LIG1 L {x:.4f} {y:.4f} {z:.4f} 1"
            )
        path = directory / name
        path.write_text("\n".join(rows) + "\n")
        return path

    def export(self, synthetic, xyz, name):
        from alambiq.engines import boltz_predict

        directory, heavy, _, native_file = synthetic
        chemistry = boltz_predict._chemistry(
            native_file, {"id": "L", "smiles": ALANINATE}
        )
        model = self.write_model(directory, heavy, xyz, name + ".cif")
        sdf = directory / (name + ".sdf")
        return chemistry, boltz_predict._positioned(chemistry, model, sdf), sdf

    def test_the_native_handedness_is_reported_as_matching(self, synthetic):
        chemistry, pose, _ = self.export(synthetic, synthetic[2], "native")
        assert chemistry["formal_charge"] == -1 and len(chemistry["atoms"]) == 6
        assert pose["stereochemistry_matches_input"] is True
        assert pose["perceived_smiles"] == chemistry["canonical_isomeric_smiles"]

    def test_the_mirror_image_keeps_its_bonds_but_is_reported_as_not_matching(
        self, synthetic
    ):
        import numpy as np
        from rdkit import Chem

        mirrored = synthetic[2] * np.array([-1.0, 1.0, 1.0])
        chemistry, pose, sdf = self.export(synthetic, mirrored, "mirror")
        assert pose["stereochemistry_matches_input"] is False
        assert pose["perceived_smiles"] != chemistry["canonical_isomeric_smiles"]
        # The enantiomer's SMILES is the input's with every tetrahedral tag reversed.
        enantiomer = Chem.MolToSmiles(Chem.MolFromSmiles("C[C@@H](N)C(=O)[O-]"))
        assert pose["perceived_smiles"] == enantiomer
        posed = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
        assert Chem.MolToSmiles(posed, isomericSmiles=False) == Chem.MolToSmiles(
            Chem.MolFromSmiles(ALANINATE), isomericSmiles=False
        )

    def test_the_sdf_carries_exactly_the_coordinates_given(self, synthetic):
        import numpy as np
        from rdkit import Chem

        shifted = synthetic[2] + np.array([1500.25, -900.5, 2000.75])
        _, pose, sdf = self.export(synthetic, shifted, "shifted")
        posed = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
        found = posed.GetConformer().GetPositions()
        assert found == pytest.approx(shifted, abs=1e-3)
        assert [a["name"] for a in pose["atoms"]] == [
            f"{a.GetSymbol()}{a.GetIdx() + 1}" for a in posed.GetAtoms()
        ]

    def test_a_pose_with_other_atom_names_is_refused(self, synthetic):
        from alambiq.engines import boltz_predict

        directory, heavy, native, native_file = synthetic
        chemistry = boltz_predict._chemistry(
            native_file, {"id": "L", "smiles": ALANINATE}
        )
        renamed = chemistry | {
            "atoms": [{**a, "name": "Z" + a["name"]} for a in chemistry["atoms"]]
        }
        model = self.write_model(directory, heavy, native, "names.cif")
        with pytest.raises(ValueError, match="atom names of the predicted ligand differ"):
            boltz_predict._positioned(renamed, model, directory / "names.sdf")

    def test_processed_chemistry_that_differs_from_the_smiles_is_refused(
        self, synthetic
    ):
        from alambiq.engines import boltz_predict

        with pytest.raises(ValueError, match="differs from the input SMILES"):
            boltz_predict._chemistry(synthetic[3], {"id": "L", "smiles": "CCO"})

    def test_a_pickle_with_other_molecules_is_refused(self, synthetic, tmp_path):
        import pickle

        from alambiq.engines import boltz_predict

        other = tmp_path / "other.pkl"
        other.write_bytes(pickle.dumps({"LIG1": None, "LIG2": None}))
        with pytest.raises(ValueError, match="Unexpected Boltz extra-molecule"):
            boltz_predict._chemistry(other, {"id": "L", "smiles": ALANINATE})


pytestmark_engine = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_BOLTZ to run Boltz-2 on a tiny case",
)


def run(params, tmp_path, name="run"):
    from alambiq.api import run as execute

    step = {
        "id": "fold",
        "capability": "structure.predict_biomolecular_complex",
        "params": params,
    }
    return execute({"id": "fold", "steps": [step]}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("fold")


def request(**changes):
    return {
        "proteins": [{"id": "A", "sequence": SEQUENCE}],
        "ligand": {"id": "L", "smiles": ALANINATE},
        "threads": 4,
        **TINY,
        **changes,
    }


@pytestmark_engine
class TestEngineGuards:
    def test_a_gpu_is_used_when_there_is_one_and_refused_not_replaced_when_there_is_not(
        self, tmp_path
    ):
        import subprocess

        has_cuda = (
            subprocess.run(
                [
                    os.environ["ALAMBIQ_ENV_BOLTZ"],
                    "-I",
                    "-c",
                    "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)",
                ]
            ).returncode
            == 0
        )
        outcome = run(request(accelerator="gpu"), tmp_path)
        if has_cuda:
            assert outcome.ok, outcome.steps["fold"].error
        else:
            assert not outcome.ok
            assert "CUDA is unavailable" in outcome.steps["fold"].error

    def test_the_apple_gpu_runs_in_the_patched_environment(self):
        from alambiq.capabilities.characterize.structure_predict_biomolecular_complex import (
            PredictRequest,
            ProteinChain,
            environment_for,
        )

        mps = PredictRequest(proteins=[ProteinChain("A", SEQUENCE)], accelerator="mps")
        cpu = PredictRequest(proteins=[ProteinChain("A", SEQUENCE)])
        assert environment_for(mps) == "boltz_mps" and environment_for(cpu) == "boltz"

    def test_an_alignment_that_does_not_start_with_the_sequence_stops_the_run(
        self, tmp_path
    ):
        bad = tmp_path / "bad.a3m"
        bad.write_text(">q\nACDEFGHIKLMNPQRSTVWY\n")
        outcome = run(
            request(proteins=[{"id": "A", "sequence": SEQUENCE, "msa": str(bad)}]),
            tmp_path,
        )
        assert not outcome.ok
        assert "first A3M row must be exactly" in outcome.steps["fold"].error


@pytestmark_engine
class TestTinyComplex:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_BOLTZ"] = ENV
        return run(
            request(diffusion_samples=2, seed=3), tmp_path_factory.mktemp("fold")
        )

    def test_there_are_as_many_samples_as_requested_each_with_a_structure(
        self, outcome
    ):
        out = ok(outcome)
        assert [s["index"] for s in out["samples"]] == [0, 1]
        for sample in out["samples"]:
            text = Path(sample["structure"]).read_text()
            assert text.startswith("data_") and "_atom_site." in text

    def test_the_chains_in_the_structure_are_the_chains_asked_for(self, outcome):
        import gemmi

        out = ok(outcome)
        structure = gemmi.read_structure(
            out["samples"][0]["structure"], format=gemmi.CoorFormat.Mmcif
        )
        chains = {c.name: [r.name for r in c] for c in structure[0]}
        assert set(chains) == {"A", "L"}
        assert len(chains["A"]) == len(SEQUENCE) == out["chains"][0]["residues"]
        assert chains["L"] == ["LIG1"]

    def test_the_confidence_numbers_lie_in_the_ranges_they_must(self, outcome):
        for sample in ok(outcome)["samples"]:
            for key in (
                "confidence_score",
                "ptm",
                "iptm",
                "ligand_iptm",
                "protein_iptm",
                "complex_plddt",
                "complex_iplddt",
            ):
                assert 0.0 <= sample[key] <= 1.0, key
            assert sample["complex_pde"] >= 0 and sample["complex_ipde"] >= 0
            assert set(sample["chains_ptm"]) == {"0", "1"}
            assert set(sample["arrays"]) == {"pae", "pde", "plddt"}
            assert json.loads(Path(sample["confidence_file"]).read_text())[
                "confidence_score"
            ] == pytest.approx(sample["confidence_score"])

    def test_the_input_boltz_read_is_the_request(self, outcome):
        import yaml

        text = Path(ok(outcome)["input_yaml"]).read_text()
        assert yaml.safe_load(text) == {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": SEQUENCE, "msa": "empty"}},
                {"ligand": {"id": "L", "smiles": ALANINATE}},
            ],
        }

    def test_the_ligand_keeps_its_bonds_charge_and_atom_count_and_its_stereo_flag_is_honest(
        self, outcome
    ):
        Chem = pytest.importorskip("rdkit.Chem")

        out = ok(outcome)
        ligand = out["ligand"]
        expected = Chem.MolFromSmiles(ALANINATE)
        assert ligand["canonical_smiles"] == Chem.MolToSmiles(expected)
        assert ligand["formal_charge"] == -1
        assert ligand["atoms"] == expected.GetNumHeavyAtoms() == 6
        plain = Chem.MolToSmiles(expected, isomericSmiles=False)
        for sample in out["samples"]:
            posed = Chem.SDMolSupplier(sample["ligand_pose"], removeHs=False)[0]
            assert Chem.MolToSmiles(posed, isomericSmiles=False) == plain
            # The flag is what RDKit perceives from the pose, whatever the model drew.
            same = Chem.MolToSmiles(posed) == Chem.MolToSmiles(expected)
            assert sample["ligand_stereochemistry_matches"] is same

    def test_the_ligand_sdf_has_the_coordinates_of_the_same_named_atoms(self, outcome):
        import gemmi

        Chem = pytest.importorskip("rdkit.Chem")

        for sample in ok(outcome)["samples"]:
            block = gemmi.cif.read_file(sample["structure"]).sole_block()
            category = block.get_mmcif_category("_atom_site.")
            rows = [
                dict(zip(category, row, strict=True))
                for row in zip(*category.values(), strict=True)
            ]
            by_name = {
                r["label_atom_id"]: [float(r[f"Cartn_{a}"]) for a in "xyz"]
                for r in rows
                if r["label_asym_id"] == "L"
            }
            posed = Chem.SDMolSupplier(sample["ligand_pose"], removeHs=False)[0]
            mapping = json.loads(posed.GetProp("ALAMBIQ_ATOM_MAP"))
            conformer = posed.GetConformer()
            assert len(mapping) == posed.GetNumAtoms() == len(by_name)
            for atom in mapping:
                x, y, z = conformer.GetAtomPosition(atom["index"])
                assert [x, y, z] == pytest.approx(
                    by_name[atom["name"]], rel=1e-5, abs=1e-3
                )
                assert (
                    posed.GetAtomWithIdx(atom["index"]).GetSymbol().upper()
                    == atom["element"].upper()
                )

    def test_the_chemistry_file_records_the_native_atom_names(self, outcome):
        chemistry = json.loads(Path(ok(outcome)["ligand_chemistry"]).read_text())
        names = [a["name"] for a in chemistry["atoms"]]
        assert len(names) == len(set(names)) == 6
        assert chemistry["component_id"] == "LIG1" and "V3000" in chemistry["molblock"]

    def test_the_notes_say_the_run_is_below_the_defaults_and_ran_without_an_alignment(
        self, outcome
    ):
        notes = " ".join(outcome.steps["fold"].notes)
        assert (
            "below the default settings" in notes
            and "ran from its sequence alone" in notes
        )

    def test_the_versions_and_model_are_recorded(self, outcome):
        out = ok(outcome)
        assert out["versions"]["boltz"] == "2.2.1"
        assert out["model"] == "Boltz-2 2.2.1" and out["accelerator"] == "cpu"
        assert out["seed"] == 3 and out["sampling_steps"] == 2


@pytestmark_engine
class TestSeedsAndAlignments:
    def test_the_same_seed_gives_the_same_sample_and_another_seed_a_different_one(
        self, tmp_path
    ):
        os.environ["ALAMBIQ_ENV_BOLTZ"] = ENV
        first = ok(run(request(seed=1, threads=1), tmp_path, "a"))
        again = ok(run(request(seed=1, threads=1), tmp_path, "b"))
        other = ok(run(request(seed=2, threads=1), tmp_path, "c"))

        def digest(out):
            return hashlib.sha256(
                Path(out["samples"][0]["structure"]).read_bytes()
            ).hexdigest()

        assert digest(first) == digest(again)
        assert digest(first) != digest(other)

    def test_affinity_returns_an_ensemble_and_two_models_with_probabilities(
        self, tmp_path
    ):
        os.environ["ALAMBIQ_ENV_BOLTZ"] = ENV
        outcome = run(
            request(affinity=True, affinity_sampling_steps=2, affinity_samples=1),
            tmp_path,
        )
        out = ok(outcome)
        # Boltz neutralizes the alaninate for affinity, and the result says so.
        assert out["ligand"]["formal_charge"] == 0
        assert "standardized the ligand" in " ".join(outcome.steps["fold"].notes)
        affinity = out["affinity"]
        assert len(affinity["models"]) == 2
        for estimate in [affinity, *affinity["models"]]:
            assert 0.0 <= estimate["binder_probability"] <= 1.0
            assert isinstance(estimate["log10_ic50_um"], float)
        assert affinity["ligand_atoms"] == 6 and affinity["samples"] == 1
        written = json.loads(Path(affinity["file"]).read_text())
        assert written["affinity_pred_value"] == pytest.approx(affinity["log10_ic50_um"])
        import yaml

        given = yaml.safe_load(Path(out["input_yaml"]).read_text())
        assert given["properties"] == [{"affinity": {"binder": "L"}}]

    def test_a_protein_alone_runs_without_ligand_files(self, tmp_path):
        os.environ["ALAMBIQ_ENV_BOLTZ"] = ENV
        out = ok(run(request(ligand=None), tmp_path))
        assert out["ligand"] is None and out["ligand_chemistry"] is None
        assert out["samples"][0]["ligand_pose"] is None
        assert (
            out["samples"][0]["ligand_iptm"] in (None, 0.0)
            or out["samples"][0]["iptm"] is not None
        )

    def test_an_alignment_is_normalised_counted_and_handed_to_boltz(self, tmp_path):
        os.environ["ALAMBIQ_ENV_BOLTZ"] = ENV
        mutant = SEQUENCE[:5] + "A" + SEQUENCE[6:]
        a3m = tmp_path / "wrapped.a3m"
        a3m.write_text(
            f">query\n{SEQUENCE[:10]}\n{SEQUENCE[10:]}\n>homolog\n{mutant}\n>gapped\n{SEQUENCE[:3]}---{SEQUENCE[6:]}\n"
        )
        outcome = run(
            request(
                ligand=None,
                proteins=[{"id": "A", "sequence": SEQUENCE.lower(), "msa": str(a3m)}],
                max_msa_seqs=2,
                num_subsampled_msa=2,
            ),
            tmp_path,
        )
        out = ok(outcome)
        assert out["chains"][0]["alignment"] == "3 A3M rows"
        import yaml

        given = yaml.safe_load(Path(out["input_yaml"]).read_text())["sequences"][0][
            "protein"
        ]
        assert given["sequence"] == SEQUENCE
        assert given["msa"].endswith("chain-00.a3m")
        assert "ran from its sequence alone" not in " ".join(outcome.steps["fold"].notes)


@pytest.mark.skipif(
    not (ENV_MPS and Path(ENV_MPS).is_file()),
    reason="set ALAMBIQ_ENV_BOLTZ_MPS to run the Apple GPU case",
)
class TestAppleGpu:
    def test_a_tiny_prediction_runs_on_the_apple_gpu(self, tmp_path):
        os.environ["ALAMBIQ_ENV_BOLTZ_MPS"] = ENV_MPS
        os.environ["ALAMBIQ_ENV_BOLTZ"] = ENV or ENV_MPS
        out = ok(run(request(accelerator="mps", ligand=None), tmp_path))
        assert out["accelerator"] == "mps"
        assert 0.0 <= out["samples"][0]["confidence_score"] <= 1.0
