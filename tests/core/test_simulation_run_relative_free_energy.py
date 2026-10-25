"""simulation.run_relative_free_energy: the checks, and real OpenFE on transformations with a known answer.

The checks on ligands and mappings are exact chemistry and are tested on small
molecules by hand. The sampling is tested on the one transformation whose answer is
known without a model of the physics: ligand A to an identical copy of itself. The
two end states have the same Hamiltonian, so every state has the same energy in every
configuration, MBAR returns exactly zero with a vanishing error, the overlap between any
two states is 1/K for K states, and the effective sample size of every state is the full
sample count. These hold for the solvent leg and for the protein-bound leg, so they
test the whole pipeline (parameterisation, solvation, hybrid topology, replica exchange,
MBAR, gathering) for plumbing errors, though not the force field.

The real runs use 5 windows and 2 iterations of 0.02 ps: they test the machinery and
nothing about convergence, which the capability itself says. Set ALAMBIQ_ENV_OPENFE to
the pinned OpenFE environment's Python to run them.
"""

from __future__ import annotations

import importlib
import json
import os
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem  # noqa: E402

from alambiq.capabilities.investigate.science import rbfe  # noqa: E402

ENV = os.environ.get("ALAMBIQ_ENV_OPENFE", "")
ENGINES = Path(__file__).resolve().parents[2] / "src/alambiq/engines"


def molecule(smiles: str, seed: int = 7, charges=None, hydrogens: bool = True):
    mol = (
        Chem.AddHs(Chem.MolFromSmiles(smiles))
        if hydrogens
        else Chem.MolFromSmiles(smiles)
    )
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    if charges is None:
        # Alternating charges that sum to zero and are not all zero.
        charges = [0.1 if i % 2 else -0.1 for i in range(mol.GetNumAtoms())]
        charges[-1] -= sum(charges)
    mol.SetProp("atom.dprop.PartialCharge", " ".join(f"{c:.6f}" for c in charges))
    return mol


def identity_pairs(mol) -> list[list[int]]:
    return [[i, i] for i in range(mol.GetNumAtoms())]


SETTINGS = {
    "replicas": 3,
    "lambda_windows": 11,
    "threads": 8,
    "timeout_seconds": 3600,
    "temperature_kelvin": 298.15,
    "pressure_bar": 1.0,
    "ionic_strength_molar": 0.15,
    "solvent_padding_nm": 1.5,
    "equilibration_ps": 1.0,
    "production_ps": 5.0,
    "time_per_iteration_ps": 1.0,
    "checkpoint_interval_ps": 1.0,
    "positions_interval_ps": 1.0,
    "atom_mapping": [[0, 0]],
}


class TestSettings:
    def test_valid_settings_pass(self):
        rbfe.check_settings(**SETTINGS)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"replicas": 0}, "replicas requires an integer in \\[1, 10\\]"),
            ({"replicas": 11}, "replicas requires"),
            ({"lambda_windows": 4}, "lambda_windows requires"),
            ({"lambda_windows": 42}, "lambda_windows requires"),
            ({"threads": 0}, "threads requires"),
            ({"timeout_seconds": 10}, "timeout_seconds requires"),
            (
                {"temperature_kelvin": 200.0},
                "temperature_kelvin requires a finite value",
            ),
            ({"pressure_bar": 0.0}, "pressure_bar requires"),
            ({"ionic_strength_molar": 1.5}, "ionic_strength_molar requires"),
            ({"solvent_padding_nm": 1.0}, "solvent_padding_nm requires"),
            ({"equilibration_ps": float("nan")}, "equilibration_ps requires"),
            ({"time_per_iteration_ps": 0.021}, "whole 2 fs steps"),
            (
                {"production_ps": 5.5},
                "whole 2 fs steps and whole replica-exchange iterations",
            ),
            ({"checkpoint_interval_ps": 1.5}, "whole 2 fs steps and whole"),
            (
                {"checkpoint_interval_ps": 6.0, "positions_interval_ps": 1.0},
                "fit within production",
            ),
            ({"atom_mapping": []}, "1–500 distinct"),
            ({"atom_mapping": [[0, 0], [0, 1]]}, "distinct zero-based"),
            ({"atom_mapping": [[0, 0], [1, 0]]}, "distinct zero-based"),
            ({"atom_mapping": [[-1, 0]]}, "distinct zero-based"),
        ],
    )
    def test_settings_that_cannot_run_are_refused_by_name(self, changes, message):
        with pytest.raises(ValueError, match=message):
            rbfe.check_settings(**{**SETTINGS, **changes})


class TestLigandChecks:
    def test_a_valid_pair_returns_its_identity(self):
        a = molecule("CCO")
        found = rbfe.validate_ligands(a, a, identity_pairs(a))
        assert (
            found["smiles"] == ["CCO", "CCO"]
            and found["mapped_atoms"] == a.GetNumAtoms()
        )
        assert found["formal_charges"] == [0, 0]

    def test_a_ligand_may_map_part_of_itself_onto_a_larger_one(self):
        a, b = molecule("CCO"), molecule("CCCO")
        # Map the ethanol chain onto the first two carbons and the oxygen of propanol,
        # leaving propanol's third carbon and its hydrogens unmapped.
        pairs = [[0, 0], [1, 1], [2, 3]]
        found = rbfe.validate_ligands(a, b, [[0, 0], [1, 1]])
        assert found["mapped_atoms"] == 2 and pairs

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (
                lambda: molecule("CC[O-]", charges=[0.0] * 9 + [0.0]),
                "connected neutral",
            ),
            (lambda: molecule("C.C"), "connected neutral"),
        ],
    )
    def test_a_charged_or_split_ligand_is_refused(self, build, message):
        a = build()
        with pytest.raises(ValueError, match=message):
            rbfe.validate_ligands(a, a, [[0, 0]])

    def test_a_ligand_without_a_conformer_is_refused(self):
        a = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        with pytest.raises(ValueError, match="one finite 3D conformer"):
            rbfe.validate_ligands(a, a, [[0, 0]])

    def test_a_ligand_with_implicit_hydrogens_is_refused(self):
        a = molecule("CCO", hydrogens=False)
        with pytest.raises(ValueError, match="hydrogens must be explicit"):
            rbfe.validate_ligands(a, a, [[0, 0]])

    def test_unspecified_stereochemistry_is_refused(self):
        a = molecule("CC(F)Cl")
        with pytest.raises(ValueError, match="stereochemistry requires explicit"):
            rbfe.validate_ligands(a, a, [[0, 0]])

    def test_charges_must_be_present_numeric_neutral_and_not_all_zero(self):
        a = molecule("CCO")
        a.ClearProp("atom.dprop.PartialCharge")
        with pytest.raises(ValueError, match="precomputed partial charges"):
            rbfe.validate_ligands(a, a, [[0, 0]])
        a.SetProp("atom.dprop.PartialCharge", "x y z")
        with pytest.raises(ValueError, match="must be numerical"):
            rbfe.validate_ligands(a, a, [[0, 0]])
        a.SetProp("atom.dprop.PartialCharge", " ".join(["0.0"] * a.GetNumAtoms()))
        with pytest.raises(ValueError, match="sum to the neutral formal charge"):
            rbfe.validate_ligands(a, a, [[0, 0]])
        a.SetProp("atom.dprop.PartialCharge", " ".join(["0.1"] * a.GetNumAtoms()))
        with pytest.raises(ValueError, match="sum to the neutral formal charge"):
            rbfe.validate_ligands(a, a, [[0, 0]])

    def test_a_mapped_atom_outside_its_ligand_is_refused(self):
        a = molecule("CCO")
        with pytest.raises(ValueError, match="outside its exact ligand"):
            rbfe.validate_ligands(a, a, [[0, 0], [99, 1]])

    def test_a_mapping_that_is_not_one_to_one_is_refused(self):
        a = molecule("CCO")
        with pytest.raises(ValueError, match="one-to-one"):
            rbfe.validate_ligands(a, a, [[0, 0], [1, 0]])

    def test_a_mapped_pair_must_keep_its_element(self):
        a = molecule("CCO")
        with pytest.raises(
            ValueError, match="preserve element, isotope, aromaticity and formal charge"
        ):
            rbfe.validate_ligands(a, a, [[0, 2]])

    def test_mapped_core_bonds_and_connectivity_must_be_preserved(self):
        a, b = molecule("CCC"), molecule("C=CC")
        with pytest.raises(
            ValueError, match="connectivity and bond orders must be preserved"
        ):
            rbfe.validate_ligands(a, b, [[0, 0], [1, 1]])

    def test_a_ring_touched_by_the_mapping_must_be_mapped_whole(self):
        a = molecule("c1ccccc1C")
        pairs = [[i, i] for i in range(4)]
        with pytest.raises(ValueError, match="Map each participating ring completely"):
            rbfe.validate_ligands(a, a, pairs)

    def test_the_mapped_core_must_be_one_connected_fragment(self):
        a = molecule("CCCC")
        with pytest.raises(ValueError, match="one connected fragment"):
            rbfe.validate_ligands(a, a, [[0, 0], [3, 3]])

    def test_a_stereocentre_must_keep_its_orientation_with_its_whole_neighbourhood(
        self,
    ):
        r = molecule("C[C@H](F)Cl", seed=3)
        s = molecule("C[C@@H](F)Cl", seed=3)
        n = r.GetNumAtoms()
        # All atoms mapped to the same atoms of the enantiomer: the orientation flips.
        with pytest.raises(
            ValueError, match="Mapped stereocenters must preserve spatial orientation"
        ):
            rbfe.validate_ligands(r, s, identity_pairs(r))
        # Leaving a neighbour of the stereocentre out of the mapping is also refused.
        with pytest.raises(
            ValueError, match="complete mapped tetrahedral neighborhood"
        ):
            rbfe.validate_ligands(r, r, [[i, i] for i in range(n) if i != 2])

    def test_the_same_enantiomer_maps_onto_itself(self):
        r = molecule("C[C@H](F)Cl", seed=3)
        assert (
            rbfe.validate_ligands(r, r, identity_pairs(r))["mapped_atoms"]
            == r.GetNumAtoms()
        )


class TestCudaPrecision:
    """The double-precision selection, with stand-ins for OpenMM and OpenFE's platform helper."""

    @pytest.fixture
    def engine(self, monkeypatch):
        monkeypatch.syspath_prepend(str(ENGINES))
        return importlib.import_module("openfe_rbfe")

    class Platform:
        def __init__(self, name="CUDA"):
            self.name = name
            self.defaults = {
                "Precision": "mixed",
                "DeviceIndex": "0",
                "DeterministicForces": "false",
            }

        def getName(self):
            return self.name

        def getPropertyNames(self):
            return [*self.defaults, "Other"]

        def getPropertyDefaultValue(self, key):
            return self.defaults[key]

        def setPropertyDefaultValue(self, key, value):
            self.defaults[key] = value

    def test_the_cpu_needs_no_adapter_and_reports_native_precision(self, engine):
        with engine.qualified_precision("CPU") as receipt:
            assert receipt["precision"] == "native"

    def test_an_unknown_backend_is_refused(self, engine):
        with (
            pytest.raises(ValueError, match="CPU or CUDA"),
            engine.qualified_precision("OpenCL"),
        ):
            pass

    def test_every_selection_is_made_double_and_counted_then_everything_is_restored(
        self, engine
    ):
        platform = self.Platform()
        compute = types.SimpleNamespace(get_openmm_platform=lambda *a, **k: platform)
        original = compute.get_openmm_platform
        with engine.qualified_precision(
            "CUDA", compute=compute, platform=platform
        ) as receipt:
            compute.get_openmm_platform()
            compute.get_openmm_platform()
            assert platform.defaults["Precision"] == "double"
            assert receipt["platform_selections"] == 2
        assert platform.defaults["Precision"] == "mixed"
        assert compute.get_openmm_platform is original

    def test_a_platform_that_is_not_cuda_is_refused(self, engine):
        platform = self.Platform("CPU")
        compute = types.SimpleNamespace(get_openmm_platform=lambda *a, **k: platform)
        with engine.qualified_precision("CUDA", compute=compute, platform=platform):
            with pytest.raises(ValueError, match="needs a CUDA context"):
                compute.get_openmm_platform()


ENV_MD = os.environ.get("ALAMBIQ_ENV_OPENMM", "")
STRUCTURES = Path(__file__).resolve().parents[1] / "fixtures/structures"
TINY = {
    "lambda_windows": 5,
    "equilibration_ps": 0.02,
    "production_ps": 0.04,
    "time_per_iteration_ps": 0.02,
    "checkpoint_interval_ps": 0.02,
    "positions_interval_ps": 0.02,
    "threads": 8,
}

needs_openfe = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file() and ENV_MD and Path(ENV_MD).is_file()),
    reason="set ALAMBIQ_ENV_OPENFE and ALAMBIQ_ENV_OPENMM to run OpenFE on identity transformations",
)


def execute(steps, tmp_path, name="run"):
    from alambiq.api import run as go

    return go({"id": "rbfe", "steps": steps}, run_dir=tmp_path / name)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Trp-cage with hydrogens, and benzene with AshGC charges placed beside it."""
    os.environ["ALAMBIQ_ENV_OPENFE"] = ENV
    os.environ["ALAMBIQ_ENV_OPENMM"] = ENV_MD
    directory = tmp_path_factory.mktemp("rbfe-inputs")
    outcome = execute(
        [
            {
                "id": "protein",
                "capability": "structure.prepare_protein_system",
                "params": {
                    "structure": str(STRUCTURES / "1L2Y.cif"),
                    "chain_id": "A",
                    "solvent": "gbn2",
                },
            },
            {
                "id": "ligands",
                "capability": "structure.prepare_relative_binding_inputs",
                "params": {
                    "structure": str(STRUCTURES / "181L.cif"),
                    "chain_id": "A",
                    "anchor_residue_name": "BNZ",
                    "anchor_atom_names": ["C1", "C2", "C3", "C4", "C5", "C6"],
                    "ligands": [
                        {"id": "benzene", "smiles": "c1ccccc1", "embedding_seed": 1},
                        {"id": "toluene", "smiles": "Cc1ccccc1", "embedding_seed": 2},
                    ],
                    "edges": [{"id": "bt", "a": "benzene", "b": "toluene"}],
                    "complete_terminals": True,
                },
            },
        ],
        directory,
    )
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    protein = Path(outcome.outputs("protein")["topology_pdb"])
    benzene = Path(outcome.outputs("ligands")["ligands"][0]["sdf"])
    # Put the benzene 8 angstrom beyond the protein so the two cannot overlap.
    xyz = []
    for line in protein.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    xyz = np.array(xyz)
    molecule = Chem.SDMolSupplier(str(benzene), removeHs=False)[0]
    positions = molecule.GetConformer().GetPositions()
    target = np.array([xyz[:, 0].max() + 8.0, xyz[:, 1].mean(), xyz[:, 2].mean()])
    shifted = positions - positions.mean(axis=0) + target
    for i, point in enumerate(shifted):
        molecule.GetConformer().SetAtomPosition(i, point.tolist())
    placed = directory / "benzene.sdf"
    with Chem.SDWriter(str(placed)) as writer:
        writer.write(molecule)
    return {
        "protein": str(protein),
        "ligand": str(placed),
        "atoms": molecule.GetNumAtoms(),
    }


def leg(inputs, phase, replicas):
    return {
        "id": phase,
        "capability": "simulation.run_relative_free_energy",
        "params": {
            "phase": phase,
            "edge": {"id": "same", "a": "benzene", "b": "benzene-again"},
            "ligand_a": inputs["ligand"],
            "ligand_b": inputs["ligand"],
            "atom_mapping": [{"a": i, "b": i} for i in range(inputs["atoms"])],
            "replicas": replicas,
            "keep_native_records": phase == "solvent",
            **({"protein": inputs["protein"]} if phase == "bound" else {}),
            **TINY,
        },
    }


@needs_openfe
class TestIdentityTransformations:
    @pytest.fixture(scope="class")
    def outcome(self, inputs, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_OPENFE"] = ENV
        steps = [
            leg(inputs, "solvent", 2),
            leg(inputs, "bound", 2),
            {
                "id": "analysis",
                "capability": "simulation.analyze_binding_free_energy",
                "params": {
                    "title": "benzene to itself",
                    "edges": [{"id": "same", "a": "benzene", "b": "benzene-again"}],
                    "checks": {
                        "maximum_edge_replica_sem_kcal_mol": 0.01,
                        "minimum_adjacent_overlap_each_replica": 0.19,
                    },
                },
                "bind": {
                    "legs.0": "solvent.leg_record",
                    "legs.1": "bound.leg_record",
                    "structural_diagnostics": "bound.structural_diagnostics",
                },
            },
        ]
        return execute(steps, tmp_path_factory.mktemp("legs"))

    def out(self, outcome, step):
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        return outcome.outputs(step)

    @pytest.mark.parametrize("phase", ["solvent", "bound"])
    def test_a_ligand_transformed_into_itself_costs_nothing(self, outcome, phase):
        out = self.out(outcome, phase)
        assert len(out["repeats"]) == 2
        for repeat in out["repeats"]:
            assert abs(repeat["delta_g_kcal_mol"]) < 1e-4
            assert repeat["mbar_error_kcal_mol"] < 1e-4

    @pytest.mark.parametrize("phase", ["solvent", "bound"])
    def test_identical_states_overlap_completely(self, outcome, phase):
        for repeat in self.out(outcome, phase)["repeats"]:
            # Five states with equal energies in every configuration: MBAR's overlap
            # matrix has every entry 1/5, and every state keeps all its samples.
            assert repeat["minimum_adjacent_overlap"] == pytest.approx(0.2, abs=1e-6)
            assert repeat["minimum_effective_samples"] > 1.0

    def test_the_repeats_are_distinct_runs(self, outcome):
        record = json.loads(
            Path(self.out(outcome, "solvent")["leg_record"]).read_text()
        )
        ids = [r["id"] for r in record["replicas"]]
        digests = {r["trajectory_sha256"] for r in record["replicas"]}
        assert len(set(ids)) == 2 and len(digests) == 2

    def test_the_record_carries_the_identities_the_analysis_needs(self, outcome):
        solvent = json.loads(
            Path(self.out(outcome, "solvent")["leg_record"]).read_text()
        )
        bound = json.loads(Path(self.out(outcome, "bound")["leg_record"]).read_text())
        assert solvent["protein_sha256"] is None and len(bound["protein_sha256"]) == 64
        assert solvent["method_sha256"] == bound["method_sha256"]
        assert solvent["mapping_sha256"] == bound["mapping_sha256"]
        assert solvent["ligand_sha256"] == bound["ligand_sha256"]
        assert solvent["temperature_kelvin"] == 298.15

    def test_the_bound_leg_keeps_a_structural_file_for_each_repeat(self, outcome):
        files = self.out(outcome, "bound")["structural_diagnostics"]
        assert len(files) == 2 and all(Path(p).is_file() for p in files.values())

    def test_the_binding_difference_of_the_identity_edge_is_zero(self, outcome):
        (edge,) = self.out(outcome, "analysis")["edges"]
        assert abs(edge["delta_delta_g_kcal_mol"]) < 1e-4
        assert edge["replica_sem_kcal_mol"] < 1e-3

    def test_the_checks_on_the_pair_pass(self, outcome):
        analysis = self.out(outcome, "analysis")
        assert analysis["status"] == "pass"
        assert {c["diagnostic"] for c in analysis["checks"]} >= {
            "edge_replica_sem",
            "overlap",
        }

    def test_the_native_records_are_kept_only_when_asked_for(self, outcome):
        assert self.out(outcome, "bound")["native_records"] is None

    def test_the_native_records_are_kept_and_the_notes_say_the_run_is_short(
        self, outcome
    ):
        out = self.out(outcome, "solvent")
        assert (
            out["native_records"] is not None and Path(out["native_records"]).is_file()
        )
        assert "that this check expects for a converged free energy" in " ".join(
            outcome.steps["solvent"].notes
        )

    def test_a_ligand_that_is_not_charged_by_ashgc_is_refused(self, inputs, tmp_path):
        molecule = Chem.SDMolSupplier(inputs["ligand"], removeHs=False)[0]
        charges = [
            float(x) for x in molecule.GetProp("atom.dprop.PartialCharge").split()
        ]
        charges[0] += 0.01
        charges[1] -= 0.01
        molecule.SetProp(
            "atom.dprop.PartialCharge", " ".join(f"{c:.6f}" for c in charges)
        )
        altered = tmp_path / "altered.sdf"
        with Chem.SDWriter(str(altered)) as writer:
            writer.write(molecule)
        step = leg({**inputs, "ligand": str(altered)}, "solvent", 1)
        outcome = execute([step], tmp_path)
        assert not outcome.ok
        assert (
            "not the AshGC charges of this molecule" in outcome.steps["solvent"].error
        )

    def test_the_phase_and_protein_must_agree(self, inputs, tmp_path):
        no_protein = leg(inputs, "bound", 1)
        del no_protein["params"]["protein"]
        outcome = execute([no_protein], tmp_path)
        assert "the bound phase needs a protein" in outcome.steps["bound"].error
        with_protein = leg(inputs, "solvent", 1)
        with_protein["params"]["protein"] = inputs["protein"]
        outcome = execute([with_protein], tmp_path, "b")
        assert "protein must be left out" in outcome.steps["solvent"].error
