"""structure.prepare_protein_system with the real OpenMM and PDBFixer, on Trp-cage.

Trp-cage (PDB 1L2Y, chain A) is twenty residues, NLYIQWLKDGGPSSGRPPPS. Its first NMR
model carries 304 atoms of which 150 are hydrogens, so 154 are heavy. At pH 7 its
side chains and termini give it a net charge of +1: Lys8 and Arg16 and the amino
terminus are positive and Asp9 and the carboxy terminus negative. Every count below
follows from that.

Set ALAMBIQ_ENV_OPENMM to the pinned openmm-ensemble environment's Python.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

ENV = os.environ.get("ALAMBIQ_ENV_OPENMM", "")
STRUCTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/1L2Y.cif"
HEMOGLOBIN = Path(__file__).resolve().parents[1] / "fixtures/structures/1A3N.cif"
SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"

pytestmark = pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_OPENMM to run the OpenMM tests",
)


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    if ENV:
        monkeypatch.setenv("ALAMBIQ_ENV_OPENMM", ENV)


def prepare(tmp_path: Path, name: str = "run", **params):
    from alambiq.api import run

    step = {
        "id": "prep",
        "capability": "structure.prepare_protein_system",
        "params": {"structure": str(STRUCTURE), "chain_id": "A", **params},
    }
    return run({"id": "p", "steps": [step]}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("prep")


def charges(path: str) -> list[float]:
    return [
        float(q)
        for q in re.findall(r'<Particle[^>]*\bq="([^"]+)"', Path(path).read_text())
    ]


def source_atoms() -> tuple[int, int]:
    """Heavy and hydrogen atoms of the first model of the source file."""
    import gemmi

    model = gemmi.read_structure(str(STRUCTURE))[0]
    atoms = [a for chain in model for residue in chain for a in residue]
    hydrogens = sum(a.element.name == "H" for a in atoms)
    return len(atoms) - hydrogens, hydrogens


@pytest.fixture(scope="module")
def implicit(tmp_path_factory):
    os.environ["ALAMBIQ_ENV_OPENMM"] = ENV
    tmp_path = tmp_path_factory.mktemp("implicit")
    outcome = prepare(tmp_path, solvent="gbn2")
    return outcome, ok(outcome)


@pytest.fixture(scope="module")
def explicit(tmp_path_factory):
    os.environ["ALAMBIQ_ENV_OPENMM"] = ENV
    tmp_path = tmp_path_factory.mktemp("explicit")
    outcome = prepare(tmp_path, solvent="tip3p", padding_nm=1.0)
    return outcome, ok(outcome)


class TestImplicit:
    def test_the_twenty_residues_keep_their_sequence_and_their_hydrogens_are_replaced(
        self, implicit
    ):
        _, out = implicit
        assert out["protein_residues"] == 20
        assert Path(out["sequence_fasta"]).read_text().split()[1] == SEQUENCE
        heavy, hydrogens = source_atoms()
        assert (
            out["removed_source_atoms"] == {"hydrogen": hydrogens} == {"hydrogen": 150}
        )
        assert out["protein_atoms"] == out["atom_count"] == 304
        assert heavy == 154

    def test_every_heavy_atom_of_the_source_is_mapped_to_a_prepared_atom(
        self, implicit
    ):
        _, out = implicit
        manifest = json.loads(Path(out["manifest"]).read_text())
        assert len(manifest["heavy_atom_mapping"]) == 154
        assert (
            manifest["added_heavy_atoms"] == []
            and manifest["heavy_atom_completion"] is None
        )

    def test_minimisation_lowered_the_energy_and_the_forces_are_finite(self, implicit):
        _, out = implicit
        assert (
            out["potential_energy_after_kj_mol"] < out["potential_energy_before_kj_mol"]
        )
        assert (
            0
            < out["rms_force_after_kj_mol_nm"]
            < out["max_force_after_kj_mol_nm"]
            < 1e6
        )

    def test_there_is_no_box_and_the_solvent_is_recorded_as_implicit(self, implicit):
        _, out = implicit
        assert out["box_volume_nm3"] is None
        manifest = json.loads(Path(out["manifest"]).read_text())
        assert manifest["water_model"] == "implicit-GBn2/ACE"
        assert manifest["nonbonded_method"] == "NoCutoff"
        assert manifest["environment"]["solvent_model"] == "gbn2"

    def test_the_geometry_is_described_and_no_residue_is_inverted(self, implicit):
        _, out = implicit
        assert out["inverted_alpha_residues"] == []
        assert 0 <= out["max_peptide_planarity_deviation_degrees"] < 90

    def test_the_prepared_files_agree_with_each_other(self, implicit):
        _, out = implicit
        particles = len(charges(out["system_xml"]))
        assert particles == out["atom_count"]
        assert Path(out["topology_pdb"]).read_text().count("ATOM") == out["atom_count"]

    def test_the_versions_and_force_field_are_recorded(self, implicit):
        _, out = implicit
        assert (
            out["versions"]["openmm"].startswith("8.")
            and out["versions"]["force_field"] == "amber-ff14SB"
        )


class TestExplicit:
    def test_the_system_is_electrically_neutral_with_the_counter_ions_the_charge_needs(
        self, explicit
    ):
        _, out = explicit
        assert sum(charges(out["system_xml"])) == pytest.approx(0, abs=1e-4)
        counts = out["residue_counts"]
        # Net protein charge +1, so one more chloride than sodium; the rest is added salt.
        assert counts["CL"] - counts["NA"] == 1

    def test_the_atoms_are_the_protein_plus_three_per_water_plus_the_ions(
        self, explicit
    ):
        _, out = explicit
        counts = out["residue_counts"]
        assert (
            out["atom_count"]
            == out["protein_atoms"] + 3 * counts["HOH"] + counts["NA"] + counts["CL"]
        )
        assert out["protein_atoms"] == 304

    def test_the_box_is_a_dodecahedron_and_it_holds_the_water(self, explicit):
        _, out = explicit
        manifest = json.loads(Path(out["manifest"]).read_text())
        vectors = manifest["periodic_cell"]["vectors_nm"]
        assert (
            manifest["box_shape"] == "dodecahedron"
            and manifest["water_model"] == "tip3p"
        )
        assert vectors[1][0] != 0 or vectors[2][0] != 0  # not an orthogonal box
        assert out["box_volume_nm3"] == pytest.approx(
            manifest["periodic_cell"]["volume_nm3"]
        )
        # Water is roughly 33 molecules per nm3.
        assert 25 < out["residue_counts"]["HOH"] / out["box_volume_nm3"] < 36

    def test_a_cube_with_more_padding_is_a_bigger_box(self, explicit, tmp_path):
        _, small = explicit
        cube = ok(prepare(tmp_path, solvent="tip3p", padding_nm=1.5, box_shape="cube"))
        manifest = json.loads(Path(cube["manifest"]).read_text())
        vectors = manifest["periodic_cell"]["vectors_nm"]
        assert vectors[0][1] == vectors[0][2] == vectors[1][0] == 0
        assert cube["box_volume_nm3"] > small["box_volume_nm3"]

    def test_more_salt_means_more_ion_pairs(self, explicit, tmp_path):
        _, base = explicit
        salty = ok(
            prepare(tmp_path, solvent="tip3p", padding_nm=1.0, ionic_strength_molar=0.6)
        )
        assert salty["residue_counts"]["NA"] > base["residue_counts"]["NA"]
        assert salty["residue_counts"]["CL"] - salty["residue_counts"]["NA"] == 1


class TestProtonationAndSubstitution:
    def test_protonating_the_aspartate_makes_the_protein_charge_plus_two(
        self, tmp_path
    ):
        out = ok(
            prepare(
                tmp_path,
                solvent="tip3p",
                padding_nm=1.0,
                residue_variants=[{"residue": "9", "variant": "ASH"}],
            )
        )
        counts = out["residue_counts"]
        assert counts["CL"] - counts["NA"] == 2
        assert sum(charges(out["system_xml"])) == pytest.approx(0, abs=1e-4)

    def test_a_substitution_rebuilds_the_residue_from_its_template_and_changes_the_sequence(
        self, tmp_path
    ):
        out = ok(
            prepare(
                tmp_path,
                solvent="gbn2",
                heavy_atom_completion="pdbfixer-standard",
                substitutions=[{"residue": "3", "wild_type": "Y", "mutant": "A"}],
            )
        )
        assert (
            Path(out["sequence_fasta"]).read_text().split()[1] == "NLAIQWLKDGGPSSGRPPPS"
        )
        assert out["substitutions_applied"] == 1
        # Tyrosine to alanine drops CG, CD1, CD2, CE1, CE2, CZ and OH: seven heavy atoms.
        assert out["removed_source_atoms"]["substitution-heavy-atom"] == 7
        manifest = json.loads(Path(out["manifest"]).read_text())
        assert manifest["heavy_atom_completion"]["provider"] == "pdbfixer"
        assert manifest["heavy_atom_completion"]["version"].startswith("1.")

    def test_completion_alone_adds_nothing_when_nothing_is_missing(self, tmp_path):
        out = ok(
            prepare(tmp_path, solvent="gbn2", heavy_atom_completion="pdbfixer-standard")
        )
        assert out["added_heavy_atoms"] == 0 and out["substitutions_applied"] == 0

    def test_a_substitution_names_the_wild_type_it_expects(self, tmp_path):
        outcome = prepare(
            tmp_path,
            solvent="gbn2",
            heavy_atom_completion="pdbfixer-standard",
            substitutions=[{"residue": "3", "wild_type": "F", "mutant": "A"}],
        )
        assert (
            "wild_type differs from the selected source residue"
            in outcome.steps["prep"].error
        )


class TestEveryOption:
    def test_a_pdb_file_with_a_stated_format_prepares_the_same_protein(self, tmp_path):
        import gemmi

        model = gemmi.read_structure(str(STRUCTURE))
        pdb = tmp_path / "structure.dat"
        pdb.write_text(model.make_pdb_string())
        out = ok(
            prepare(
                tmp_path, solvent="gbn2", structure=str(pdb), structure_format="pdb"
            )
        )
        assert Path(out["sequence_fasta"]).read_text().split()[1] == SEQUENCE
        assert out["protein_atoms"] == 304

    def test_a_higher_solute_dielectric_changes_the_implicit_energy(
        self, implicit, tmp_path
    ):
        _, base = implicit
        other = ok(prepare(tmp_path, solvent="gbn2", solute_dielectric=4.0))
        assert other["potential_energy_after_kj_mol"] != pytest.approx(
            base["potential_energy_after_kj_mol"], abs=1.0
        )

    def test_a_looser_force_tolerance_stops_minimisation_earlier(
        self, implicit, tmp_path
    ):
        _, tight = implicit
        loose = ok(
            prepare(tmp_path, solvent="gbn2", minimization_tolerance_kj_mol_nm=1000.0)
        )
        assert (
            loose["potential_energy_after_kj_mol"]
            > tight["potential_energy_after_kj_mol"]
        )

    def test_the_platform_properties_are_passed_to_openmm_and_recorded(self, tmp_path):
        out = ok(
            prepare(tmp_path, solvent="gbn2", platform_properties={"Threads": "2"})
        )
        manifest = json.loads(Path(out["manifest"]).read_text())
        assert manifest["platform_properties"]["Threads"] == "2"

    def test_other_components_in_the_file_are_refused_or_dropped_as_the_policy_says(
        self, tmp_path
    ):
        import gemmi

        model = gemmi.read_structure(str(STRUCTURE))
        model.remove_hydrogens()
        while len(model) > 1:
            del model[len(model) - 1]
        water = "HETATM 9001  O   HOH B 101       0.000   0.000   0.000  1.00  0.00           O\n"
        text = model.make_pdb_string()
        lines = text.splitlines(keepends=True)
        last = max(
            i for i, line in enumerate(lines) if line.startswith(("ATOM", "TER"))
        )
        pdb = tmp_path / "with-water.pdb"
        pdb.write_text("".join(lines[: last + 1]) + water + "".join(lines[last + 1 :]))
        refused = prepare(tmp_path, solvent="gbn2", structure=str(pdb))
        assert "would discard source components" in refused.steps["prep"].error
        out = ok(
            prepare(
                tmp_path,
                "b",
                solvent="gbn2",
                structure=str(pdb),
                source_component_policy="selected-chain-only",
            )
        )
        manifest = json.loads(Path(out["manifest"]).read_text())
        assert [
            r["residue_name"] for r in manifest["source_selection"]["excluded_residues"]
        ] == ["HOH"]


class TestRefusals:
    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"chain_id": "Z"}, "exactly one"),
            ({"chain_id": " "}, "exact chain_id"),
            (
                {"solvent": "tip3p", "padding_nm": 5.0},
                "padding_nm is 5.0; a value from 1 to 3 is required",
            ),
            ({"ph": 15}, "ph is 15; a value from 0 to 14 is required"),
            ({"ionic_strength_molar": 2}, "ionic_strength_molar"),
            (
                {"model_index": 1000},
                "model_index is 1000; an integer from 0 to 999 is required",
            ),
            ({"model_index": 38}, "unavailable"),
            ({"seed": 0}, "seed is 0; an integer from 1 to 2147483647 is required"),
            ({"minimization_max_iterations": 0}, "minimization_max_iterations"),
            ({"solvent": "gbn2", "solvent_dielectric": 5}, "solvent_dielectric"),
            ({"solvent": "gbn2", "temperature_kelvin": 900}, "temperature_kelvin"),
            ({"solvent": "vacuum"}, "expected one of"),
            (
                {"substitutions": [{"residue": "3", "wild_type": "Y", "mutant": "A"}]},
                "heavy_atom_completion",
            ),
            (
                {
                    "residue_variants": [
                        {"residue": "9", "variant": "ASH"},
                        {"residue": "9", "variant": "ASP"},
                    ]
                },
                "unique source residues",
            ),
            (
                {"residue_variants": [{"residue": "9", "variant": "XXX"}]},
                "expected one of",
            ),
            (
                {
                    "heavy_atom_completion": "pdbfixer-standard",
                    "substitutions": [
                        {"residue": "3", "wild_type": "Y", "mutant": "Y"}
                    ],
                },
                "distinct standard",
            ),
        ],
    )
    def test_a_setup_that_cannot_be_meant_is_refused_in_plain_words(
        self, tmp_path, changes, message
    ):
        outcome = prepare(tmp_path, **{"solvent": "gbn2", **changes})
        assert outcome.steps["prep"].status == "failed"
        assert message in outcome.steps["prep"].error

    def test_a_file_whose_format_is_unclear_must_say_so(self, tmp_path):
        odd = tmp_path / "structure.dat"
        odd.write_text(STRUCTURE.read_text())
        outcome = prepare(tmp_path, solvent="gbn2", structure=str(odd))
        assert "set structure_format" in outcome.steps["prep"].error

    def test_a_chain_holding_a_nonstandard_residue_is_not_a_standard_protein_chain(
        self, tmp_path
    ):
        import gemmi

        model = gemmi.read_structure(str(STRUCTURE))
        model.remove_hydrogens()
        pdb = tmp_path / "modified.pdb"
        pdb.write_text(model.make_pdb_string().replace("GLN", "MSE"))
        outcome = prepare(tmp_path, solvent="gbn2", structure=str(pdb))
        assert outcome.steps["prep"].status == "failed"
        assert "standard protein residues" in outcome.steps["prep"].error

    def test_a_structure_with_alternate_locations_asks_for_a_choice_first(
        self, tmp_path
    ):
        if not HEMOGLOBIN.is_file():
            pytest.skip("the hemoglobin structure is not in this checkout")
        outcome = prepare(tmp_path, solvent="gbn2", structure=str(HEMOGLOBIN))
        assert (
            "Alternate atom locations require explicit upstream selection"
            in outcome.steps["prep"].error
        )

    def test_the_same_preparation_is_reused_and_a_different_ph_runs_again(
        self, tmp_path
    ):
        first = prepare(tmp_path, "same", solvent="gbn2")
        second = prepare(tmp_path, "same", solvent="gbn2")
        other = prepare(tmp_path, "same", solvent="gbn2", ph=5.0)
        assert (first.steps["prep"].status, second.steps["prep"].status) == (
            "completed",
            "reused",
        )
        assert other.steps["prep"].status == "completed"
