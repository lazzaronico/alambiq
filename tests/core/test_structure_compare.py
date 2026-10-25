"""structure.compare on four alpha carbons whose superposition is known exactly.

The reference chain has alpha carbons at (0,0,0), (2,0,0), (0,3,0) and (0,0,4). The
mobile chain is the same four points rotated a quarter turn about z and shifted by
(10, -3, 5): a rigid motion, so the best fit has zero RMSD, and applying the fit
returns the mobile atoms to the reference points.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("gemmi")

NAMES = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE"}
POINTS = [(0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 4)]
MOVED = [(-y + 10, x - 3, z + 5) for x, y, z in POINTS]


def cif(points, sequence="ACDE", model=1, missing=(), extra="") -> str:
    text = f"""data_test
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.pdbx_seq_one_letter_code_can
1 'polypeptide(L)' {sequence}
loop_
_struct_asym.id
_struct_asym.entity_id
long_chain 1
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""
    for i, (aa, (x, y, z)) in enumerate(zip(sequence, points, strict=True), 1):
        if i not in missing:
            text += f"ATOM {i} C CA . {NAMES[aa]} long_chain 1 {i} . {x} {y} {z} {i + 40} author {model}\n"
    return text + extra


def inspect_step(
    step_id: str, directory: Path, points, sequence="ACDE", **kwargs
) -> dict:
    (directory / f"{step_id}.faa").write_text(f">{step_id}\n{sequence}\n")
    (directory / f"{step_id}.cif").write_text(cif(points, sequence, **kwargs))
    return {
        "id": step_id,
        "capability": "protein.inspect",
        "params": {
            "fasta": str(directory / f"{step_id}.faa"),
            "structure": str(directory / f"{step_id}.cif"),
            "model_number": kwargs.get("model", 1),
        },
    }


def compare(
    tmp_path: Path,
    reference=POINTS,
    mobile=MOVED,
    *,
    ref_kw=None,
    mob_kw=None,
    extra_steps=(),
    **params,
):
    from alambiq.api import run as execute

    ref_kw, mob_kw = ref_kw or {}, mob_kw or {}
    steps = [
        inspect_step("ref", tmp_path, reference, **ref_kw),
        inspect_step("mob", tmp_path, mobile, **mob_kw),
        *extra_steps,
        {
            "id": "cmp",
            "capability": "structure.compare",
            "bind": {"reference": "ref.record", "mobile": "mob.record"},
            "params": {
                "reference_chain": "long_chain",
                "mobile_chain": "long_chain",
                **params,
            },
        },
    ]
    return execute({"id": "c", "steps": steps}, run_dir=tmp_path / "run")


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("cmp")


class TestRigidFit:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        return compare(tmp_path_factory.mktemp("rigid"), ref_kw={"model": 2})

    def test_a_rigid_motion_is_undone_exactly(self, outcome):
        out = ok(outcome)
        assert out["fitted_pairs"] == 4 and out["requested_pairs"] == 4
        assert out["rmsd_after_angstrom"] < 1e-6
        assert out["rmsd_before_angstrom"] > 5
        assert out["correspondence"] == "identical-query-positions"

    def test_the_fit_is_a_proper_rotation_and_a_translation(self, outcome):
        import numpy as np

        out = ok(outcome)
        rotation = np.array(out["rotation"])
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        # Mobile -> reference: rotate back a quarter turn and undo the shift.
        moved = np.array(MOVED[1], dtype=float)
        assert np.allclose(
            rotation @ moved + out["translation_angstrom"], POINTS[1], atol=1e-6
        )

    def test_the_exported_structure_puts_the_mobile_atoms_on_the_reference_points(
        self, outcome
    ):
        from alambiq.capabilities.characterize.science.dossier import inspect_structure

        out = ok(outcome)
        text = Path(out["fitted_structure"]).read_text()
        _, atoms = inspect_structure(text, "ACDE", 1)
        for target, atom in zip(POINTS, atoms, strict=True):
            for key, value in zip("xyz", target, strict=True):
                assert atom[key] == pytest.approx(value, abs=1e-6)

    def test_the_record_keeps_the_sources_and_a_hash_of_the_export(self, outcome):
        import hashlib

        out = ok(outcome)
        record = json.loads(Path(out["record"]).read_text())
        assert record["reference"]["model_number"] == 2
        assert record["method"] == "mapped-ca-superposition-v1"
        assert (
            record["fitted_structure_sha256"]
            == hashlib.sha256(
                Path(out["fitted_structure"]).read_text().encode()
            ).hexdigest()
        )
        assert record["fit_policy"].startswith(
            "Uniform alpha-carbon weights; every pair with an observed alpha carbon in "
            "both structures enters the fit"
        )

    def test_every_pair_lists_its_distances(self, outcome):
        pairs = ok(outcome)["pairs"]
        assert [(p["reference"], p["mobile"]) for p in pairs] == [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
        ]
        assert all(
            p["status"] == "fitted" and p["distance_after_angstrom"] < 1e-6
            for p in pairs
        )
        assert (
            pairs[0]["reference_residue"] == "A" and pairs[1]["mobile_residue"] == "C"
        )

    def test_the_report_states_the_rmsd_of_alpha_carbons(self, outcome, tmp_path):
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "RMSD of alpha carbons" in page

    def test_the_same_structure_against_itself_has_zero_rmsd_before_and_after(
        self, tmp_path
    ):
        out = ok(compare(tmp_path, POINTS, POINTS))
        assert out["rmsd_before_angstrom"] == pytest.approx(0, abs=1e-9)
        assert out["rmsd_after_angstrom"] == pytest.approx(0, abs=1e-9)


class TestWhatIsLeftOut:
    def test_a_missing_alpha_carbon_is_listed_and_the_denominator_is_not_quietly_changed(
        self, tmp_path
    ):
        outcome = compare(tmp_path, POINTS, MOVED, mob_kw={"missing": (4,)})
        out = ok(outcome)
        assert (out["requested_pairs"], out["fitted_pairs"]) == (4, 3)
        assert out["pairs"][3]["status"] == "missing-mobile-ca"
        assert out["mobile_coverage"] == 0.75 and out["reference_coverage"] == 0.75
        assert "left out of the fit" in " ".join(outcome.steps["cmp"].notes)

    def test_a_fit_covering_under_half_a_query_says_so(self, tmp_path):
        ten = [(1.7 * i, (i % 3) * 2.1, (i % 4) * 1.3 + 0.4 * i) for i in range(10)]
        moved = [(-y + 10, x - 3, z + 5) for x, y, z in ten]
        outcome = compare(
            tmp_path,
            ten,
            moved,
            ref_kw={"sequence": "ACDEFACDEF"},
            mob_kw={"sequence": "ACDEFACDEF"},
            residue_pairs=[{"reference": i, "mobile": i} for i in range(1, 5)],
        )
        out = ok(outcome)
        assert out["reference_coverage"] == 0.4 and out["mobile_coverage"] == 0.4
        assert out["rmsd_after_angstrom"] < 1e-6
        assert "Only 40% of the reference query is included in the fit" in " ".join(
            outcome.steps["cmp"].notes
        )

    def test_fewer_than_three_pairs_is_refused(self, tmp_path):
        outcome = compare(tmp_path, POINTS, MOVED, mob_kw={"missing": (3, 4)})
        assert "at least 3 are required" in outcome.steps["cmp"].error

    @pytest.mark.parametrize(
        "points",
        [[(0, 0, 0)] * 4, [(i, i, i) for i in range(4)]],
        ids=["coincident", "collinear"],
    )
    def test_points_that_do_not_define_a_rotation_are_refused(self, tmp_path, points):
        assert "noncollinear" in compare(tmp_path, POINTS, points).steps["cmp"].error


class TestCorrespondence:
    def test_different_sequences_need_an_explicit_correspondence(self, tmp_path):
        outcome = compare(tmp_path, POINTS, MOVED, mob_kw={"sequence": "ACDF"})
        assert "neither residue_pairs nor alignment" in outcome.steps["cmp"].error

    def test_explicit_pairs_fit_proteins_of_different_sequence(self, tmp_path):
        pairs = [{"reference": i, "mobile": i} for i in range(1, 5)]
        out = ok(
            compare(
                tmp_path,
                POINTS,
                MOVED,
                mob_kw={"sequence": "ACDF"},
                residue_pairs=pairs,
            )
        )
        assert out["correspondence"] == "explicit-query-positions"
        assert out["pairs"][3]["mobile_residue"] == "F"
        assert out["rmsd_after_angstrom"] < 1e-6

    def test_reordered_pairs_fit_the_pairs_not_the_order(self, tmp_path):
        # The mobile points are listed in reverse; pairing 1<->4, 2<->3, ... undoes it.
        out = ok(
            compare(
                tmp_path,
                POINTS,
                list(reversed(MOVED)),
                residue_pairs=[{"reference": i, "mobile": 5 - i} for i in range(1, 5)],
            )
        )
        assert out["rmsd_after_angstrom"] < 1e-6

    @pytest.mark.parametrize(
        "pairs",
        [
            [{"reference": 1, "mobile": 1}] * 3,
            [
                {"reference": 0, "mobile": 1},
                {"reference": 1, "mobile": 2},
                {"reference": 2, "mobile": 3},
            ],
            [
                {"reference": 5, "mobile": 1},
                {"reference": 1, "mobile": 2},
                {"reference": 2, "mobile": 3},
            ],
        ],
        ids=["repeated", "out-of-range-low", "out-of-range-high"],
    )
    def test_a_pairing_that_is_not_one_to_one_and_in_range_is_refused(
        self, tmp_path, pairs
    ):
        assert (
            "unique, in-range"
            in compare(tmp_path, residue_pairs=pairs).steps["cmp"].error
        )

    def test_an_alignment_gives_the_correspondence_through_its_columns(self, tmp_path):
        # Reference ACDE against a three-residue mobile ACF: the gap leaves reference
        # position 3 unpaired, so columns 1, 2 and 4 pair reference 1, 2, 4 with mobile 1, 2, 3.
        (tmp_path / "aln.fasta").write_text(">r\nACDE\n>m\nAC-F\n")
        mobile = [MOVED[0], MOVED[1], MOVED[3]]
        out = ok(
            compare(
                tmp_path,
                POINTS,
                mobile,
                mob_kw={"sequence": "ACF"},
                alignment=str(tmp_path / "aln.fasta"),
                alignment_format="fasta",
                reference_row="r",
                mobile_row="m",
            )
        )
        assert out["correspondence"] == "selected-alignment-columns"
        assert [(p["reference"], p["mobile"]) for p in out["pairs"]] == [
            (1, 1),
            (2, 2),
            (4, 3),
        ]
        assert out["fitted_pairs"] == 3 and out["rmsd_after_angstrom"] < 1e-6

    def test_residue_pairs_and_an_alignment_together_are_refused(self, tmp_path):
        (tmp_path / "aln.fasta").write_text(">r\nACDE\n>m\nACDE\n")
        outcome = compare(
            tmp_path,
            alignment=str(tmp_path / "aln.fasta"),
            alignment_format="fasta",
            reference_row="r",
            mobile_row="m",
            residue_pairs=[{"reference": 1, "mobile": 1}],
        )
        assert "Both alignment and residue_pairs are set" in outcome.steps["cmp"].error

    def test_alignment_options_without_an_alignment_are_refused(self, tmp_path):
        outcome = compare(tmp_path, alignment_format="fasta")
        assert "without alignment" in outcome.steps["cmp"].error


class TestSourcesAreChecked:
    def test_a_record_whose_atoms_were_edited_is_refused(self, tmp_path):
        from alambiq.api import run as execute

        outcome = compare(tmp_path)
        assert outcome.ok
        record_path = Path(outcome.outputs("ref")["record"])
        edited = json.loads(record_path.read_text())
        edited["atoms"][0]["x"] += 10
        bad = tmp_path / "edited.json"
        bad.write_text(json.dumps(edited))
        steps = [
            {
                "id": "cmp",
                "capability": "structure.compare",
                "params": {
                    "reference": str(bad),
                    "mobile": str(Path(outcome.outputs("mob")["record"])),
                    "reference_chain": "long_chain",
                    "mobile_chain": "long_chain",
                },
            }
        ]
        again = execute({"id": "c2", "steps": steps}, run_dir=tmp_path / "run2")
        assert (
            "differ from those parsed from its structure source"
            in again.steps["cmp"].error
        )

    def test_a_chain_that_is_not_in_the_record_is_refused(self, tmp_path):
        outcome = compare(tmp_path, mobile_chain="elsewhere")
        assert "label_asym_id" in outcome.steps["cmp"].error

    def test_an_ambiguous_alpha_carbon_is_refused(self, tmp_path):
        extra = "ATOM 5 C CA A ALA long_chain 1 1 . 1 0 0 41 author 1\n"
        outcome = compare(tmp_path, mob_kw={"extra": extra})
        assert "more than one alpha carbon" in outcome.steps["cmp"].error
