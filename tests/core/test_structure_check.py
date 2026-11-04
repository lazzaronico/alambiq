"""structure.check on constructed structures whose findings are known by construction.

Every structure below is written atom by atom, so the expected finding follows from the
coordinates: a residue given without CB lacks CB, two residues 5 Å apart form a break, a
peptide bond built at a stated dihedral has that omega, and an alpha carbon with N, C and CB
along three axes has the signed volume of their triple product.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

pytest.importorskip("gemmi")


def atom(
    serial, name, residue, chain, number, xyz, element, *, altloc=" ", het=False, occupancy=1.0
) -> str:
    padded = f" {name:<3}" if len(name) < 4 and len(element) == 1 else f"{name:<4}"
    x, y, z = xyz
    record = "HETATM" if het else "ATOM  "
    return (
        f"{record}{serial:5d} {padded}{altloc}{residue:>3} {chain}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{0.0:6.2f}          {element:>2}"
    )


def pdb(rows: list[tuple]) -> str:
    """Rows of (name, residue, chain, number, xyz, element, keywords)."""
    lines = []
    for i, row in enumerate(rows, 1):
        keywords = row[6] if len(row) > 6 else {}
        lines.append(atom(i, *row[:6], **keywords))
    return "\n".join(lines) + "\nEND\n"


def dipeptide(theta: float) -> str:
    """Alanine then glycine with the CA-C-N-CA dihedral at theta degrees (180 is trans)."""
    ca2 = (1.33 + 0.6, 1.4 * math.cos(math.radians(theta)), 1.4 * math.sin(math.radians(theta)))
    return pdb(
        [
            ("N", "ALA", "A", 1, (-1.5, 1.4, 1.0), "N"),
            ("CA", "ALA", "A", 1, (-0.6, 1.4, 0.0), "C"),
            ("C", "ALA", "A", 1, (0.0, 0.0, 0.0), "C"),
            ("O", "ALA", "A", 1, (0.0, -1.2, 0.0), "O"),
            ("CB", "ALA", "A", 1, (-0.6, 2.4, -1.0), "C"),
            ("N", "GLY", "A", 2, (1.33, 0.0, 0.0), "N"),
            ("CA", "GLY", "A", 2, ca2, "C"),
            ("C", "GLY", "A", 2, (3.5, 2.5, 2.5), "C"),
            ("O", "GLY", "A", 2, (4.5, 2.5, 3.5), "O"),
        ]
    )


def check(tmp_path: Path, text: str, name: str = "model.pdb", **params) -> dict:
    from alambiq.api import run as execute

    path = tmp_path / name
    path.write_text(text)
    outcome = execute(
        {
            "id": "c",
            "steps": [
                {"id": "check", "capability": "structure.check", "params": {"structure": str(path), **params}}
            ],
        },
        run_dir=tmp_path / "run",
    )
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("check")


def failure(tmp_path: Path, text: str, name: str = "model.pdb", **params) -> str:
    from alambiq.api import run as execute

    path = tmp_path / name
    path.write_text(text)
    outcome = execute(
        {
            "id": "c",
            "steps": [
                {"id": "check", "capability": "structure.check", "params": {"structure": str(path), **params}}
            ],
        },
        run_dir=tmp_path / "run",
    )
    assert not outcome.ok
    return " ".join(s.error for s in outcome.steps.values() if s.error)


class TestPeptideBonds:
    def test_a_trans_bond_is_not_reported(self, tmp_path):
        out = check(tmp_path, dipeptide(180))
        assert out["peptide_bonds"] == [] and out["breaks"] == []

    def test_a_cis_bond_is_reported_with_omega_near_zero(self, tmp_path):
        bond = check(tmp_path, dipeptide(0))["peptide_bonds"][0]
        assert bond["cis"] and not bond["twisted"]
        assert abs(bond["omega_degrees"]) < 1e-6

    def test_a_bond_twisted_by_60_degrees_is_reported_with_that_deviation(self, tmp_path):
        bond = check(tmp_path, dipeptide(120))["peptide_bonds"][0]
        assert bond["twisted"] and not bond["cis"]
        assert bond["deviation_degrees"] == pytest.approx(60.0, abs=0.1)

    def test_the_deviation_limit_decides_what_is_twisted(self, tmp_path):
        text = dipeptide(150)  # 30 degrees from trans
        assert check(tmp_path, text, peptide_deviation_degrees=45.0)["peptide_bonds"] == []
        twisted = check(tmp_path, text, "again.pdb", peptide_deviation_degrees=20.0)["peptide_bonds"]
        assert len(twisted) == 1 and twisted[0]["twisted"]


class TestBreaks:
    def two_residues(self, separation: float, second_number: int = 2) -> str:
        return pdb(
            [
                ("N", "GLY", "A", 1, (0.0, 0.0, 0.0), "N"),
                ("CA", "GLY", "A", 1, (1.4, 0.0, 0.0), "C"),
                ("C", "GLY", "A", 1, (2.0, 1.3, 0.0), "C"),
                ("O", "GLY", "A", 1, (2.0, 2.5, 0.0), "O"),
                ("N", "GLY", "A", second_number, (2.0 + separation, 1.3, 0.0), "N"),
                ("CA", "GLY", "A", second_number, (3.0 + separation, 2.3, 0.0), "C"),
                ("C", "GLY", "A", second_number, (4.0 + separation, 1.3, 0.0), "C"),
                ("O", "GLY", "A", second_number, (4.0 + separation, 0.1, 0.0), "O"),
            ]
        )

    def test_residues_5_angstrom_apart_form_a_break_with_that_distance(self, tmp_path):
        out = check(tmp_path, self.two_residues(5.0, second_number=5))
        (gap,) = out["breaks"]
        assert gap["c_n_distance_angstrom"] == pytest.approx(5.0, abs=1e-6)
        assert gap["numbering_gap"] == 3
        assert out["peptide_bonds"] == []

    def test_a_distance_below_the_break_distance_is_a_link(self, tmp_path):
        out = check(tmp_path, self.two_residues(1.33))
        assert out["breaks"] == []

    def test_the_break_distance_is_applied(self, tmp_path):
        text = self.two_residues(3.0)
        assert len(check(tmp_path, text)["breaks"]) == 1
        assert check(tmp_path, text, "wide.pdb", break_distance_angstrom=3.5)["breaks"] == []


class TestAlphaCarbons:
    def alanine(self, cb_z: float) -> str:
        return pdb(
            [
                ("CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C"),
                ("N", "ALA", "A", 1, (1.5, 0.0, 0.0), "N"),
                ("C", "ALA", "A", 1, (0.0, 1.5, 0.0), "C"),
                ("CB", "ALA", "A", 1, (0.0, 0.0, cb_z), "C"),
                ("O", "ALA", "A", 1, (0.0, 2.7, 0.0), "O"),
            ]
        )

    def test_n_c_cb_along_x_y_z_has_the_L_arrangement(self, tmp_path):
        assert check(tmp_path, self.alanine(1.5))["alpha_centers"] == []

    def test_reflecting_cb_inverts_the_alpha_carbon_with_the_volume_of_the_triple_product(self, tmp_path):
        (center,) = check(tmp_path, self.alanine(-1.5))["alpha_centers"]
        assert center["orientation"] == "inverted"
        assert center["signed_volume_angstrom3"] == pytest.approx(-1.5 * 1.5 * 1.5)

    def test_glycine_and_residues_without_cb_are_not_assessed(self, tmp_path):
        text = pdb(
            [
                ("CA", "ALA", "A", 1, (0.0, 0.0, 0.0), "C"),
                ("N", "ALA", "A", 1, (1.5, 0.0, 0.0), "N"),
                ("C", "ALA", "A", 1, (0.0, 1.5, 0.0), "C"),
            ]
        )
        out = check(tmp_path, text)
        assert out["unassessed_alpha_centers"] == 1 and out["alpha_centers"] == []


class TestResidueFindings:
    def test_a_residue_without_cb_lacks_cb(self, tmp_path):
        out = check(tmp_path, dipeptide(180))
        # The glycine is complete; the alanine of dipeptide() has all five atoms.
        assert out["missing_atoms"] == []
        text = pdb(
            [
                ("N", "ALA", "A", 1, (0.0, 0.0, 0.0), "N"),
                ("CA", "ALA", "A", 1, (1.4, 0.0, 0.0), "C"),
                ("C", "ALA", "A", 1, (2.0, 1.3, 0.0), "C"),
                ("O", "ALA", "A", 1, (2.0, 2.5, 0.0), "O"),
            ]
        )
        (finding,) = check(tmp_path, text, "nocb.pdb")["missing_atoms"]
        assert finding["name"] == "ALA" and finding["detail"] == "atoms CB"

    def test_protonation_names_count_as_the_standard_residue(self, tmp_path):
        text = pdb(
            [
                ("N", "HID", "A", 1, (0.0, 0.0, 0.0), "N"),
                ("CA", "HID", "A", 1, (1.4, 0.0, 0.0), "C"),
                ("C", "HID", "A", 1, (2.0, 1.3, 0.0), "C"),
                ("O", "HID", "A", 1, (2.0, 2.5, 0.0), "O"),
                ("CB", "HID", "A", 1, (1.4, -1.5, 0.0), "C"),
            ]
        )
        (finding,) = check(tmp_path, text)["missing_atoms"]
        assert finding["detail"] == "atoms CG, ND1, CD2, CE1, NE2"

    def test_alternate_locations_and_zero_occupancy_are_listed(self, tmp_path):
        text = pdb(
            [
                ("N", "SER", "A", 1, (0.0, 0.0, 0.0), "N"),
                ("CA", "SER", "A", 1, (1.4, 0.0, 0.0), "C"),
                ("C", "SER", "A", 1, (2.0, 1.3, 0.0), "C"),
                ("O", "SER", "A", 1, (2.0, 2.5, 0.0), "O"),
                ("CB", "SER", "A", 1, (1.4, -1.5, 0.0), "C"),
                ("OG", "SER", "A", 1, (1.4, -2.5, 0.5), "O", {"altloc": "A", "occupancy": 0.6}),
                ("OG", "SER", "A", 1, (0.4, -2.5, 0.5), "O", {"altloc": "B", "occupancy": 0.0}),
            ]
        )
        out = check(tmp_path, text)
        assert out["alternate_locations"][0]["detail"] == "locations A, B"
        assert out["zero_occupancy"][0]["detail"] == "atoms OG"
        assert out["missing_atoms"] == []

    def test_an_amino_acid_outside_the_20_is_listed_and_not_tested_for_completeness(self, tmp_path):
        text = pdb(
            [
                ("N", "MSE", "A", 1, (0.0, 0.0, 0.0), "N", {"het": True}),
                ("CA", "MSE", "A", 1, (1.4, 0.0, 0.0), "C", {"het": True}),
            ]
        )
        out = check(tmp_path, text)
        assert [r["name"] for r in out["nonstandard_residues"]] == ["MSE"]
        assert out["missing_atoms"] == [] and out["ligands"] == []
        assert out["chains"][0]["sequence"] == "M"


class TestHeteroAtoms:
    def test_ligand_metal_water_and_hydrogens_are_counted(self, tmp_path):
        text = pdb(
            [
                ("N", "GLY", "A", 1, (0.0, 0.0, 0.0), "N"),
                ("CA", "GLY", "A", 1, (1.4, 0.0, 0.0), "C"),
                ("C", "GLY", "A", 1, (2.0, 1.3, 0.0), "C"),
                ("O", "GLY", "A", 1, (2.0, 2.5, 0.0), "O"),
                ("H", "GLY", "A", 1, (-0.5, 0.8, 0.0), "H"),
                ("ZN", "ZN", "A", 101, (20.0, 20.0, 20.0), "Zn", {"het": True}),
                ("C1", "LIG", "A", 102, (30.0, 30.0, 30.0), "C", {"het": True}),
                ("C2", "LIG", "A", 102, (31.5, 30.0, 30.0), "C", {"het": True}),
                ("O", "HOH", "A", 103, (40.0, 40.0, 40.0), "O", {"het": True}),
            ]
        )
        out = check(tmp_path, text)
        assert [(m["name"], m["element"]) for m in out["metals"]] == [("ZN", "Zn")]
        assert [(h["name"], h["atoms"]) for h in out["ligands"]] == [("LIG", 2)]
        assert out["waters"] == 1 and out["hydrogens"] == 1 and out["atoms"] == 9


class TestCloseContacts:
    def two_serines(self, separation: float) -> str:
        """Two serine side-chain oxygens, in residues 3 and 40, `separation` apart."""
        return pdb(
            [
                ("OG", "SER", "A", 3, (0.0, 0.0, 0.0), "O"),
                ("OG", "SER", "A", 40, (0.0, 0.0, separation), "O"),
            ]
        )

    def test_two_distant_residues_with_atoms_1_5_angstrom_apart_clash(self, tmp_path):
        out = check(tmp_path, self.two_serines(1.5))
        (contact,) = out["close_contacts"]
        assert contact["distance_angstrom"] == pytest.approx(1.5)
        assert contact["first"]["atom"] == contact["second"]["atom"] == "OG"
        assert {contact["first"]["number"], contact["second"]["number"]} == {3, 40}

    def test_a_distance_above_the_clash_distance_is_not_reported_and_the_distance_is_applied(self, tmp_path):
        text = self.two_serines(3.0)
        assert check(tmp_path, text)["close_contacts"] == []
        wide = check(tmp_path, text, "wide.pdb", clash_distance_angstrom=3.5)["close_contacts"]
        assert len(wide) == 1 and wide[0]["distance_angstrom"] == pytest.approx(3.0)

    def test_residues_with_consecutive_numbers_are_not_contacts(self, tmp_path):
        text = pdb(
            [
                ("OG", "SER", "A", 3, (0.0, 0.0, 0.0), "O"),
                ("OG", "SER", "A", 4, (0.0, 0.0, 1.5), "O"),
            ]
        )
        assert check(tmp_path, text)["close_contacts"] == []

    def test_bonded_neighbors_and_the_atoms_inside_a_residue_are_not_contacts(self, tmp_path):
        assert check(tmp_path, dipeptide(180))["close_contacts"] == []

    def test_the_sulfur_pair_of_a_disulfide_bond_is_not_a_contact(self, tmp_path):
        text = pdb(
            [
                ("SG", "CYS", "A", 3, (0.0, 0.0, 0.0), "S"),
                ("SG", "CYS", "A", 40, (0.0, 0.0, 2.05), "S"),
            ]
        )
        assert check(tmp_path, text)["close_contacts"] == []

    def test_a_metal_next_to_a_residue_atom_is_not_a_contact(self, tmp_path):
        text = pdb(
            [
                ("SG", "CYS", "A", 3, (0.0, 0.0, 0.0), "S"),
                ("ZN", "ZN", "A", 101, (0.0, 0.0, 2.3), "Zn", {"het": True}),
            ]
        )
        assert check(tmp_path, text)["close_contacts"] == []

    def test_different_alternate_locations_do_not_clash(self, tmp_path):
        text = pdb(
            [
                ("OG", "SER", "A", 3, (0.0, 0.0, 0.0), "O", {"altloc": "A"}),
                ("OG", "SER", "A", 40, (0.0, 0.0, 1.5), "O", {"altloc": "B"}),
            ]
        )
        assert check(tmp_path, text)["close_contacts"] == []


class TestFileHandling:
    def test_the_digest_is_that_of_the_file(self, tmp_path):
        text = dipeptide(180)
        out = check(tmp_path, text)
        assert out["structure_sha256"] == hashlib.sha256(text.encode()).hexdigest()

    def test_chains_report_the_observed_sequence(self, tmp_path):
        chain = check(tmp_path, dipeptide(180))["chains"][0]
        assert chain["id"] == "A" and chain["sequence"] == "AG" and chain["residues"] == 2
        assert chain["first"] == "1" and chain["last"] == "2"

    def test_the_model_index_selects_the_model(self, tmp_path):
        first = dipeptide(180).replace("END\n", "")
        second = dipeptide(0).replace("END\n", "")
        text = f"MODEL        1\n{first}\nENDMDL\nMODEL        2\n{second}\nENDMDL\nEND\n"
        out = check(tmp_path, text)
        assert out["model_count"] == 2 and out["peptide_bonds"] == []
        again = check(tmp_path, text, "again.pdb", model_index=1)
        assert len(again["peptide_bonds"]) == 1 and again["model_index"] == 1

    def test_a_model_index_beyond_the_models_is_refused_with_the_range(self, tmp_path):
        message = failure(tmp_path, dipeptide(180), model_index=3)
        assert "holds 1 model(s)" in message and "0 to 0" in message

    def test_an_extension_that_names_no_format_needs_structure_format(self, tmp_path):
        message = failure(tmp_path, dipeptide(180), "model.txt")
        assert "set structure_format" in message
        assert check(tmp_path, dipeptide(180), "model.txt", structure_format="pdb")["atoms"] == 9

    def test_an_mmcif_file_is_read(self, tmp_path):
        import gemmi

        source = tmp_path / "source.pdb"
        source.write_text(dipeptide(0))
        structure = gemmi.read_pdb(str(source))
        structure.make_mmcif_document().write_file(str(tmp_path / "model.cif"))
        out = check(tmp_path, (tmp_path / "model.cif").read_text(), "same.cif")
        assert out["chains"][0]["sequence"] == "AG" and len(out["peptide_bonds"]) == 1

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("model_index", 1000),
            ("break_distance_angstrom", 1.0),
            ("peptide_deviation_degrees", 0.5),
            ("clash_distance_angstrom", 5.0),
        ],
    )
    def test_a_value_outside_its_range_is_refused_by_name(self, tmp_path, name, value):
        assert name in failure(tmp_path, dipeptide(180), **{name: value})
