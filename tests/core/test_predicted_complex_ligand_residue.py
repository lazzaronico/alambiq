"""The ligand residue that the preparation of a predicted complex reports, read from its topology."""

from __future__ import annotations

import pytest

from alambiq.capabilities.investigate.science.predicted_complex import ligand_residue


def atom(serial: int, name: str, residue: str, chain: str, number: int, hetero: bool = False) -> str:
    kind = "HETATM" if hetero else "ATOM  "
    return f"{kind}{serial:5d} {name:<4s} {residue:>3s} {chain}{number:4d}    0.000   0.000   0.000  1.00  0.00"


def write(tmp_path, lines: list[str]):
    path = tmp_path / "topology.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def test_the_ligand_follows_the_protein_and_has_the_chain_the_file_names(tmp_path):
    path = write(
        tmp_path,
        [
            atom(1, "N", "MET", "0", 1),
            atom(2, "CA", "MET", "0", 1),
            atom(3, "N", "LYS", "0", 2),
            "TER",
            atom(4, "C001", "LIG", "1", 3, hetero=True),
            atom(5, "C002", "LIG", "1", 3, hetero=True),
            "TER",
            atom(6, "O", "HOH", "2", 4, hetero=True),
        ],
    )
    assert ligand_residue(path) == {
        "chain_id": "1",
        "residue_number": 3,
        "residue_name": "LIG",
        "residue_index": 2,
    }


def test_a_blank_chain_column_is_named_by_the_zero_based_chain_index(tmp_path):
    path = write(
        tmp_path,
        [
            atom(1, "N", "MET", " ", 1),
            "TER",
            atom(2, "C001", "LIG", " ", 2, hetero=True),
            "TER",
        ],
    )
    found = ligand_residue(path)
    assert (found["chain_id"], found["residue_number"], found["residue_index"]) == ("1", 2, 1)


def test_a_topology_without_the_ligand_or_with_two_is_refused(tmp_path):
    none = write(tmp_path, [atom(1, "N", "MET", "0", 1)])
    with pytest.raises(ValueError, match="0 residues named LIG"):
        ligand_residue(none)
    two = write(
        tmp_path,
        [
            atom(1, "C001", "LIG", "1", 1, hetero=True),
            atom(2, "C001", "LIG", "1", 2, hetero=True),
        ],
    )
    with pytest.raises(ValueError, match="2 residues named LIG"):
        ligand_residue(two)
