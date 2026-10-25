"""Reading a Boltz-predicted protein-ligand complex for simulation.

Only standard amino-acid residues and the one ligand residue Boltz names LIG1 are
accepted; anything else is refused. The protein must have
exactly the heavy atoms of its residues (the terminal OXT is added later), the
declared sequences, consecutive numbering, no alternate locations and finite
coordinates. Disulfides are found by sulfur distance and an ambiguous distance is
refused. The result is a protein PDB with the names the force field expects.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

RESIDUE_TO_CODE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
EXPECTED_HEAVY_ATOMS = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    },
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
}


def finite_position(atom: Any) -> bool:
    return all(math.isfinite(value) for value in (atom.pos.x, atom.pos.y, atom.pos.z))


def read_prediction(structure_path: Path, sequences: list[str] | None):
    import gemmi

    # Stored files have no extension, so the format is named.
    structure = gemmi.read_structure(str(structure_path), format=gemmi.CoorFormat.Mmcif)
    if len(structure) != 1:
        raise ValueError(
            f"The predicted complex contains {len(structure)} models; exactly 1 is required"
        )
    model = structure[0]
    protein_chains = []
    ligand_residues = []
    unsupported = []
    for chain in model:
        protein = []
        for residue in chain:
            name = residue.name.upper()
            if name in RESIDUE_TO_CODE:
                protein.append(residue)
            elif name == "LIG1":
                ligand_residues.append((chain.name, residue))
            else:
                unsupported.append(f"{chain.name}:{residue.seqid.num}:{name}")
        if protein:
            protein_chains.append((chain.name, protein))
    if unsupported:
        raise ValueError(
            "The predicted complex contains unsupported residues: "
            + ", ".join(unsupported)
            + "; only standard amino acids and one LIG1 residue are accepted"
        )
    if sequences is None:
        sequences = [
            "".join(RESIDUE_TO_CODE[r.name.upper()] for r in residues)
            for _, residues in protein_chains
        ]
    if len(protein_chains) != len(sequences):
        raise ValueError(
            "The protein chain count does not match the declared sequences: "
            f"{len(protein_chains)} chains, {len(sequences)} sequences; the counts must be equal"
        )
    if len(ligand_residues) != 1:
        raise ValueError(
            f"The predicted complex contains {len(ligand_residues)} LIG1 residues; "
            "exactly one LIG1 residue is required"
        )
    for chain_index, ((chain_name, residues), declared) in enumerate(
        zip(protein_chains, sequences, strict=True)
    ):
        observed = "".join(
            RESIDUE_TO_CODE[residue.name.upper()] for residue in residues
        )
        if observed != declared.upper():
            raise ValueError(
                f"Protein sequence mismatch for chain {chain_index} ({chain_name}): the "
                "sequence in the structure differs from the declared sequence; the two "
                "must be equal"
            )
        for residue_index, residue in enumerate(residues, start=1):
            if residue.seqid.num != residue_index:
                raise ValueError(
                    f"Residue {residue.seqid.num} of chain {chain_name} is at position "
                    f"{residue_index}; residue numbers must be consecutive starting at 1"
                )
            atoms = [atom for atom in residue if atom.element.name not in {"H", "D"}]
            if any(atom.altloc not in {"\x00", " ", "."} for atom in atoms):
                raise ValueError(
                    f"Residue {residue_index} of chain {chain_name} has alternate atom "
                    "locations; only atoms without alternate locations are accepted"
                )
            if any(not finite_position(atom) for atom in atoms):
                raise ValueError(
                    f"Residue {residue_index} of chain {chain_name} has non-finite "
                    "coordinates; finite coordinates are required"
                )
            names = [atom.name.strip() for atom in atoms]
            if len(names) != len(set(names)):
                raise ValueError(
                    f"Residue {residue_index} of chain {chain_name} repeats a heavy-atom "
                    "name; unique names within a residue are required"
                )
            expected = EXPECTED_HEAVY_ATOMS[residue.name.upper()]
            if set(names) != expected:
                missing = sorted(expected - set(names))
                extra = sorted(set(names) - expected)
                raise ValueError(
                    f"Protein heavy-atom mismatch at {chain_name}:{residue_index}; "
                    f"missing={missing}, extra={extra}; exactly the heavy atoms of the "
                    "standard residue are required"
                )
    ligand_chain, ligand = ligand_residues[0]
    ligand_atoms = [atom for atom in ligand if atom.element.name not in {"H", "D"}]
    if any(atom.altloc not in {"\x00", " ", "."} for atom in ligand_atoms):
        raise ValueError(
            "The LIG1 residue has alternate atom locations; only atoms without "
            "alternate locations are accepted"
        )
    if any(not finite_position(atom) for atom in ligand_atoms):
        raise ValueError(
            "The LIG1 residue has non-finite coordinates; finite coordinates are required"
        )
    names = [atom.name.strip() for atom in ligand_atoms]
    if len(names) != len(set(names)):
        raise ValueError(
            "The LIG1 residue repeats an atom name; ligand atom names must be unique"
        )
    return protein_chains, ligand_chain, ligand_atoms, sequences


def find_disulfides(protein_chains: list[tuple[str, list[Any]]]):
    cysteines = []
    global_index = 0
    for chain_name, residues in protein_chains:
        for residue in residues:
            global_index += 1
            if residue.name.upper() == "CYS":
                sulfur = next(atom for atom in residue if atom.name.strip() == "SG")
                cysteines.append(
                    (chain_name, residue.seqid.num, global_index, sulfur.pos)
                )
    bonds = []
    ambiguous = []
    for left, right in itertools.combinations(cysteines, 2):
        distance = left[3].dist(right[3])
        if distance <= 2.5:
            bonds.append((left, right, distance))
        elif distance <= 3.5:
            ambiguous.append((left, right, distance))
    if ambiguous:
        details = ", ".join(
            f"{left[0]}:{left[1]}-{right[0]}:{right[1]}={distance:.3f} Å"
            for left, right, distance in ambiguous
        )
        raise ValueError(
            "Ambiguous cysteine SG geometry: "
            + details
            + "; an SG-SG distance above 2.5 Å and up to 3.5 Å is compatible with both a disulfide and free thiols"
        )
    occurrences = [item[2] for bond in bonds for item in bond[:2]]
    if len(occurrences) != len(set(occurrences)):
        raise ValueError(
            "A cysteine has an SG-SG distance of at most 2.5 Å to more than one other "
            "cysteine; multiple disulfides on one cysteine are not supported"
        )
    return bonds


def pdb_atom_name(name: str, element: str) -> str:
    if len(name) >= 4:
        return name[:4]
    return f" {name:<3}" if len(element) == 1 else f"{name:<4}"


def write_protein_pdb(
    protein_chains: list[tuple[str, list[Any]]],
    disulfides: list[tuple[Any, Any, float]],
    output: Path,
) -> None:
    disulfide_residues = {item[2] for bond in disulfides for item in bond[:2]}
    lines = []
    serial = 1
    global_index = 0
    chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if len(protein_chains) > len(chain_ids):
        raise ValueError(
            f"The predicted complex has {len(protein_chains)} protein chains; at most "
            f"{len(chain_ids)} are supported by the PDB chain identifiers"
        )
    for chain_index, (_, residues) in enumerate(protein_chains):
        pdb_chain = chain_ids[chain_index]
        for residue in residues:
            global_index += 1
            residue_name = residue.name.upper()
            if global_index in disulfide_residues:
                residue_name = "CYX"
            elif residue_name == "HIS":
                residue_name = "HIE"
            for atom in residue:
                if atom.element.name in {"H", "D"}:
                    continue
                name = atom.name.strip()
                element = atom.element.name.upper()
                lines.append(
                    f"ATOM  {serial:5d} {pdb_atom_name(name, element)} {residue_name:>3} "
                    f"{pdb_chain}{residue.seqid.num:4d}    "
                    f"{atom.pos.x:8.3f}{atom.pos.y:8.3f}{atom.pos.z:8.3f}"
                    f"  1.00{float(atom.b_iso):6.2f}          {element:>2}"
                )
                serial += 1
        lines.append("TER")
    lines.append("END")
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
