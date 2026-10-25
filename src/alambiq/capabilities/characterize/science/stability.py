"""Mapping a structure and substitutions into the form the stability model reads.

Pure functions over text. The model needs one chain's complete backbone
(N, CA, C, O for every residue, in order, with no chain break) and the
substitutions by position in that chain; this reads them from a PDB or mmCIF
text and refuses anything that would make the mapping a guess: a chain that
does not match the stated construct, a residue without an unambiguous occupied
backbone, a break, or a substitution whose wild-type residue is not what the
structure has there.
"""

from __future__ import annotations

import math
from typing import Any

from alambiq.capabilities.shared.sequences import parse_residue  # noqa: F401

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def normalize_structure(
    structure_text: str,
    structure_format: str,
    chain_id: str,
    model_index: int,
    sequence: str,
    mutations: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the model's request and a description of what the model did not see."""
    import gemmi

    if not 2 <= len(sequence) <= 1024 or set(sequence) - set(ALPHABET):
        raise ValueError(
            f"sequence: the construct has {len(sequence)} residues or contains a "
            "residue outside the 20 canonical amino acids; a complete construct of "
            "2–1024 canonical residues is required"
        )
    if not chain_id.strip():
        raise ValueError("chain_id is empty; one exact author chain identifier is required")
    structure = (
        gemmi.read_pdb_string(structure_text)
        if structure_format == "pdb"
        else gemmi.make_structure_from_block(
            gemmi.cif.read_string(structure_text).sole_block()
        )
    )
    # mmCIF can store polymer, ligand and water parts of one author chain apart.
    structure.merge_chain_parts()
    if not 0 <= model_index < len(structure):
        raise ValueError(
            f"model_index: the requested structure model is absent; the structure "
            f"holds {len(structure)} model(s), numbered from 0"
        )
    model = structure[model_index]
    chains = [chain for chain in model if chain.name == chain_id]
    if len(chains) != 1:
        raise ValueError(
            f"chain_id: author chain {chain_id!r} must resolve uniquely; the model "
            f"holds {len(chains)} chains with that identifier"
        )

    residues: list[dict[str, str]] = []
    coordinates_angstrom: list[list[list[float]]] = []
    letters: list[str] = []
    seen: set[tuple[str, str]] = set()
    omitted_residues: list[dict[str, str]] = []
    water = 0
    for residue in chains[0]:
        info = gemmi.find_tabulated_residue(residue.name)
        if residue.is_water():
            water += 1
            continue
        if residue.entity_type in (
            gemmi.EntityType.NonPolymer,
            gemmi.EntityType.Branched,
        ):
            omitted_residues.append(
                {
                    "residue_id": str(residue.seqid.num),
                    "insertion_code": residue.seqid.icode.strip(),
                    "residue_name": residue.name,
                    "entity_type": residue.entity_type.name,
                }
            )
            continue
        if not info.is_amino_acid() or info.one_letter_code not in ALPHABET:
            raise ValueError(
                f"structure: residue {residue.seqid.num}{residue.seqid.icode.strip()} "
                f"({residue.name}) of chain {chain_id} is not a canonical amino acid; "
                "the selected polymer requires canonical protein residues"
            )
        key = str(residue.seqid.num), residue.seqid.icode.strip()
        if key in seen:
            raise ValueError(
                f"structure: residue {key[0]}{key[1]} occurs twice in chain {chain_id}; "
                "each residue identity must be unique"
            )
        seen.add(key)
        backbone = []
        for name in ("N", "CA", "C", "O"):
            atoms = [atom for atom in residue if atom.name == name]
            if (
                len(atoms) != 1
                or atoms[0].altloc not in ("\x00", " ")
                or atoms[0].occ <= 0
            ):
                raise ValueError(
                    f"structure: residue {key[0]}{key[1]} of chain {chain_id} lacks "
                    "an unambiguous occupied N/CA/C/O backbone; every residue "
                    "requires exactly one N, CA, C and O atom with occupancy above 0 "
                    "and no alternate-location label"
                )
            backbone.append([atoms[0].pos.x, atoms[0].pos.y, atoms[0].pos.z])
        if (
            coordinates_angstrom
            and not 1.0 <= math.dist(coordinates_angstrom[-1][2], backbone[0]) <= 1.8
        ):
            raise ValueError(
                f"structure: chain {chain_id} has a peptide-chain break before "
                f"residue {key[0]}{key[1]} (C to N distance outside 1.0 to 1.8 "
                "angstrom); a complete contiguous construct is required"
            )
        residues.append(
            {
                "chain_id": chain_id,
                "residue_id": key[0],
                "insertion_code": key[1],
                "residue_name": residue.name,
            }
        )
        coordinates_angstrom.append(backbone)
        letters.append(info.one_letter_code)
    if "".join(letters) != sequence:
        raise ValueError(
            "sequence: the observed chain sequence differs from the supplied "
            "construct sequence; the chain must match the construct residue for residue"
        )

    by_source = {
        (r["residue_id"], r["insertion_code"]): i for i, r in enumerate(residues)
    }
    if not 1 <= len(mutations) <= 20000:
        raise ValueError(
            f"mutations: {len(mutations)} substitutions are given; 1–20000 single "
            "substitutions are required"
        )
    normalized: list[dict[str, Any]] = []
    seen_substitutions: set[tuple[int, str]] = set()
    for row in mutations:
        key = row["residue_id"], row["insertion_code"]
        if key not in by_source:
            raise ValueError(
                f"mutations: source residue {key[0]}{key[1]} is absent from chain {chain_id}"
            )
        position = by_source[key]
        if row["wild_type"] != sequence[position]:
            raise ValueError(
                f"mutations: the substitution at {key[0]}{key[1]} names wild type {row['wild_type']}, "
                f"but the structure has {sequence[position]}"
            )
        if len(row["mutant"]) != 1 or row["mutant"] not in ALPHABET:
            raise ValueError(
                f"mutations: mutant {row['mutant']!r} at {key[0]}{key[1]} must be one "
                "of the 20 canonical residues"
            )
        if (position, row["mutant"]) in seen_substitutions:
            raise ValueError(
                f"Duplicate substitution in mutations: residue {key[0]}{key[1]} to "
                f"{row['mutant']} is listed twice"
            )
        seen_substitutions.add((position, row["mutant"]))
        normalized.append(
            {
                "position": position,
                "wild_type": row["wild_type"],
                "mutant": row["mutant"],
            }
        )

    request = {
        "sequence": sequence,
        "residues": residues,
        "coordinates_angstrom": coordinates_angstrom,
        "mutations": normalized,
    }
    omitted = {
        "other_chains": [c.name for c in model if c.name != chain_id],
        "selected_chain_omitted_residues": omitted_residues,
        "selected_chain_water_residues": water,
        "scope": (
            "Model context is the selected protein chain backbone; other "
            "chains, ligands, waters and side-chain coordinates are excluded"
        ),
    }
    return request, omitted
