"""Checks of one model of a protein structure: missing atoms, alternate locations, chain breaks,
peptide planarity, alpha-carbon chirality, close contacts, ligands and metals."""

from __future__ import annotations

import math
from typing import Any

import gemmi

EXPECTED_ATOMS = {
    "ALA": "N CA C O CB",
    "ARG": "N CA C O CB CG CD NE CZ NH1 NH2",
    "ASN": "N CA C O CB CG OD1 ND2",
    "ASP": "N CA C O CB CG OD1 OD2",
    "CYS": "N CA C O CB SG",
    "GLN": "N CA C O CB CG CD OE1 NE2",
    "GLU": "N CA C O CB CG CD OE1 OE2",
    "GLY": "N CA C O",
    "HIS": "N CA C O CB CG ND1 CD2 CE1 NE2",
    "ILE": "N CA C O CB CG1 CG2 CD1",
    "LEU": "N CA C O CB CG CD1 CD2",
    "LYS": "N CA C O CB CG CD CE NZ",
    "MET": "N CA C O CB CG SD CE",
    "PHE": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ",
    "PRO": "N CA C O CB CG CD",
    "SER": "N CA C O CB OG",
    "THR": "N CA C O CB OG1 CG2",
    "TRP": "N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH",
    "VAL": "N CA C O CB CG1 CG2",
}
EXPECTED = {name: set(atoms.split()) for name, atoms in EXPECTED_ATOMS.items()}
# Protonation and tautomer names that Amber-style files use for the standard residues.
ALIASES = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
}  # fmt: skip
HYDROGENS = {"H", "D"}


def residue_id(chain: gemmi.Chain, residue: gemmi.Residue) -> dict[str, Any]:
    return {
        "chain": chain.name,
        "number": residue.seqid.num,
        "insertion": residue.seqid.icode.strip(),
        "name": residue.name,
    }


def standard_name(residue: gemmi.Residue) -> str | None:
    name = ALIASES.get(residue.name, residue.name)
    return name if name in EXPECTED else None


def kind(residue: gemmi.Residue) -> str:
    """`standard`, `nonstandard` (an amino acid outside the 20), `water`, `metal` or `ligand`."""
    if standard_name(residue):
        return "standard"
    info = gemmi.find_tabulated_residue(residue.name)
    if info is not None and info.is_water():
        return "water"
    if info is not None and info.is_amino_acid():
        return "nonstandard"
    if len(residue) == 1 and residue[0].element.is_metal:
        return "metal"
    return "ligand"


def first_atom(residue: gemmi.Residue, name: str) -> gemmi.Atom | None:
    """The atom of the first conformer that carries the name."""
    found = residue.find_atom(name, "*")
    return found


def load_model(path: str, fmt: str, model_index: int) -> tuple[gemmi.Structure, gemmi.Model]:
    if fmt == "pdb":
        structure = gemmi.read_pdb(path)
    else:
        structure = gemmi.make_structure_from_block(gemmi.cif.read(path).sole_block())
    if not 0 <= model_index < len(structure):
        raise ValueError(
            f"model_index is {model_index}; the file holds {len(structure)} model(s), "
            f"so an index from 0 to {len(structure) - 1} is required"
        )
    return structure, structure[model_index]


def chain_summaries(model: gemmi.Model) -> list[dict[str, Any]]:
    chains = []
    for chain in model:
        residues = [r for r in chain if kind(r) in ("standard", "nonstandard")]
        if not residues:
            continue
        letters = []
        for r in residues:
            info = gemmi.find_tabulated_residue(standard_name(r) or r.name)
            letters.append(info.one_letter_code.upper() if info and info.one_letter_code != " " else "X")
        chains.append(
            {
                "id": chain.name,
                "residues": len(residues),
                "first": f"{residues[0].seqid.num}{residues[0].seqid.icode.strip()}",
                "last": f"{residues[-1].seqid.num}{residues[-1].seqid.icode.strip()}",
                "sequence": "".join(letters),
            }
        )
    return chains


def residue_issues(model: gemmi.Model) -> dict[str, list[dict[str, Any]]]:
    """Alternate locations, missing heavy atoms, zero occupancy and nonstandard residues."""
    found: dict[str, list[dict[str, Any]]] = {
        "alternate_locations": [], "missing_atoms": [], "zero_occupancy": [], "nonstandard_residues": [],
    }  # fmt: skip
    for chain in model:
        for residue in chain:
            what = kind(residue)
            identity = residue_id(chain, residue)
            if what == "nonstandard":
                found["nonstandard_residues"].append({**identity, "detail": "amino acid outside the 20 standard residues"})
            if what not in ("standard", "nonstandard"):
                continue
            altlocs = sorted({a.altloc for a in residue if a.altloc != "\0"})
            if altlocs:
                found["alternate_locations"].append({**identity, "detail": "locations " + ", ".join(altlocs)})
            zero = sorted({a.name for a in residue if a.occ == 0 and a.element.name not in HYDROGENS})
            if zero:
                found["zero_occupancy"].append({**identity, "detail": "atoms " + ", ".join(zero)})
            standard = standard_name(residue)
            if standard:
                present = {a.name for a in residue}
                missing = [a for a in EXPECTED_ATOMS[standard].split() if a not in present]
                if missing:
                    found["missing_atoms"].append({**identity, "detail": "atoms " + ", ".join(missing)})
    return found


def links(model: gemmi.Model, break_distance: float, deviation_limit: float) -> dict[str, Any]:
    """Chain breaks and the geometry of each peptide bond between consecutive amino acids."""
    breaks, peptides, unassessed = [], [], []
    for chain in model:
        residues = [r for r in chain if kind(r) in ("standard", "nonstandard")]
        for a, b in zip(residues, residues[1:], strict=False):
            c, n = first_atom(a, "C"), first_atom(b, "N")
            if c is None or n is None:
                unassessed.append(
                    {"from": residue_id(chain, a), "to": residue_id(chain, b), "detail": "C or N is absent"}
                )
                continue
            distance = c.pos.dist(n.pos)
            if distance > break_distance:
                breaks.append(
                    {
                        "from": residue_id(chain, a),
                        "to": residue_id(chain, b),
                        "distance": distance,
                        "numbering_gap": b.seqid.num - a.seqid.num - 1,
                    }
                )
                continue
            omega = gemmi.calculate_omega(a, b)
            if math.isnan(omega):
                unassessed.append(
                    {"from": residue_id(chain, a), "to": residue_id(chain, b), "detail": "CA is absent"}
                )
                continue
            degrees = math.degrees(omega)
            deviation = min(abs(degrees), 180 - abs(degrees))
            cis = abs(degrees) < 90
            if cis or deviation > deviation_limit:
                peptides.append(
                    {
                        "from": residue_id(chain, a),
                        "to": residue_id(chain, b),
                        "omega_degrees": degrees,
                        "deviation_degrees": deviation,
                        "cis": cis,
                        "twisted": deviation > deviation_limit,
                    }
                )
    return {"breaks": breaks, "peptides": peptides, "unassessed_links": unassessed}


def alpha_centers(model: gemmi.Model) -> dict[str, Any]:
    """The signed volume dot(N-CA, cross(C-CA, CB-CA)) of each alpha carbon, positive for L."""
    flagged, unassessed = [], []
    for chain in model:
        for residue in chain:
            if kind(residue) != "standard" or standard_name(residue) == "GLY":
                continue
            atoms = [first_atom(residue, name) for name in ("N", "CA", "C", "CB")]
            if any(a is None for a in atoms):
                unassessed.append(residue_id(chain, residue))
                continue
            n, ca, c, cb = (a.pos for a in atoms)
            n_v, c_v, cb_v = n - ca, c - ca, cb - ca
            volume = n_v.dot(c_v.cross(cb_v))
            if volume <= 0:
                flagged.append(
                    {**residue_id(chain, residue), "signed_volume_angstrom3": volume,
                     "orientation": "inverted" if volume < 0 else "planar"}
                )  # fmt: skip
    return {"alpha_centers": flagged, "unassessed_alpha_centers": unassessed}


def hetero(model: gemmi.Model) -> dict[str, Any]:
    ligands, metals, waters, hydrogens, atoms = [], [], 0, 0, 0
    for chain in model:
        for residue in chain:
            what = kind(residue)
            atoms += len(residue)
            hydrogens += sum(a.element.name in HYDROGENS for a in residue)
            if what == "water":
                waters += 1
            elif what == "metal":
                metals.append({**residue_id(chain, residue), "element": residue[0].element.name})
            elif what == "ligand":
                ligands.append({**residue_id(chain, residue), "atoms": len(residue)})
    return {"ligands": ligands, "metals": metals, "waters": waters, "hydrogens": hydrogens, "atoms": atoms}


def close_contacts(
    structure: gemmi.Structure, model: gemmi.Model, distance: float
) -> list[dict[str, Any]]:
    """Pairs of heavy atoms of amino-acid residues closer than `distance` that no bond explains.

    Pairs within a residue or between residues whose numbers differ by at most 1 in one chain, pairs
    involving a metal, the sulfur pair of a disulfide bond and pairs of different alternate
    locations are left out. Contacts with symmetry mates are not searched.
    """
    cell = structure.cell
    neighbors = gemmi.NeighborSearch(model, cell, 5).populate()
    search = gemmi.ContactSearch(distance)
    search.ignore = gemmi.ContactSearch.Ignore.SameResidue
    contacts = []
    seen = set()
    for result in search.find_contacts(neighbors):
        if result.image_idx != 0:
            continue
        one, two = result.partner1, result.partner2
        if any(p.atom.element.name in HYDROGENS or p.atom.element.is_metal for p in (one, two)):
            continue
        if any(kind(p.residue) not in ("standard", "nonstandard") for p in (one, two)):
            continue
        if one.chain.name == two.chain.name and abs(one.residue.seqid.num - two.residue.seqid.num) <= 1:
            continue
        if one.atom.name == "SG" and two.atom.name == "SG":
            continue
        if one.atom.altloc != "\0" and two.atom.altloc != "\0" and one.atom.altloc != two.atom.altloc:
            continue
        key = tuple(sorted((p.chain.name, p.residue.seqid.num, p.residue.seqid.icode, p.atom.name, p.atom.altloc) for p in (one, two)))
        if key in seen:
            continue
        seen.add(key)
        contacts.append(
            {
                "first": {**residue_id(one.chain, one.residue), "atom": one.atom.name},
                "second": {**residue_id(two.chain, two.residue), "atom": two.atom.name},
                "distance": result.dist,
            }
        )
    return sorted(contacts, key=lambda c: c["distance"])


def check_model(
    path: str,
    fmt: str,
    model_index: int,
    *,
    break_distance: float,
    deviation_limit: float,
    clash_distance: float,
) -> dict[str, Any]:
    structure, model = load_model(path, fmt, model_index)
    record = {
        "model_count": len(structure),
        "chains": chain_summaries(model),
        **residue_issues(model),
        **links(model, break_distance, deviation_limit),
        **alpha_centers(model),
        **hetero(model),
        "close_contacts": close_contacts(structure, model, clash_distance),
        "gemmi_version": f"gemmi {gemmi.__version__}",
    }
    return record
