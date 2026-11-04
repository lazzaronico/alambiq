"""Check one model of a protein structure file and report what is missing, doubled, broken or crowded.

The result lists, for the selected model of a PDB or mmCIF file:

- the chains of amino-acid residues with their observed sequences;
- residues with alternate locations, heavy atoms missing from a standard residue, atoms of
  zero occupancy, and amino acids outside the 20 standard residues;
- chain breaks, where the carbonyl carbon of one residue lies farther than the break distance
  from the nitrogen of the next residue in the file, with the number of residues the
  numbering skips;
- peptide bonds that are cis or twisted from planarity beyond a stated deviation;
- alpha carbons with an inverted or planar arrangement of N, C and CB;
- pairs of heavy atoms closer than the clash distance that no bond explains;
- ligands, metal ions, waters and hydrogens, with counts.

The method:

- gemmi reads the file. The residues of the first conformer supply the atoms of each test.
- A standard residue is expected to carry the heavy atoms of its standard side chain and of the
  backbone atoms N, CA, C and O. Histidine, cysteine, aspartate, glutamate and lysine
  protonation names (HID, HIE, HIP, CYX, ASH, GLH, LYN and similar) count as the standard
  residue. OXT is not required.
- Two consecutive amino acids of a chain are linked when the C-N distance does not exceed the
  break distance. The torsion omega of a linked peptide bond is the dihedral angle
  CA-C-N-CA. Its deviation is the smaller angle to 0 or 180 degrees.
- The chirality of an alpha carbon is the signed volume dot(N-CA, cross(C-CA, CB-CA)) in
  cubic angstroms. It is positive for the L arrangement, including cysteine. Glycine has no
  stereocenter.
- A close contact is a pair of heavy atoms of amino-acid residues closer than the clash
  distance. Pairs inside a residue, between residues whose numbers differ by at most 1 in a
  chain, involving a metal ion, between the sulfur atoms of a disulfide bond, and between
  different alternate locations are left out.

Limits.

- The check reports the file as it is and changes nothing. The researcher decides which
  findings matter for the intended use.
- Residues that gemmi does not read as amino acids are ligands, metals or waters. A
  nonstandard amino acid is listed and its atoms are not tested for completeness.
- The missing-atom test uses the heavy atoms of the standard residues in a table. A modified
  side chain or a covalently attached group appears as an extra atom or as a ligand.
- Contacts with symmetry mates are not searched.
- Hydrogen atoms are counted and left out of the contact search.
- Peptide planarity and chirality are assessed from the atoms of the first conformer.
- The expected geometry follows from the residue table and the angles above. The values are
  descriptive: a flagged residue may be real, for example a cis proline, a strained loop or
  a lower-resolution region.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.investigate.structure_check_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class CheckRequest:
    """The structure file, the model, and the thresholds of the checks.

    structure:
        A path to a PDB or mmCIF file.
    structure_format:
        `pdb` or `mmcif`. Without it, the extension of the file decides: `.pdb` and `.ent` read
        as PDB, `.cif` and `.mmcif` as mmCIF.
    model_index:
        The zero-based index of the model to check, 0 to 999; default 0.
    break_distance_angstrom:
        The C-N distance above which two consecutive residues of a chain are reported as a
        break, 1.4 to 10; default 2.0.
    peptide_deviation_degrees:
        The deviation of omega from 0 or 180 degrees above which a peptide bond is reported
        as twisted, 1 to 90; default 30. A cis peptide bond is reported whatever its deviation.
    clash_distance_angstrom:
        The distance below which a pair of heavy atoms is reported as a close contact, 1.0 to
        4.0; default 2.2.
    """

    structure: str
    structure_format: Literal["pdb", "mmcif"] | None = None
    model_index: int = 0
    break_distance_angstrom: float = 2.0
    peptide_deviation_degrees: float = 30.0
    clash_distance_angstrom: float = 2.2


@dataclass(frozen=True, slots=True)
class ChainSummary:
    """One chain of amino-acid residues."""

    id: str
    residues: int
    first: str
    """The number of the first residue, with its insertion code."""
    last: str
    sequence: str
    """The one-letter code of the observed residues; X for a residue without one."""


@dataclass(frozen=True, slots=True)
class ResidueFinding:
    """A residue with a finding."""

    chain: str
    number: int
    insertion: str
    name: str
    detail: str


@dataclass(frozen=True, slots=True)
class Break:
    """Two consecutive residues of a chain whose carbonyl carbon and nitrogen lie apart."""

    chain: str
    from_number: int
    from_name: str
    to_number: int
    to_name: str
    c_n_distance_angstrom: float
    numbering_gap: int
    """The number of residue numbers skipped between the two residues."""


@dataclass(frozen=True, slots=True)
class PeptideBond:
    """A peptide bond that is cis or twisted from planarity."""

    chain: str
    from_number: int
    from_name: str
    to_number: int
    to_name: str
    omega_degrees: float
    deviation_degrees: float
    """The smaller angle between omega and 0 or 180 degrees."""
    cis: bool
    twisted: bool


@dataclass(frozen=True, slots=True)
class AlphaCenter:
    """An alpha carbon whose N, C and CB do not form the L arrangement."""

    chain: str
    number: int
    insertion: str
    name: str
    signed_volume_angstrom3: float
    orientation: str
    """`inverted` for a negative volume, `planar` for zero."""


@dataclass(frozen=True, slots=True)
class AtomRef:
    """One atom of a residue."""

    chain: str
    number: int
    insertion: str
    name: str
    atom: str


@dataclass(frozen=True, slots=True)
class Contact:
    """A pair of heavy atoms closer than the clash distance."""

    first: AtomRef
    second: AtomRef
    distance_angstrom: float


@dataclass(frozen=True, slots=True)
class Hetero:
    """A ligand or a metal ion."""

    chain: str
    number: int
    insertion: str
    name: str
    atoms: int
    element: str | None
    """The element of a metal ion; None for a ligand."""


@dataclass(frozen=True, slots=True)
class CheckResult:
    """What the checks found in the selected model."""

    structure_sha256: str
    model_count: int
    model_index: int
    atoms: int
    chains: list[ChainSummary]
    alternate_locations: list[ResidueFinding]
    missing_atoms: list[ResidueFinding]
    zero_occupancy: list[ResidueFinding]
    nonstandard_residues: list[ResidueFinding]
    breaks: list[Break]
    unassessed_links: int
    """Consecutive amino acids whose C, N or CA is absent, so that the link was not assessed."""
    peptide_bonds: list[PeptideBond]
    alpha_centers: list[AlphaCenter]
    unassessed_alpha_centers: int
    """Standard residues without N, CA, C or CB, whose chirality was not assessed."""
    close_contacts: list[Contact]
    ligands: list[Hetero]
    metals: list[Hetero]
    waters: int
    hydrogens: int
    gemmi_version: str
    findings: dict[str, int]
    """The number of findings in each list above, by name."""


def _finding(row: dict) -> ResidueFinding:
    return ResidueFinding(
        chain=row["chain"],
        number=row["number"],
        insertion=row["insertion"],
        name=row["name"],
        detail=row["detail"],
    )


def _ref(row: dict) -> AtomRef:
    return AtomRef(
        chain=row["chain"],
        number=row["number"],
        insertion=row["insertion"],
        name=row["name"],
        atom=row["atom"],
    )


@capability(
    "structure.check",
    version="1.0",
    family="investigate",
    summary="Report missing atoms, alternate locations, breaks and close contacts in a structure",
    render=render,
)
def check(request: CheckRequest, ctx: StepContext) -> CheckResult:
    """Read one model of the structure file and report the findings of each check."""
    from alambiq.capabilities.investigate.science.structure_check import check_model

    path = ctx.resolve(request.structure)
    fmt = request.structure_format
    if fmt is None:
        suffix = path.suffix.lower()
        if suffix in (".pdb", ".ent"):
            fmt = "pdb"
        elif suffix in (".cif", ".mmcif"):
            fmt = "mmcif"
        else:
            raise ValueError(
                f"The extension {path.suffix!r} of the structure file does not identify "
                "PDB or mmCIF; set structure_format"
            )
    for name, value, low, high in (
        ("model_index", request.model_index, 0, 999),
        ("break_distance_angstrom", request.break_distance_angstrom, 1.4, 10.0),
        ("peptide_deviation_degrees", request.peptide_deviation_degrees, 1.0, 90.0),
        ("clash_distance_angstrom", request.clash_distance_angstrom, 1.0, 4.0),
    ):
        if not low <= value <= high:
            raise ValueError(f"{name} is {value}; a value from {low} to {high} is required")
    record = check_model(
        str(path),
        fmt,
        request.model_index,
        break_distance=request.break_distance_angstrom,
        deviation_limit=request.peptide_deviation_degrees,
        clash_distance=request.clash_distance_angstrom,
    )
    result = CheckResult(
        structure_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_count=record["model_count"],
        model_index=request.model_index,
        atoms=record["atoms"],
        chains=[ChainSummary(**c) for c in record["chains"]],
        alternate_locations=[_finding(r) for r in record["alternate_locations"]],
        missing_atoms=[_finding(r) for r in record["missing_atoms"]],
        zero_occupancy=[_finding(r) for r in record["zero_occupancy"]],
        nonstandard_residues=[_finding(r) for r in record["nonstandard_residues"]],
        breaks=[
            Break(
                chain=b["from"]["chain"],
                from_number=b["from"]["number"],
                from_name=b["from"]["name"],
                to_number=b["to"]["number"],
                to_name=b["to"]["name"],
                c_n_distance_angstrom=b["distance"],
                numbering_gap=b["numbering_gap"],
            )
            for b in record["breaks"]
        ],
        unassessed_links=len(record["unassessed_links"]),
        peptide_bonds=[
            PeptideBond(
                chain=p["from"]["chain"],
                from_number=p["from"]["number"],
                from_name=p["from"]["name"],
                to_number=p["to"]["number"],
                to_name=p["to"]["name"],
                omega_degrees=p["omega_degrees"],
                deviation_degrees=p["deviation_degrees"],
                cis=p["cis"],
                twisted=p["twisted"],
            )
            for p in record["peptides"]
        ],
        alpha_centers=[
            AlphaCenter(
                chain=a["chain"],
                number=a["number"],
                insertion=a["insertion"],
                name=a["name"],
                signed_volume_angstrom3=a["signed_volume_angstrom3"],
                orientation=a["orientation"],
            )
            for a in record["alpha_centers"]
        ],
        unassessed_alpha_centers=len(record["unassessed_alpha_centers"]),
        close_contacts=[
            Contact(
                first=_ref(c["first"]),
                second=_ref(c["second"]),
                distance_angstrom=c["distance"],
            )
            for c in record["close_contacts"]
        ],
        ligands=[
            Hetero(
                chain=h["chain"],
                number=h["number"],
                insertion=h["insertion"],
                name=h["name"],
                atoms=h["atoms"],
                element=None,
            )
            for h in record["ligands"]
        ],
        metals=[
            Hetero(
                chain=h["chain"],
                number=h["number"],
                insertion=h["insertion"],
                name=h["name"],
                atoms=1,
                element=h["element"],
            )
            for h in record["metals"]
        ],
        waters=record["waters"],
        hydrogens=record["hydrogens"],
        gemmi_version=record["gemmi_version"],
        findings={
            "alternate_locations": len(record["alternate_locations"]),
            "missing_atoms": len(record["missing_atoms"]),
            "zero_occupancy": len(record["zero_occupancy"]),
            "nonstandard_residues": len(record["nonstandard_residues"]),
            "breaks": len(record["breaks"]),
            "peptide_bonds": len(record["peptides"]),
            "alpha_centers": len(record["alpha_centers"]),
            "close_contacts": len(record["close_contacts"]),
        },
    )
    if result.model_count > 1:
        ctx.note(
            f"The file holds {result.model_count} models; model_index {request.model_index} was checked."
        )
    return result
