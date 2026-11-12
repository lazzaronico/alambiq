"""Superpose a structure on a reference over all its chains and measure where its ligand lies.

A predicted complex holds a protein and a ligand. A deposited structure of the same protein
holds the catalytic site: a metal, a cofactor, a bound inhibitor. The capability superposes the
predicted structure on the deposited one by the alpha carbons of the protein and reports the
distance from the ligand to chosen atoms of the deposited structure, for example from the
carbonyl oxygen of a substrate to the catalytic zinc.

The method:

- Both files are mmCIF or PDB, read by content. The first model is used. Chain ids are the
  author chain ids (`auth_asym_id` of an mmCIF).
- Residues correspond when they have the same author residue number and residue name in paired
  chains. Alpha carbons of the paired residues enter the fit with equal weights. The fit is the
  least-squares rotation of Kabsch (1976, 1978), computed by singular value decomposition.
- Chains are paired by the assignment of mobile protein chains to reference protein chains with
  the smallest alpha-carbon RMSD, which is searched over all assignments (at most 6 chains), or
  taken from `chain_map`. A prediction of a homo-oligomer names its chains in no relation to
  the crystal, so the search is needed for it.
- The ligand is the one residue of the mobile structure that matches `ligand`. Its heavy atoms
  are moved by the fit.
- Each target names a residue name and an atom name of the reference, for example `ZN` and `ZN`
  or `NAP` and `C4N`. The capability measures the distance from the ligand to that atom in every
  residue of the reference that has it, so a tetramer gives four distances. A target that names
  `ligand_atom` measures that atom; without it, the nearest heavy atom of the ligand to each
  reference atom is reported.

Limits.

- The distances state where the predicted ligand lies relative to the reference atoms. They do
  not state that the pose is the productive binding mode, and a reference structure that holds
  no substrate does not show where the substrate binds.
- The reference atoms are measured as deposited; they carry the experimental uncertainty, and
  a metal or cofactor that the prediction does not contain does not influence the predicted pose.
- A mobile structure that was trained on, or is a close homologue of, the reference makes
  an agreement unsurprising; it checks a campaign on a known answer.
- The fit uses alpha carbons only. A site that moves relative to the rest of the protein is
  placed by the fit of the whole protein.
- When the mobile structure is nearly symmetric, several assignments of chains fit with almost
  the same RMSD, and they place the ligand differently relative to the reference atoms. The
  assignments within `equivalent_fit_angstrom` of the smallest RMSD are equivalent fits. The
  result reports the chosen fit and, for each target, the range of the nearest distance over
  all equivalent fits, so a distance that depends on the choice is visible.
- Ligand atom names are those of the file. A predicted structure names atoms by the program
  that wrote it, so the names change with the program; `ligand.atoms` lists each atom with the
  atoms it is bonded to (bonds read from distances), which identifies an atom by its
  chemistry.
- `own_targets` measure the ligand against atoms of the mobile structure itself, for example
  the cofactor and the metal that a prediction holds, and `own_contacts` gives the nearest heavy
  atom of the mobile structure outside the ligand to each ligand atom. They describe the
  prediction by itself; `targets` and `contacts` compare it with the reference, and a
  cofactor that the prediction places differently from the deposited one gives different
  numbers in the two.
- `contacts` gives, for each ligand atom, the nearest heavy atom of the whole reference, so a
  ligand placed on the cofactor or inside the protein shows as a short distance. A metal that
  the ligand coordinates is the nearest atom of that ligand atom at the coordination distance.
- Bonds of the ligand come from distances between heavy atoms (covalent radii plus 0.45 Å).

References.

- Kabsch W. A solution for the best rotation to relate two sets of vectors. Acta Crystallogr A
  32, 922-923 (1976). https://doi.org/10.1107/S0567739476001873
- Kabsch W. A discussion of the solution for the best rotation to relate two sets of vectors.
  Acta Crystallogr A 34, 827-828 (1978). https://doi.org/10.1107/S0567739478001680
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alambiq.capabilities.investigate.structure_measure_ligand_site_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class LigandSelector:
    """The ligand residue of the mobile structure.

    The selector must match exactly one residue.

    residue_name:
        The residue name in the file, for example `LIG1`.
    chain_id:
        The chain id. A prediction writes its ligand as a chain of its own.
    residue_number:
        The author residue number.

    At least one of `residue_name` and `chain_id` is required.
    """

    residue_name: str | None = None
    chain_id: str | None = None
    residue_number: int | None = None


@dataclass(frozen=True, slots=True)
class SiteTarget:
    """An atom of the reference that the ligand is measured against.

    label:
        A name for the measurement, unique in the request.
    residue_name:
        The residue name of the reference atom, for example `ZN` or `NAP`.
    atom:
        The atom name in the reference, for example `ZN` or `C4N`.
    ligand_atom:
        The atom of the ligand to measure. Without it the nearest heavy atom of the ligand
        to each reference atom is reported.
    """

    label: str
    residue_name: str
    atom: str
    ligand_atom: str | None = None


@dataclass(frozen=True, slots=True)
class SiteRequest:
    """The two structures, the ligand and the reference atoms.

    reference:
        An mmCIF or PDB file, for example a deposited structure.
    mobile:
        An mmCIF or PDB file with the protein and the ligand, for example a predicted
        complex. It is moved onto the reference.
    ligand:
        The ligand residue of the mobile structure.
    targets:
        1 to 20 atoms of the reference.
    chain_map:
        Optional. {mobile chain: reference chain} for the chains that correspond. Without
        it the mobile and the reference structure must have the same number of protein
        chains (at most 6), and the assignment with the smallest RMSD is used.
    equivalent_fit_angstrom:
        0 to 5 Å. Chain assignments whose RMSD lies within this of the smallest are
        equivalent fits: the result gives the range of every nearest distance over them.
        Default 0.1.
    own_targets:
        0 to 20 atoms of the mobile structure itself, for example the NADP and the zinc that a
        prediction holds. The ligand is measured against them as against `targets`, and the
        fit does not enter. Default none.
    """

    reference: str
    mobile: str
    ligand: LigandSelector
    targets: list[SiteTarget]
    chain_map: dict[str, str] = field(default_factory=dict)
    equivalent_fit_angstrom: float = 0.1
    own_targets: list[SiteTarget] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SiteFit:
    """The superposition."""

    rmsd_angstrom: float
    fitted_residues: int
    mobile_coverage: float
    """Fraction of the alpha carbons of the paired mobile chains that took part in the fit."""
    reference_coverage: float
    chain_assignment: dict[str, str]
    """{mobile chain: reference chain}."""
    assignments_tested: int
    runner_up_rmsd_angstrom: float | None
    """The smallest RMSD of an assignment other than the chosen one; None if only one was tested."""
    equivalent_fits: int
    """Assignments within `equivalent_fit_angstrom` of the smallest RMSD, the chosen one included."""
    rotation: list[list[float]]
    translation_angstrom: list[float]
    """A mobile point x lies at rotation x + translation in the reference frame."""


@dataclass(frozen=True, slots=True)
class LigandAtom:
    name: str
    element: str
    bonded_to: list[str]


@dataclass(frozen=True, slots=True)
class LigandSummary:
    chain_id: str
    residue_number: int
    residue_name: str
    atoms: list[LigandAtom]
    centroid_angstrom: list[float]
    """The centroid of the heavy atoms in the reference frame."""


@dataclass(frozen=True, slots=True)
class SiteDistance:
    """The distance from a ligand atom to one atom of the reference."""

    chain_id: str
    residue_number: int
    ligand_atom: str
    distance_angstrom: float


@dataclass(frozen=True, slots=True)
class SiteContact:
    """The reference atom nearest to one ligand atom."""

    ligand_atom: str
    chain_id: str
    residue_number: int
    residue_name: str
    atom: str
    distance_angstrom: float


@dataclass(frozen=True, slots=True)
class OwnMeasurement:
    """One own target, with the distance to every residue of the mobile structure that has the atom."""

    label: str
    residue_name: str
    atom: str
    ligand_atom_requested: str | None
    nearest: SiteDistance
    instances: list[SiteDistance]
    """Every residue with the atom, from the nearest to the farthest."""


@dataclass(frozen=True, slots=True)
class SiteMeasurement:
    """One target, with the distance to every reference residue that has the atom."""

    label: str
    residue_name: str
    atom: str
    ligand_atom_requested: str | None
    nearest: SiteDistance
    """The reference residue nearest to the ligand in the chosen fit."""
    instances: list[SiteDistance]
    """Every reference residue with the atom, from the nearest to the farthest."""
    nearest_range_angstrom: list[float]
    """The smallest and the largest nearest distance over the equivalent fits."""


@dataclass(frozen=True, slots=True)
class SiteResult:
    """The fit, the ligand, and the distances."""

    fit: SiteFit
    ligand: LigandSummary
    measurements: list[SiteMeasurement]
    contacts: list[SiteContact]
    """For each ligand atom, the nearest heavy atom of the reference (any residue, waters
    excluded) in the chosen fit, from the closest pair to the farthest. A distance far below
    the sum of the van der Waals radii shows that the ligand overlaps the reference."""
    own_measurements: list[OwnMeasurement]
    """The distances to `own_targets`, in the mobile structure."""
    own_contacts: list[SiteContact]
    """For each ligand atom, the nearest heavy atom of the mobile structure outside the ligand
    (waters excluded), from the closest pair to the farthest."""
    fitted_structure: str
    """The mobile structure in the reference frame, as an mmCIF file."""
    record: str
    gemmi_version: str


@capability(
    "structure.measure_ligand_site",
    version="1.2",
    family="investigate",
    summary="Superpose a predicted complex on a reference and measure its ligand to reference atoms",
    render=render,
)
def measure_ligand_site(request: SiteRequest, ctx: StepContext) -> SiteResult:
    """Fit the mobile structure on the reference over all chains and measure the ligand."""
    from alambiq.capabilities.investigate.science.ligand_site import (
        measure_site,
        read_structure,
        structure_text,
    )

    reference = read_structure(ctx.read_text(request.reference), "reference")
    mobile = read_structure(ctx.read_text(request.mobile), "mobile")
    record, placed = measure_site(
        mobile,
        reference,
        ligand={
            "residue_name": request.ligand.residue_name,
            "chain_id": request.ligand.chain_id,
            "residue_number": request.ligand.residue_number,
        },
        targets=[
            {
                "label": t.label,
                "residue_name": t.residue_name,
                "atom": t.atom,
                "ligand_atom": t.ligand_atom,
            }
            for t in request.targets
        ],
        chain_map=request.chain_map or None,
        tolerance=request.equivalent_fit_angstrom,
        own_targets=[
            {
                "label": t.label,
                "residue_name": t.residue_name,
                "atom": t.atom,
                "ligand_atom": t.ligand_atom,
            }
            for t in request.own_targets
        ],
    )
    fit = record["fit"]
    if fit["mobile_residues"] and fit["fitted_residues"] / fit["mobile_residues"] < 0.5:
        ctx.note(
            f"Only {fit['fitted_residues'] / fit['mobile_residues']:.0%} of the mobile alpha "
            "carbons correspond to a reference residue (same chain pairing, author number and "
            "residue name); the fit rests on the rest."
        )
    runner_up = fit["runner_up_rmsd_angstrom"]
    if fit["equivalent_fits"] > 1:
        ctx.note(
            f"{fit['equivalent_fits']} assignments of the chains fit within "
            f"{request.equivalent_fit_angstrom:g} Å of the smallest RMSD "
            f"({fit['rmsd_angstrom']:.3f} Å); the nearest distance of each target varies with "
            "the assignment, and its range is in nearest_range_angstrom."
        )
    return SiteResult(
        fit=SiteFit(
            rmsd_angstrom=fit["rmsd_angstrom"],
            fitted_residues=fit["fitted_residues"],
            mobile_coverage=fit["fitted_residues"] / fit["mobile_residues"],
            reference_coverage=fit["fitted_residues"] / fit["reference_residues"],
            chain_assignment=fit["chain_assignment"],
            assignments_tested=fit["assignments_tested"],
            runner_up_rmsd_angstrom=runner_up,
            equivalent_fits=fit["equivalent_fits"],
            rotation=fit["rotation"],
            translation_angstrom=fit["translation_angstrom"],
        ),
        ligand=LigandSummary(
            chain_id=record["ligand"]["chain_id"],
            residue_number=record["ligand"]["residue_number"],
            residue_name=record["ligand"]["residue_name"],
            atoms=[LigandAtom(**a) for a in record["ligand"]["atoms"]],
            centroid_angstrom=record["ligand"]["centroid_angstrom"],
        ),
        measurements=[
            SiteMeasurement(
                label=m["label"],
                residue_name=m["residue_name"],
                atom=m["atom"],
                ligand_atom_requested=m["ligand_atom_requested"],
                nearest=SiteDistance(**m["nearest"]),
                instances=[SiteDistance(**i) for i in m["instances"]],
                nearest_range_angstrom=m["nearest_range_angstrom"],
            )
            for m in record["measurements"]
        ],
        contacts=[SiteContact(**c) for c in record["contacts"]],
        own_measurements=[
            OwnMeasurement(
                label=m["label"],
                residue_name=m["residue_name"],
                atom=m["atom"],
                ligand_atom_requested=m["ligand_atom_requested"],
                nearest=SiteDistance(**m["nearest"]),
                instances=[SiteDistance(**i) for i in m["instances"]],
            )
            for m in record["own_measurements"]
        ],
        own_contacts=[SiteContact(**c) for c in record["own_contacts"]],
        fitted_structure=str(
            ctx.keep_text(structure_text(placed), "chemical/x-mmcif").path
        ),
        record=ctx.keep_json(record),
        gemmi_version=record["gemmi_version"],
    )
