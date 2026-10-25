"""Prepare one protein chain for molecular dynamics: hydrogens, solvent, minimization.

One standard-protein chain of a PDB or mmCIF file is selected by its author chain id
and stripped of its hydrogens. Hydrogens are then added at a stated pH, with residue
protonation variants that can be overridden in the request. The capability
parameterizes the chain with Amber ff14SB, places it in explicit TIP3P water with NaCl
and neutralizing ions or in implicit GBn2 solvent, and minimizes the energy. Optionally
PDBFixer rebuilds missing heavy atoms of standard residues and applies explicit residue
substitutions from standard templates. The output is a serialized OpenMM system and
state, the topology, the prepared structure and a manifest (a JSON file that lists every
operation performed).

The result is a starting point for dynamics, with every decision (which atoms were
removed, which residues were protonated in which state, which were rebuilt) recorded and
the geometry of the result described.

Limits. Protonation uses the pH-dependent defaults of OpenMM, assigned per residue type
at the stated pH, and the overrides of the request. The default protonation of a histidine
in a catalytic site can differ from its actual state; `residue_variants` overrides it.
Only a single standard protein chain is prepared, as it appears in the selected model:
cofactors, ligands, metals, other chains and nonstandard residues are refused
(`require-protein-only`) or explicitly dropped (`selected-chain-only`, with what was
dropped listed). Completion applies to missing heavy atoms of existing standard
residues: with `heavy_atom_completion: pdbfixer-standard` the atoms other than the
backbone atoms N, CA, C and O are rebuilt, and a chain with an incomplete or
discontinuous backbone is refused. Minimization stops at the stated maximum number of
iterations or force tolerance; when the tolerance is met first, the root-mean-square of
the force components at the end point is below it, and the forces before and after are
reported. A
substitution is a modeled residue placed from a standard template. The geometry
diagnostics (peptide planarity, alpha-carbon chirality) describe the prepared
coordinates and leave them unchanged. A twisted peptide bond or an inverted alpha-carbon in the
input remains in the prepared system, and dynamics starts from it. A chain of fewer
than 3 or more than 1,000 residues, or a prepared system of more than 250,000 atoms,
is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from alambiq.capabilities.investigate.structure_prepare_protein_system_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

VARIANTS = (
    "ASH",
    "ASP",
    "CYS",
    "CYX",
    "GLH",
    "GLU",
    "HID",
    "HIE",
    "HIP",
    "HIN",
    "LYN",
    "LYS",
)
RESIDUE_LETTERS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True, slots=True)
class ResidueVariant:
    """A protonation state to impose on one residue.

    residue:
        The author residue number, with an insertion letter if there is one.
    variant:
        An OpenMM variant: ASH/ASP (aspartate), CYS/CYX (cysteine, free or in a
        disulfide), GLH/GLU, HID/HIE/HIP/HIN (histidine tautomers and cations),
        LYN/LYS.
    """

    residue: str
    variant: Literal[
        "ASH",
        "ASP",
        "CYS",
        "CYX",
        "GLH",
        "GLU",
        "HID",
        "HIE",
        "HIP",
        "HIN",
        "LYN",
        "LYS",
    ]


@dataclass(frozen=True, slots=True)
class Mutation:
    """A residue to substitute from a standard template.

    residue:
        The author residue number, with an insertion letter if there is one.
    wild_type, mutant:
        One-letter codes; the wild type is checked against the structure.
    """

    residue: str
    wild_type: str
    mutant: str


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    """The structure, the chain, and the physical setup.

    structure:
        A path to a PDB or mmCIF file.
    chain_id:
        The author chain to prepare.
    structure_format:
        `pdb` or `mmcif`; by default read from the file extension.
    model_index:
        Which model of an ensemble to use, from 0.
    solvent:
        `tip3p` for explicit water in a periodic box, or `gbn2` for implicit
        solvent with no box.
    ph:
        0 to 14, for hydrogen placement.
    ionic_strength_molar:
        Added NaCl in an explicit box (0 to 1), or the Debye screening for
        `gbn2`. Neutralizing ions are added in addition.
    padding_nm:
        Explicit only: water between the solute and the box edge, 1 to 3 nm.
    box_shape:
        Explicit only: `cube`, `dodecahedron` or `octahedron`.
    temperature_kelvin, solute_dielectric, solvent_dielectric:
        Implicit only: the screening temperature and dielectric constants.
    minimization_max_iterations, minimization_tolerance_kj_mol_nm:
        Bounds on the minimization, 1 to 10,000 iterations and a force
        tolerance from 0.01 to 1,000.
    heavy_atom_completion:
        `pdbfixer-standard` rebuilds missing heavy atoms of standard residues;
        required for `substitutions`.
    source_component_policy:
        `require-protein-only` refuses a file whose chain or file holds anything
        but standard protein; `selected-chain-only` drops the rest and lists it.
    residue_variants, substitutions:
        Protonation overrides and explicit residue substitutions.
    platform, platform_properties, seed:
        The OpenMM platform (`CPU`, `Reference`, `CUDA`, `OpenCL`, with no
        fallback), its properties (by default one CPU thread, so that the same seed gives the
        same result), and the seed for completion.
    """

    structure: str
    chain_id: str
    structure_format: Literal["pdb", "mmcif"] | None = None
    model_index: int = 0
    solvent: Literal["tip3p", "gbn2"] = "tip3p"
    ph: float = 7.0
    ionic_strength_molar: float = 0.15
    padding_nm: float = 2.0
    box_shape: Literal["cube", "dodecahedron", "octahedron"] = "dodecahedron"
    temperature_kelvin: float = 300.0
    solute_dielectric: float = 1.0
    solvent_dielectric: float = 78.5
    minimization_max_iterations: int = 5000
    minimization_tolerance_kj_mol_nm: float = 10.0
    heavy_atom_completion: Literal["none", "pdbfixer-standard"] = "none"
    source_component_policy: Literal["require-protein-only", "selected-chain-only"] = (
        "require-protein-only"
    )
    residue_variants: list[ResidueVariant] = field(default_factory=list)
    substitutions: list[Mutation] = field(default_factory=list)
    platform: Literal["CPU", "Reference", "CUDA", "OpenCL"] = "CPU"
    platform_properties: dict[str, str] = field(default_factory=dict)
    seed: int = 1


@dataclass(frozen=True, slots=True)
class PrepareResult:
    """The prepared system's size, energies and geometry, and where its files are."""

    solvent: str
    protein_residues: int
    protein_atoms: int
    atom_count: int
    residue_counts: dict[str, int]
    """Residues of each name in the prepared system, water and ions included."""
    potential_energy_before_kj_mol: float
    potential_energy_after_kj_mol: float
    rms_force_after_kj_mol_nm: float
    max_force_after_kj_mol_nm: float
    box_volume_nm3: float | None
    removed_source_atoms: dict[str, int]
    added_heavy_atoms: int
    substitutions_applied: int
    max_peptide_planarity_deviation_degrees: float | None
    inverted_alpha_residues: list[int]
    system_xml: str
    state_xml: str
    topology_pdb: str
    prepared_structure: str
    sequence_fasta: str
    manifest: str
    versions: dict[str, Any]


@capability(
    "structure.prepare_protein_system",
    version="1.0",
    family="investigate",
    summary="Prepare a protein chain for molecular dynamics: hydrogens, solvent, minimization",
    render=render,
    environment="openmm",
)
def prepare_protein_system(request: PrepareRequest, ctx: StepContext) -> PrepareResult:
    """Prepare the chain, minimize it, and describe what was done."""
    from alambiq.capabilities.investigate.science.simulation import (
        parse_residue,
        platform_properties,
    )

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
    if not request.chain_id.strip():
        raise ValueError("chain_id is empty; one exact chain_id is required")
    explicit = request.solvent == "tip3p"
    for name, value, low, high in (
        ("model_index", request.model_index, 0, 999),
        ("seed", request.seed, 1, 2147483647),
        ("minimization_max_iterations", request.minimization_max_iterations, 1, 10000),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; an integer from {low} to {high} is required"
            )
    for name, value, low, high in (
        ("ph", request.ph, 0, 14),
        ("ionic_strength_molar", request.ionic_strength_molar, 0, 1),
        (
            "minimization_tolerance_kj_mol_nm",
            request.minimization_tolerance_kj_mol_nm,
            0.01,
            1000,
        ),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value:g}; a value from {low:g} to {high:g} is required"
            )
    if explicit and not 1 <= request.padding_nm <= 3:
        raise ValueError(
            f"padding_nm is {request.padding_nm}; a value from 1 to 3 is required"
        )
    if not explicit:
        for name, value, low, high in (
            ("temperature_kelvin", request.temperature_kelvin, 200, 400),
            ("solute_dielectric", request.solute_dielectric, 1, 20),
            ("solvent_dielectric", request.solvent_dielectric, 20, 120),
        ):
            if not low <= value <= high:
                raise ValueError(
                    f"{name} is {value:g}; a value from {low:g} to {high:g} is required"
                )
    if request.substitutions and request.heavy_atom_completion != "pdbfixer-standard":
        raise ValueError(
            "substitutions is set while heavy_atom_completion is "
            f"{request.heavy_atom_completion}; substitutions require "
            "heavy_atom_completion: pdbfixer-standard"
        )
    seen: set[tuple[str, str]] = set()
    variants, mutations = [], []
    for item in request.residue_variants:
        number, insertion = parse_residue(item.residue)
        if (number, insertion) in seen:
            raise ValueError(
                f"residue_variants lists residue {item.residue} more than once; "
                "unique source residues are required"
            )
        seen.add((number, insertion))
        variants.append(
            {"residue_id": number, "insertion_code": insertion, "variant": item.variant}
        )
    seen = set()
    for item in request.substitutions:
        number, insertion = parse_residue(item.residue)
        if (number, insertion) in seen:
            raise ValueError(
                f"substitutions lists residue {item.residue} more than once; "
                "unique source residues are required"
            )
        seen.add((number, insertion))
        if (
            item.wild_type not in RESIDUE_LETTERS
            or item.mutant not in RESIDUE_LETTERS
            or item.wild_type == item.mutant
        ):
            raise ValueError(
                f"The substitution at residue {item.residue} has wild_type "
                f"{item.wild_type!r} and mutant {item.mutant!r}; distinct standard "
                "one-letter codes are required for both"
            )
        mutations.append(
            {
                "residue_id": number,
                "insertion_code": insertion,
                "wild_type": item.wild_type,
                "mutant": item.mutant,
            }
        )
    if len(mutations) > 1000:
        raise ValueError(
            f"substitutions has {len(mutations)} entries; at most 1,000 are allowed"
        )

    config: dict[str, Any] = {
        "structure": str(path),
        "structure_format": fmt,
        "chain_id": request.chain_id,
        "model_index": request.model_index,
        "seed": request.seed,
        "ph": request.ph,
        "ionic_strength_molar": request.ionic_strength_molar,
        "minimization_max_iterations": request.minimization_max_iterations,
        "minimization_tolerance_kj_mol_nm": request.minimization_tolerance_kj_mol_nm,
        "heavy_atom_completion": request.heavy_atom_completion,
        "source_component_policy": request.source_component_policy,
        "residue_variants": variants,
        "substitutions": mutations,
        "platform": request.platform,
        "platform_properties": platform_properties(
            request.platform, request.platform_properties
        ),
    }
    if explicit:
        config.update(padding_nm=request.padding_nm, box_shape=request.box_shape)
    else:
        config.update(
            solvent_model="gbn2",
            temperature_kelvin=request.temperature_kelvin,
            solute_dielectric=request.solute_dielectric,
            solvent_dielectric=request.solvent_dielectric,
        )

    output = ctx.workdir / "prepared"
    found = ctx.run_in_environment(
        "openmm",
        "openmm_prepare:run",
        {"config": config, "output": str(output)},
        timeout=4 * 3600,
    )

    def keep(name: str, media: str) -> str:
        return str(ctx.keep(output / name, media).path)

    after = found["forces_after"]
    geometry = found["protein_geometry"]
    cell = found["periodic_cell"]
    removed = found["removed_source_atoms"]
    if removed.get("hydrogen"):
        ctx.note(
            f"{removed['hydrogen']} source hydrogens are removed and replaced by the "
            f"OpenMM placement at pH {request.ph:g}."
        )
    if geometry["inverted_alpha_residues_one_based"]:
        ctx.note(
            "These residues have inverted alpha-carbon chirality in the prepared coordinates: "
            + ", ".join(map(str, geometry["inverted_alpha_residues_one_based"]))
            + "."
        )
    worst = geometry["maximum_peptide_planarity_deviation_degrees"]
    if worst is not None and worst > 30:
        ctx.note(
            f"The largest peptide bond deviation from cis or trans planarity is {worst:.0f} "
            "degrees."
        )
    return PrepareResult(
        solvent=request.solvent,
        protein_residues=found["protein_residues"],
        protein_atoms=found["protein_atoms"],
        atom_count=found["atom_count"],
        residue_counts=found["residue_counts"],
        potential_energy_before_kj_mol=found["potential_energy_before_kj_mol"],
        potential_energy_after_kj_mol=found["potential_energy_after_kj_mol"],
        rms_force_after_kj_mol_nm=after["rms_component_kj_mol_nm"],
        max_force_after_kj_mol_nm=after["maximum_atom_kj_mol_nm"],
        box_volume_nm3=None if cell is None else cell["volume_nm3"],
        removed_source_atoms=removed,
        added_heavy_atoms=len(found["added_heavy_atoms"]),
        substitutions_applied=len(found["substitutions"]),
        max_peptide_planarity_deviation_degrees=worst,
        inverted_alpha_residues=geometry["inverted_alpha_residues_one_based"],
        system_xml=keep("system.xml", "application/xml"),
        state_xml=keep("state.xml", "application/xml"),
        topology_pdb=keep("topology.pdb", "chemical/x-pdb"),
        prepared_structure=keep("prepared.cif", "chemical/x-mmcif"),
        sequence_fasta=keep("protein.fasta", "text/x-fasta"),
        manifest=ctx.keep_json(found),
        versions=found["versions"] | {"force_field": found["protein_force_field"]},
    )
