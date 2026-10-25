"""Predict and rank candidate ligand-binding pockets on a protein surface with P2Rank.

P2Rank assigns each point of the protein's solvent-accessible surface a score from a
model trained on known ligand-binding sites, using features of the local chemistry. It
clusters the high-scoring points into pockets and ranks the pockets. Each pocket is
reported with a score, P2Rank's probability (a value from 0 to 1), the number of surface
points and atoms in it, a center in angstroms and the residues that P2Rank assigns to it. For a protein
inspected with `protein.inspect`, the residues are also mapped to positions in the
query sequence, by exact author chain and residue number.

Two configurations exist, and the request states which applies because they are
different models: `experimental` for a deposited structure, and `predicted` for a
model from a structure predictor, which omits the B-factor feature because the B-factor
column of a predicted structure holds a confidence value.

The result gives where this model, trained on known ligand-binding sites, ranks
candidate pockets on this structure, with the residues that line them.

Limits. A pocket score orders the pockets of one structure; a high rank marks a
candidate ligand site to be tested. A protein with no ligand site receives ranked
pockets as well, unless P2Rank returns none. The center is a single point; the lining
residues are listed in the result. Detection reads the structure as given, so ligands, cofactors and other chains
present in the file affect the result, and the pockets refer to the assembly in the file.
On a predicted structure the pockets include the errors of the prediction, which the
confidence values in the file describe. P2Rank runs on one thread with a 2 GB Java heap.
The result contains the `maximum_pockets` highest-ranked pockets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from alambiq.capabilities.investigate.structure_detect_binding_pockets_report import (
    render,
)
from alambiq.core import resources
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

RESOURCE = "p2rank-2.5.1"
CONFIGURATION = {"experimental": "default", "predicted": "alphafold"}
SUFFIXES = {
    ".pdb": ".pdb",
    ".ent": ".pdb",
    ".cif": ".cif",
    ".mmcif": ".cif",
    ".bcif": ".bcif",
    ".gz": None,
}


@dataclass(frozen=True, slots=True)
class PocketsRequest:
    """The structure, and which kind of structure it is.

    structure_kind:
        `experimental` or `predicted`. They select different P2Rank models.
    structure:
        A path to a PDB, mmCIF or BCIF file, which may be gzipped. Exactly one of
        `structure` and `protein` is supplied.
    protein:
        A path to a `protein.inspect` record that holds a structure, so that pocket
        residues are mapped to query positions.
    maximum_pockets:
        The number of highest-ranked pockets in the result, 1 to 50.
    """

    structure_kind: Literal["experimental", "predicted"]
    structure: str | None = None
    protein: str | None = None
    maximum_pockets: int = 20


@dataclass(frozen=True, slots=True)
class Pocket:
    """One ranked pocket."""

    rank: int
    name: str
    score: float
    probability: float
    sas_points: int
    surface_atoms: int
    center_angstrom: list[float]
    residue_ids: list[str]
    """P2Rank's residue names: author chain, underscore, author residue number and insertion code."""
    query_positions: list[int] | None
    """Query positions of the pocket's residues, or None if no protein record was given."""
    unmapped_residue_ids: list[str]


@dataclass(frozen=True, slots=True)
class PocketsResult:
    """The ranked pockets and where P2Rank's own tables are stored."""

    structure_kind: str
    configuration: str
    pockets: list[Pocket]
    residue_memberships: int
    predictions_table: str
    residues_table: str
    record: str
    versions: dict[str, str]


@capability(
    "structure.detect_binding_pockets",
    version="1.0",
    family="investigate",
    summary="Predict and rank ligand-binding pockets on a structure with P2Rank",
    render=render,
    environment="p2rank",
    resources=(RESOURCE,),
)
def detect_binding_pockets(request: PocketsRequest, ctx: StepContext) -> PocketsResult:
    """Run P2Rank on the structure and read its pockets."""
    from alambiq.capabilities.investigate.science.pockets import (
        map_pockets,
        parse_pockets,
        structure_from_dossier,
        validate_residue_table,
    )

    if (request.structure is None) == (request.protein is None):
        raise ValueError(
            "structure and protein are both set or both empty; exactly one of them is required"
        )
    if not 1 <= request.maximum_pockets <= 50:
        raise ValueError(
            f"maximum_pockets is {request.maximum_pockets}; an integer from 1 to 50 is required"
        )

    dossier = None
    if request.protein is not None:
        dossier = ctx.read_json(request.protein)
        text = structure_from_dossier(dossier)
        source = ctx.workdir / "structure.cif"
        source.write_text(text, encoding="utf-8")
        suffix = ".cif"
    else:
        original = ctx.resolve(request.structure)
        name = original.name.lower()
        compressed = name.endswith(".gz")
        stem = Path(name[:-3] if compressed else name)
        suffix = SUFFIXES.get(stem.suffix)
        if suffix is None:
            raise ValueError(
                f"The file {original.name} has an extension that P2Rank does not read; "
                "a PDB, mmCIF or BCIF file, optionally gzipped, is required"
            )
        suffix += ".gz" if compressed else ""
        source = ctx.workdir / f"structure{suffix}"
        source.write_bytes(original.read_bytes())

    root = resources.verify(RESOURCE)
    pinned = resources.spec(RESOURCE)
    configuration = CONFIGURATION[request.structure_kind]
    found = ctx.run_in_environment(
        "p2rank",
        "p2rank_predict:run",
        {
            "installation": str(root / "p2rank"),
            "structure": str(source),
            "configuration": configuration,
            "version": pinned["version"],
        },
        timeout=900,
    )
    pockets = parse_pockets(found["predictions_csv"], request.maximum_pockets)
    memberships = validate_residue_table(found["residues_csv"])
    if not pockets:
        ctx.note("P2Rank found no pocket on this structure.")
    mapped = map_pockets(dossier, pockets) if dossier is not None else None
    if mapped is not None:
        problems = sum(
            len(p["unmapped_residue_ids"]) + len(p["ambiguous_residue_ids"])
            for p in mapped
        )
        if problems:
            ctx.note(
                f"{problems} pocket residue memberships do not map exactly to a query "
                "position and are listed as unmapped."
            )
    predictions = ctx.keep_text(found["predictions_csv"], "text/csv")
    residues = ctx.keep_text(found["residues_csv"], "text/csv")
    record = {
        "tool": "P2Rank",
        "versions": found["versions"],
        "configuration": configuration,
        "structure_kind": request.structure_kind,
        "resource_identity": resources.identity(RESOURCE),
        "license": pinned["license"],
        "residue_numbering": "author-chain_author-residue-number-insertion-code",
        "center_units": "angstrom",
        "pockets": mapped if mapped is not None else pockets,
    }
    return PocketsResult(
        structure_kind=request.structure_kind,
        configuration=configuration,
        pockets=[
            Pocket(
                rank=p["rank"],
                name=p["name"],
                score=p["score"],
                probability=p["probability"],
                sas_points=p["sas_point_count"],
                surface_atoms=p["surface_atom_count"],
                center_angstrom=[
                    p["center_x_angstrom"],
                    p["center_y_angstrom"],
                    p["center_z_angstrom"],
                ],
                residue_ids=p["residue_ids"],
                query_positions=p["query_positions"] if mapped is not None else None,
                unmapped_residue_ids=p["unmapped_residue_ids"]
                if mapped is not None
                else [],
            )
            for p in (mapped if mapped is not None else pockets)
        ],
        residue_memberships=memberships,
        predictions_table=str(predictions.path),
        residues_table=str(residues.path),
        record=ctx.keep_json(record),
        versions=found["versions"],
    )
