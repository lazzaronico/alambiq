"""Identify the species of a stated reaction, check its balance and read its atom maps.

The result states whether the equation balances, which species it involves and whether
the atom maps are consistent.

Every participant is one explicit species, given as a SMILES string with a role and an
exact coefficient (an integer, a fraction such as `1/3`, or a decimal). Protons, water,
cofactors and counter-ions are participants like any other. A species is identified by
its canonical isomeric SMILES, which retains stereochemistry, charge, isotopes and
disconnected fragments. The balance in element, isotope and charge uses exact rational
arithmetic. When atom maps are supplied, the capability checks that they correspond one
to one and reads the bonds that change.

A reference structure attached to a participant is compared with the species, so a wrong
enantiomer or protonation state is reported. A database accession is taken from the
request as given. The capability reports an accession assigned to participants of
different species.

Limits:

- The result applies to the equation as stated: its species, coefficients and atom maps.
- Conditions (pH, temperature, ionic strength) are recorded as context. The balance
  depends only on the species.
- A neutral, protonated, deprotonated or tautomeric form is a different species. The
  request chooses the form.
- Bond changes are reconstructed when atom maps cover every heavy atom one to one with
  unit coefficients.
- Cahn-Ingold-Prelog labels are those that RDKit assigns to each species separately.
- A request with more than 100 participants, or with a species of more than 2,000
  explicit atoms, is refused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from alambiq.capabilities.investigate.reaction_inspect_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability

SourceKind = Literal[
    "researcher-specified",
    "curated-reference",
    "publication",
    "patent",
    "rule-generated",
    "model-predicted",
]
Role = Literal[
    "substrate",
    "product",
    "cofactor",
    "cosubstrate",
    "byproduct",
    "proton",
    "water",
    "counterion",
]


@dataclass(frozen=True, slots=True)
class ReactionSource:
    """Where the equation comes from.

    kind:
        How the equation was obtained.
    citation:
        A reference a reader can follow.
    url, version, sha256:
        Optional: a link, a release or snapshot label, and the SHA-256 of the
        source file.
    """

    kind: SourceKind
    citation: str
    url: str | None = None
    version: str | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Conditions:
    """The conditions the equation was reported under, recorded as context.

    description:
        In words, such as `aqueous, pH 7`.
    pH, temperature_K, ionic_strength_M:
        Optional numbers, within physical ranges. They do not change the balance.
    """

    description: str
    pH: float | None = None
    temperature_K: float | None = None
    ionic_strength_M: float | None = None


@dataclass(frozen=True, slots=True)
class SpeciesSource:
    """A database entry the species corresponds to.

    database, accession, version:
        What the entry is and which release it was read from.
    url:
        Optional link.
    """

    database: str
    accession: str
    version: str
    url: str | None = None


@dataclass(frozen=True, slots=True)
class Participant:
    """One species of the equation.

    id:
        Identifier of the participant, unique within the equation.
    side:
        `reactant` or `product`.
    role:
        What it does. A `substrate` must be a reactant, and a `product` or
        `byproduct` a product.
    smiles:
        One explicit species, with every hydrogen that matters (such as a proton
        on a carboxylate) written out, and atom maps where available.
    coefficient:
        An integer, a fraction such as `1/3` or a decimal, above 0 and at most
        10,000.
    sources:
        Database entries it corresponds to, up to 20.
    reference_smiles:
        A reference structure for it, checked against `smiles`.
    """

    id: str
    side: Literal["reactant", "product"]
    role: Role
    smiles: str
    coefficient: int | str = 1
    sources: list[SpeciesSource] = field(default_factory=list)
    reference_smiles: str | None = None


@dataclass(frozen=True, slots=True)
class InspectRequest:
    """The equation to account for.

    id, title:
        Identifier and short title of the reaction.
    source:
        The kind of source and the citation of the equation.
    conditions:
        The conditions under which the source reports the equation.
    participants:
        2 to 100 species, at least one on each side.
    """

    id: str
    title: str
    source: ReactionSource
    conditions: Conditions
    participants: list[Participant]


@dataclass(frozen=True, slots=True)
class Species:
    """One participant as identified from its SMILES."""

    id: str
    side: str
    role: str
    coefficient: str
    chemical_id: str
    canonical_smiles: str
    formula: str
    formal_charge: int
    drawing_svg: str


@dataclass(frozen=True, slots=True)
class Balance:
    """Products minus reactants, exactly. Anything nonzero is a residual."""

    balanced: bool
    elements: dict[str, str]
    isotopes: dict[str, str]
    charge: str


@dataclass(frozen=True, slots=True)
class BondChange:
    """A bond that is made, broken or changes order, by atom map."""

    atoms: list[int]
    reactant_order: float | None
    product_order: float | None


@dataclass(frozen=True, slots=True)
class Mapping:
    """What the supplied atom maps establish."""

    status: str
    """One of absent, partial, stoichiometric-ambiguity, inconsistent, complete-heavy-atoms."""
    bond_changes: list[BondChange]
    reactant_only: list[int]
    product_only: list[int]
    unmapped_heavy_atoms: int


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A condition of the equation flagged for review."""

    code: str
    participants: list[str]
    message: str


@dataclass(frozen=True, slots=True)
class InspectResult:
    """The equation's balance, species, atom correspondence and diagnostics."""

    reaction_id: str
    """A stable identity for the net transformation, from the species and exact stoichiometry."""
    balance: Balance
    mapping: Mapping
    species: list[Species]
    diagnostics: list[Diagnostic]
    record: str
    rdkit_version: str


@capability(
    "reaction.inspect",
    version="1.0",
    family="investigate",
    summary="Identify the species of a reaction, check its balance and read its atom maps",
    render=render,
)
def inspect(request: InspectRequest, ctx: StepContext) -> InspectResult:
    """Identify each species, balance the equation exactly and read the atom maps."""
    from alambiq.capabilities.investigate.science.reactions import (
        RDKIT_VERSION,
        inspect_reaction,
    )

    participants = []
    for p in request.participants:
        row: dict = {
            "id": p.id,
            "side": p.side,
            "role": p.role,
            "smiles": p.smiles,
            "coefficient": p.coefficient,
        }
        if p.sources:
            row["sources"] = [
                {k: v for k, v in asdict(s).items() if v is not None} for s in p.sources
            ]
        if p.reference_smiles is not None:
            row["reference_smiles"] = p.reference_smiles
        participants.append(row)
    record = inspect_reaction(
        {
            "id": request.id,
            "title": request.title,
            "convention": "explicit-species",
            "source": {
                k: v for k, v in asdict(request.source).items() if v is not None
            },
            "conditions": {
                k: v for k, v in asdict(request.conditions).items() if v is not None
            },
            "participants": participants,
        }
    )

    for item in record["diagnostics"]:
        ctx.note(f"{item['code']}: {item['message']}")
    balance, mapping = record["balance"], record["mapping"]
    return InspectResult(
        reaction_id=record["reaction_id"],
        balance=Balance(
            balanced=balance["balanced"],
            elements=balance["elements"],
            isotopes=balance["isotopes"],
            charge=balance["charge"],
        ),
        mapping=Mapping(
            status=mapping["status"],
            bond_changes=[BondChange(**change) for change in mapping["bond_changes"]],
            reactant_only=mapping["reactant_only"],
            product_only=mapping["product_only"],
            unmapped_heavy_atoms=mapping["unmapped_heavy_atoms"],
        ),
        species=[
            Species(
                id=p["id"],
                side=p["side"],
                role=p["role"],
                coefficient=p["coefficient"],
                chemical_id=p["chemical_id"],
                canonical_smiles=p["canonical_smiles"],
                formula=p["formula"],
                formal_charge=p["formal_charge"],
                drawing_svg=p["drawing_svg"],
            )
            for p in record["participants"]
        ],
        diagnostics=[Diagnostic(**d) for d in record["diagnostics"]],
        record=ctx.keep_json(record),
        rdkit_version=RDKIT_VERSION,
    )
