"""Estimate the transformed Gibbs energy of reactions in aqueous solution and of a route.

For each stated reaction, with compounds identified by accession and given an activity,
this reports the standard transformed Gibbs energy ΔG′° at 298.15 K for a stated pH, pMg
and ionic strength, from eQuilibrator's component-contribution model; the energy at the
stated activities, ΔG′ = ΔG′° + RT ln Q; and ln K′. The energy of a weighted route is
the weighted sum of the reaction energies. The uncertainty of the sum uses the full
covariance matrix of the reaction estimates, so the correlation between the errors of
different reactions enters the variance of the sum. A route can instead be taken from a
stored record of `reaction.explore_routes`: the request names one route by its
zero-based index and maps each participant to an accession and an activity.

The result gives the model's estimate of whether the equation, as written and at the
stated conditions and activities, runs forward or backward, with the model's
uncertainty. Water has unit activity. The pH fixes the proton activity, so the request
gives the proton activity as null.

Limits. An estimate is a model prediction of the equilibrium position; its standard
deviation is the uncertainty of the model, conditional on the stated conditions and
activities. The estimate covers aqueous solution at 298.15 K, with magnesium given as
pMg. A reaction that cannot be estimated is reported with the reason. The statuses are
`unresolved-compounds` (an unresolved accession or a reference structure that does not
match), `unbalanced` (an unbalanced element other than hydrogen), `unresolved-formula`
(a compound without a formula), `buffer-only-equation` (all participants are protons),
`unresolved-transform` (missing dissociation data) and `unresolved-model` (a direction
that the model does not constrain). A route that uses such a reaction has no route energy. Hydrogen exchange is
fixed by the pH and is excluded from the element balance. The estimate applies to the
activities stated; activities in a cell differ from those of a dilute buffer. A request
with fewer than 2 or more than 200 compounds, or with fewer than 1 or more than 50
reactions, is refused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from alambiq.capabilities.investigate.reaction_estimate_thermodynamics_report import (
    render,
)
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


@dataclass(frozen=True, slots=True)
class AqueousConditions:
    """The solution the estimate is for.

    pH:
        0 to 14.
    pMg:
        0 to 14: the negative log of the free magnesium concentration.
    ionic_strength_M:
        0 to 0.5 molar.
    temperature_K:
        Only 298.15 is supported.
    """

    pH: float
    pMg: float
    ionic_strength_M: float
    temperature_K: float = 298.15


@dataclass(frozen=True, slots=True)
class Compound:
    """A compound of the equations, identified in the model's own database.

    id:
        Identifier of the compound in this request.
    accession:
        The identifier the model's compound database knows it by, such as
        `kegg:C00002`.
    activity:
        A dimensionless activity relative to the 1 M standard state, from 1e-15
        to 10. Protons take `null`, since pH sets them; water must be 1.
    reference_smiles:
        Optional. A reference structure to check the accession against.
    """

    id: str
    accession: str
    activity: float | None
    reference_smiles: str | None = None


@dataclass(frozen=True, slots=True)
class EquationSource:
    """Where an equation comes from."""

    kind: SourceKind
    citation: str


@dataclass(frozen=True, slots=True)
class ReactionStep:
    """One reaction as signed stoichiometry over the compounds.

    id:
        Identifier of the reaction.
    stoichiometry:
        Compound id to coefficient: negative for reactants, positive for
        products. Integers, fractions such as `1/3` or decimals.
    source:
        Where the equation comes from.
    """

    id: str
    stoichiometry: dict[str, int | str]
    source: EquationSource


@dataclass(frozen=True, slots=True)
class RouteCompound:
    """How a participant of a generated route is identified aqueously.

    smiles:
        The participant's SMILES as in the route record.
    accession:
        Its accession in the model's compound database.
    activity:
        Its activity, or `null` for protons.
    """

    smiles: str
    accession: str
    activity: float | None


@dataclass(frozen=True, slots=True)
class ThermoRequest:
    """The equations and conditions to estimate, directly or from a route.

    id:
        Identifier of this estimate.
    conditions:
        The pH, pMg and ionic strength of the solution; the temperature is fixed at
        298.15 K.
    assumption_note:
        Free text that states what the activities and conditions assume.
    compounds, reactions, route_weights:
        The equations to estimate. `route_weights` gives the number of times each
        reaction is used in the route (an integer, or a string holding a fraction or a decimal;
        above 0 and at most 10,000).
    route, route_index, route_compounds:
        A route of `reaction.explore_routes` to estimate: its stored record, the
        zero-based index of the route in it, and an aqueous identity (accession and
        activity) for every participant of the route's reactions. A request supplies
        either the first group of fields or this group, not both.
    """

    id: str
    conditions: AqueousConditions
    assumption_note: str
    compounds: list[Compound] = field(default_factory=list)
    reactions: list[ReactionStep] = field(default_factory=list)
    route_weights: dict[str, int | str] = field(default_factory=dict)
    route: str | None = None
    route_index: int | None = None
    route_compounds: list[RouteCompound] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompoundIdentity:
    """How a compound was resolved against the model's database."""

    id: str
    accession: str
    status: str
    """`resolved`, `unresolved-accession` or `reference-identity-mismatch`."""


@dataclass(frozen=True, slots=True)
class ReactionEstimate:
    """One reaction's estimate, or the reason there is none."""

    id: str
    status: str
    """`estimated`, or why not: unresolved-compounds, unbalanced, unresolved-formula,
    buffer-only-equation, unresolved-transform or unresolved-model."""
    dg0_prime_kj_mol: float | None
    dg_prime_kj_mol: float | None
    model_sd_kj_mol: float | None
    ln_q: float | None
    ln_k_prime: float | None
    unresolved_compounds: list[str]
    balance_residuals: dict[str, str]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class RouteEstimate:
    """The weighted sum of the route's reactions."""

    status: str
    """`estimated`, or `unresolved-steps` if a used reaction has no estimate."""
    weights: dict[str, Any]
    net_stoichiometry: dict[str, str]
    unresolved_steps: list[str]
    dg0_prime_kj_mol: float | None
    dg_prime_kj_mol: float | None
    model_sd_kj_mol: float | None
    ln_q: float | None
    ln_k_prime: float | None


@dataclass(frozen=True, slots=True)
class ThermoResult:
    """Every reaction's estimate, the route, and the model that made them."""

    conditions: AqueousConditions
    rt_kj_mol: float
    compounds: list[CompoundIdentity]
    reactions: list[ReactionEstimate]
    route: RouteEstimate
    record: str
    """The full record, including the covariance between reactions and the model's identity."""
    model: dict[str, Any]


@capability(
    "reaction.estimate_thermodynamics",
    version="1.0",
    family="investigate",
    summary="Estimate the transformed Gibbs energy of reactions and a route in aqueous solution",
    render=render,
)
def estimate_thermodynamics(request: ThermoRequest, ctx: StepContext) -> ThermoResult:
    """Resolve the compounds, estimate each reaction and sum the route."""
    from alambiq.capabilities.investigate.science.thermodynamics import (
        assess_thermodynamics,
        request_from_route,
    )

    conditions = asdict(request.conditions)
    direct = bool(request.compounds or request.reactions or request.route_weights)
    from_route = request.route is not None or bool(request.route_compounds)
    if direct == from_route:
        raise ValueError(
            "The request sets "
            + (
                "both the equation fields (compounds, reactions, route_weights) and the "
                "route fields (route, route_index, route_compounds)"
                if direct
                else "neither the equation fields (compounds, reactions, route_weights) "
                "nor the route fields (route, route_index, route_compounds)"
            )
            + "; exactly one of the two groups is required."
        )
    source_note = None
    if from_route:
        if (
            request.route is None
            or request.route_index is None
            or not request.route_compounds
        ):
            raise ValueError(
                "The route fields are incomplete: route, route_index and route_compounds are "
                "all required for a route estimate."
            )
        solution = {
            "id": request.id,
            "route_index": request.route_index,
            "conditions": conditions,
            "assumption_note": request.assumption_note,
            "compounds": [asdict(c) for c in request.route_compounds],
        }
        network = ctx.read_json(request.route)
        science_request = request_from_route(network, solution)
        source_note = {
            "route_id": network["routes"][request.route_index]["id"],
            "selection": solution,
        }
    else:
        science_request = {
            "id": request.id,
            "conditions": conditions,
            "assumption_note": request.assumption_note,
            "compounds": [
                {k: v for k, v in asdict(c).items() if v is not None or k == "activity"}
                for c in request.compounds
            ],
            "reactions": [asdict(r) for r in request.reactions],
            "route_weights": dict(request.route_weights),
        }
    record = assess_thermodynamics(science_request)
    if source_note is not None:
        record["route_source"] = source_note

    for row in record["reactions"]:
        if row["status"] != "estimated":
            reason = "; ".join(row["notes"]) or row["status"]
            ctx.note(f"{row['id']} has no estimate ({row['status']}): {reason}")
    route = record["route"]
    if route["status"] != "estimated":
        ctx.note(
            "The route is unresolved because these reactions have no estimate: "
            + ", ".join(route["unresolved_steps"])
        )
    return ThermoResult(
        conditions=request.conditions,
        rt_kj_mol=record["rt_kj_mol"],
        compounds=[
            CompoundIdentity(id=c["id"], accession=c["accession"], status=c["status"])
            for c in record["compounds"]
        ],
        reactions=[
            ReactionEstimate(
                id=r["id"],
                status=r["status"],
                dg0_prime_kj_mol=r["dg0_prime_kj_mol"],
                dg_prime_kj_mol=r["dg_prime_kj_mol"],
                model_sd_kj_mol=r["model_sd_kj_mol"],
                ln_q=r["ln_q"],
                ln_k_prime=r["ln_k_prime"],
                unresolved_compounds=r["unresolved_compounds"],
                balance_residuals=r["balance_residuals"],
                notes=r["notes"],
            )
            for r in record["reactions"]
        ],
        route=RouteEstimate(
            status=route["status"],
            weights=route["weights"],
            net_stoichiometry=route["net_stoichiometry"],
            unresolved_steps=route["unresolved_steps"],
            dg0_prime_kj_mol=route["dg0_prime_kj_mol"],
            dg_prime_kj_mol=route["dg_prime_kj_mol"],
            model_sd_kj_mol=route["model_sd_kj_mol"],
            ln_q=route["ln_q"],
            ln_k_prime=route["ln_k_prime"],
        ),
        record=ctx.keep_json(record),
        model=record["model"],
    )
