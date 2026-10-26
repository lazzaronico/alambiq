"""Search for routes from feedstocks to a target with reaction operators supplied in the request.

The result lists the routes found within the bounds of the request, at most `max_routes`.
Each route step carries its equation, its catalyst requirement and its enzyme citations.

Each operator is a mapped SMARTS (reactants to products, with every template atom on both
sides under the same map number). It has a source, the catalyst or processing it needs,
and optionally enzyme citations for an exact set of reactants and products. The DORAnet
package applies the operators to the feedstocks, generation by generation, within the
bounds of the request. Only equations that balance with every coproduct are retained.
The capability then follows finite inventories through the reaction network.

A route step is one application of an operator at the extent `extent_per_step`. It
consumes the amounts that it needs. A route is a sequence of route steps. It ends with at
least the target amount and returns the feedstocks named in `recover_feedstocks` to their
starting amounts.

The target status is one of:

- `inventory-route-found`: a route exists.
- `network-reachable`: the target is a species of the network, but no sequence of route
  steps ends with the target amount and restores the recovered feedstocks.
- `unreached`: the target is not reached by these operators within these bounds.

Limits:

- Each reaction of a route is an application of a supplied operator.
- A bound listed in `limits_reached` was reached. Candidates were then discarded or not
  examined, and raising the bound can add routes.
- Target matching is exact on the explicit species. A different protonation state or
  stereoisomer is a different target.
- An enzyme citation attaches to exactly the reactant and product sets it was reported
  for.
- A request with more than 30 feedstocks or more than 20 operators is refused.

`reaction.estimate_thermodynamics` estimates the transformed Gibbs energy of a route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from alambiq.capabilities.investigate.reaction_explore_routes_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class Feedstock:
    """A starting material with a finite amount.

    id:
        Identifier of the feedstock.
    smiles:
        One explicit species.
    amount:
        An integer, fraction such as `1/3` or decimal, above 0.
    role:
        `substrate`, `cosubstrate`, `cofactor`, `catalyst` or `water`.
    """

    id: str
    smiles: str
    amount: int | str
    role: Literal["substrate", "cosubstrate", "cofactor", "catalyst", "water"]


@dataclass(frozen=True, slots=True)
class Target:
    """The species to reach, and how much of it.

    smiles:
        One explicit species. Matching is exact on stereochemistry, charge and
        isotopes.
    amount:
        An integer, fraction or decimal, above 0.
    """

    smiles: str
    amount: int | str


@dataclass(frozen=True, slots=True)
class OperatorSource:
    """Where an operator comes from."""

    kind: Literal["researcher-specified", "curated-reference", "publication", "patent"]
    citation: str


@dataclass(frozen=True, slots=True)
class EnzymeEvidence:
    """An enzyme reported to catalyze exactly this equation.

    citation, enzyme:
        The citation and the enzyme.
    reactants, products:
        1 to 5 SMILES on each side. The citation attaches to a generated step only if
        these match that step's reactants and products exactly.
    """

    citation: str
    enzyme: str
    reactants: list[str]
    products: list[str]


@dataclass(frozen=True, slots=True)
class Operator:
    """A forward reaction operator.

    id, name:
        Identifier and name of the operator.
    smarts:
        A mapped reaction SMARTS with 1 to 3 reactant templates and 1 to
        5 product templates, every atom mapped on both sides. Product
        hydrogen counts and changed stereochemistry must be written explicitly.
    source:
        Where it comes from.
    catalyst_requirements:
        1 to 10 statements of what the operator needs, such as `PLP-dependent
        aminotransferase`. The statements are reported with every step the operator generates.
    enzyme_evidence:
        Up to 20 enzyme citations, each for an exact equation.
    """

    id: str
    name: str
    smarts: str
    source: OperatorSource
    catalyst_requirements: list[str]
    enzyme_evidence: list[EnzymeEvidence] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Bounds:
    """The limits of the search. Each bound has a ceiling; reaching a bound is reported.

    generations:
        Rounds of expansion, 1 to 5.
    max_recipes:
        Operator applications, 1 to 200.
    max_products_per_recipe:
        Product sets one application may produce, 1 to 50.
    max_nodes, max_edges:
        Distinct molecules and reactions retained, each 1 to 500.
    max_route_steps:
        Longest route traced, 1 to 10.
    max_route_states:
        Inventory states visited, 1 to 10,000.
    max_routes:
        Routes retained, 1 to 50.
    max_atoms:
        Largest molecule retained, 1 to 200 atoms.
    """

    generations: int = 2
    max_recipes: int = 100
    max_products_per_recipe: int = 20
    max_nodes: int = 200
    max_edges: int = 200
    max_route_steps: int = 6
    max_route_states: int = 5000
    max_routes: int = 20
    max_atoms: int = 100


@dataclass(frozen=True, slots=True)
class RoutesRequest:
    """What to search from, for what, and with which operators.

    id, title:
        Identifier and short title of the search.
    feedstocks:
        1 to 30 starting materials.
    target:
        The species to reach and its amount.
    operators:
        1 to 20 forward operators.
    extent_per_step:
        How much of each operator's equation one route step performs, as an integer,
        fraction or decimal. A smaller extent can require more route steps.
    recover_feedstocks:
        Ids of feedstocks that must return to their starting amount at the end of the
        route, so that a catalyst or cofactor is not consumed.
    bounds:
        The limits of the search, as listed in `Bounds`.
    """

    id: str
    title: str
    feedstocks: list[Feedstock]
    target: Target
    operators: list[Operator]
    extent_per_step: int | str = 1
    recover_feedstocks: list[str] = field(default_factory=list)
    bounds: Bounds = field(default_factory=Bounds)


@dataclass(frozen=True, slots=True)
class RouteStep:
    """One application of an operator in a route, at its extent."""

    reaction_id: str
    extent: str


@dataclass(frozen=True, slots=True)
class Route:
    """A sequence of route steps that reaches the target from the feedstocks."""

    id: str
    steps: list[RouteStep]
    target_amount: str
    net_stoichiometry: dict[str, str]
    """Species (by chemical id) gained or used overall."""
    balance_residuals: dict[str, str]


@dataclass(frozen=True, slots=True)
class GeneratedReaction:
    """A balanced reaction the operators produced."""

    id: str
    operator_id: str
    operator_name: str
    reactants: list[str]
    products: list[str]
    catalyst_requirements: list[str]
    enzyme_references: int
    """The number of supplied enzyme citations that match this exact equation."""


@dataclass(frozen=True, slots=True)
class Molecule:
    """A species in the network."""

    chemical_id: str
    canonical_smiles: str
    formula: str


@dataclass(frozen=True, slots=True)
class RoutesResult:
    """The routes found, the target status and the bounds that the search reached."""

    target_status: Literal["inventory-route-found", "network-reachable", "unreached"]
    target_chemical_id: str | None
    routes: list[Route]
    reactions: list[GeneratedReaction]
    molecules: list[Molecule]
    rejected_reactions: int
    limits_reached: list[str]
    """The bounds the search reached. Candidates were discarded or not examined once a listed bound was reached."""
    recipes_evaluated: int
    route_states_visited: int
    record: str
    versions: dict[str, Any]


@capability(
    "reaction.explore_routes",
    version="1.0",
    family="investigate",
    summary="Search for routes to a target with supplied reaction operators",
    render=render,
)
def explore_routes(request: RoutesRequest, ctx: StepContext) -> RoutesResult:
    """Expand the operators from the feedstocks and trace inventory routes."""
    from alambiq.capabilities.investigate.science.routes import (
        explore_routes as search,
    )

    record = search(
        {
            "id": request.id,
            "title": request.title,
            "feedstocks": [asdict(f) for f in request.feedstocks],
            "target": asdict(request.target),
            "extent_per_step": request.extent_per_step,
            "recover_feedstocks": list(request.recover_feedstocks),
            "rules": [
                {
                    "id": o.id,
                    "name": o.name,
                    "smarts": o.smarts,
                    "source": asdict(o.source),
                    "catalyst_requirements": list(o.catalyst_requirements),
                    "enzyme_evidence": [asdict(e) for e in o.enzyme_evidence],
                }
                for o in request.operators
            ],
            "bounds": asdict(request.bounds),
        }
    )

    if record["limits_reached"]:
        ctx.note(
            "The search stopped at these bounds, so it has not finished: "
            + ", ".join(record["limits_reached"])
            + "."
        )
    if record["target"]["status"] == "unreached":
        ctx.note(
            "The target is unreached by these operators from these feedstocks within "
            "these bounds."
        )
    elif record["target"]["status"] == "network-reachable":
        ctx.note(
            "The target appears in the network, but no sequence of steps closes the "
            "inventory at this extent."
        )
    return RoutesResult(
        target_status=record["target"]["status"],
        target_chemical_id=record["target"]["chemical_id"],
        routes=[
            Route(
                id=r["id"],
                steps=[RouteStep(**s) for s in r["steps"]],
                target_amount=r["target_amount"],
                net_stoichiometry=r["net_stoichiometry"],
                balance_residuals=r["balance_residuals"],
            )
            for r in record["routes"]
        ],
        reactions=[
            GeneratedReaction(
                id=r["id"],
                operator_id=r["rule_id"],
                operator_name=r["rule_name"],
                reactants=r["reactants"],
                products=r["products"],
                catalyst_requirements=r["catalyst_requirements"],
                enzyme_references=len(r["enzyme_evidence"]),
            )
            for r in record["reactions"]
        ],
        molecules=[
            Molecule(
                chemical_id=c["chemical_id"],
                canonical_smiles=c["canonical_smiles"],
                formula=c["formula"],
            )
            for c in record["compounds"]
        ],
        rejected_reactions=len(record["rejected_reactions"]),
        limits_reached=record["limits_reached"],
        recipes_evaluated=record["recipes_evaluated"],
        route_states_visited=record["route_states_visited"],
        record=ctx.keep_json(record),
        versions=record["backend"],
    )
