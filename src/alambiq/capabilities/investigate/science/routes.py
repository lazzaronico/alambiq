"""DORAnet forward expansion with complete equations and finite-feed route traces."""

import copy
import importlib.metadata
from collections import Counter, defaultdict, deque
from fractions import Fraction
from functools import lru_cache

from alambiq.capabilities.investigate.science.reactions import (
    chemistry,
    inspect_compound,
    molecule,
    rdkit_version,
)
from alambiq.capabilities.shared.records import coefficient, digest, label

PROFILE = "doranet-explicit-forward-v1"
DORANET_VERSION = "0.5.7a1"
BOUND_MAXIMA = {
    "generations": 5,
    "max_recipes": 200,
    "max_products_per_recipe": 50,
    "max_nodes": 500,
    "max_edges": 500,
    "max_route_steps": 10,
    "max_route_states": 10000,
    "max_routes": 50,
    "max_atoms": 200,
}


def canonical(smiles):
    Chem = chemistry()
    mol = molecule(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)


@lru_cache(maxsize=4096)
def composition(smiles):
    Chem = chemistry()
    mol = molecule(smiles)
    result = Counter()
    for atom in Chem.AddHs(mol).GetAtoms():
        result[atom.GetSymbol()] += 1
        if atom.GetIsotope():
            result[f"isotope:{atom.GetIsotope()}{atom.GetSymbol()}"] += 1
    result["charge"] = Chem.GetFormalCharge(mol)
    result["radical_electrons"] = sum(
        a.GetNumRadicalElectrons() for a in mol.GetAtoms()
    )
    return result


def balance(stoichiometry):
    residual = defaultdict(Fraction)
    for smiles, amount in stoichiometry.items():
        for element, count in composition(smiles).items():
            residual[element] += amount * count
    return {k: str(v) for k, v in sorted(residual.items()) if v}


def equation(reactants, products):
    net = Counter(products)
    net.subtract(Counter(reactants))
    return {k: v for k, v in sorted(net.items()) if v}


def validate_request(request, *, chemical_checks=True):
    if chemical_checks:
        from rdkit.Chem import rdChemReactions

    if not isinstance(request, dict) or set(request) != {
        "id",
        "title",
        "feedstocks",
        "target",
        "extent_per_step",
        "recover_feedstocks",
        "rules",
        "bounds",
    }:
        raise ValueError(
            "Supply id, title, feedstocks, target, extent_per_step, rules and bounds"
        )
    label(request["id"], "id")
    label(request["title"], "title", 500)
    coefficient(request["extent_per_step"])
    bounds = request["bounds"]
    if not isinstance(bounds, dict) or set(bounds) != set(BOUND_MAXIMA):
        raise ValueError("Supply every expansion and route-trace bound")
    for key, maximum in BOUND_MAXIMA.items():
        if type(bounds[key]) is not int or not 1 <= bounds[key] <= maximum:
            raise ValueError(f"{key} requires an integer in [1,{maximum}]")
    feeds = request["feedstocks"]
    if not isinstance(feeds, list) or not 1 <= len(feeds) <= 30:
        raise ValueError("Supply 1–30 explicit feedstocks")
    ids, identities = set(), set()
    for feed in feeds:
        if not isinstance(feed, dict) or set(feed) != {
            "id",
            "smiles",
            "amount",
            "role",
        }:
            raise ValueError("Each feedstock requires id, smiles, amount and role")
        label(feed["id"], "feedstock.id")
        coefficient(feed["amount"])
        label(feed["smiles"], "feedstock.smiles", 20000)
        identity = canonical(feed["smiles"]) if chemical_checks else feed["smiles"]
        if feed["id"] in ids or identity in identities:
            raise ValueError(
                "Each feedstock requires a distinct ID and chemical identity"
            )
        ids.add(feed["id"])
        identities.add(identity)
        if feed["role"] not in {
            "substrate",
            "cosubstrate",
            "cofactor",
            "catalyst",
            "water",
        }:
            raise ValueError(
                "Use substrate, cosubstrate, cofactor, catalyst or water as feedstock role"
            )
        if chemical_checks and molecule(identity).GetNumAtoms() > bounds["max_atoms"]:
            raise ValueError("Each feedstock must fit the explicit-atom bound")
    if len(identities) > bounds["max_nodes"]:
        raise ValueError("max_nodes must accommodate the supplied feedstocks")
    recovery = request["recover_feedstocks"]
    if (
        not isinstance(recovery, list)
        or any(not isinstance(k, str) for k in recovery)
        or len(set(recovery)) != len(recovery)
        or set(recovery) - ids
    ):
        raise ValueError(
            "recover_feedstocks names distinct supplied feedstock IDs to restore at the route endpoint"
        )
    target = request["target"]
    if not isinstance(target, dict) or set(target) != {"smiles", "amount"}:
        raise ValueError("Target requires smiles and amount")
    label(target["smiles"], "target.smiles", 20000)
    coefficient(target["amount"])
    if chemical_checks:
        canonical(target["smiles"])
    rules = request["rules"]
    if not isinstance(rules, list) or not 1 <= len(rules) <= 20:
        raise ValueError("Supply 1–20 explicit forward SMARTS operators")
    ids, operators = set(), set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {
            "id",
            "name",
            "smarts",
            "source",
            "catalyst_requirements",
            "enzyme_evidence",
        }:
            raise ValueError(
                "Each rule requires id, name, smarts, source, catalyst_requirements and enzyme_evidence"
            )
        for key in ("id", "name"):
            label(rule[key], f"rule.{key}", 500)
        label(rule["smarts"], "rule.smarts", 12000)
        source = rule["source"]
        if (
            not isinstance(source, dict)
            or set(source) != {"kind", "citation"}
            or source["kind"]
            not in {
                "researcher-specified",
                "curated-reference",
                "publication",
                "patent",
            }
        ):
            raise ValueError("Rule source requires a supported kind and citation")
        label(source["citation"], "rule.source.citation", 3000)
        requirements = rule["catalyst_requirements"]
        if not isinstance(requirements, list) or not 1 <= len(requirements) <= 10:
            raise ValueError("Supply 1–10 explicit catalyst or processing requirements")
        for item in requirements:
            label(item, "catalyst requirement", 1000)
        evidence = rule["enzyme_evidence"]
        if not isinstance(evidence, list) or len(evidence) > 20:
            raise ValueError(
                "enzyme_evidence is a list of up to 20 exact-equation reference records"
            )
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {
                "citation",
                "enzyme",
                "reactants",
                "products",
            }:
                raise ValueError(
                    "Enzyme evidence requires citation, enzyme, reactants and products"
                )
            label(item["citation"], "evidence.citation", 3000)
            label(item["enzyme"], "evidence.enzyme", 1000)
            for side in ("reactants", "products"):
                if not isinstance(item[side], list) or not 1 <= len(item[side]) <= 5:
                    raise ValueError(
                        "Enzyme evidence includes 1–5 species on each equation side"
                    )
                for smiles in item[side]:
                    label(smiles, "evidence.smiles", 20000)
                    if chemical_checks:
                        canonical(smiles)
        if not chemical_checks:
            if rule["id"] in ids or rule["smarts"] in operators:
                raise ValueError("Rule IDs and operators must be distinct")
            ids.add(rule["id"])
            operators.add(rule["smarts"])
            continue
        rxn = rdChemReactions.ReactionFromSmarts(rule["smarts"])
        if (
            rxn is None
            or rxn.GetNumAgentTemplates()
            or not 1 <= rxn.GetNumReactantTemplates() <= 3
            or not 1 <= rxn.GetNumProductTemplates() <= 5
        ):
            raise ValueError(
                "Operators require 1–3 reactants and 1–5 products, with catalyst requirements stated separately"
            )
        maps = []
        for templates in (rxn.GetReactants(), rxn.GetProducts()):
            atom_maps = [a.GetAtomMapNum() for mol in templates for a in mol.GetAtoms()]
            if (
                not atom_maps
                or len(atom_maps) > 500
                or 0 in atom_maps
                or len(set(atom_maps)) != len(atom_maps)
            ):
                raise ValueError(
                    "Every operator-template atom requires a unique positive map"
                )
            maps.append(set(atom_maps))
        if maps[0] != maps[1] or rxn.Validate(silent=True)[1]:
            raise ValueError(
                "Every mapped operator atom must occur on both equation sides"
            )
        left_atoms = {
            a.GetAtomMapNum(): a for mol in rxn.GetReactants() for a in mol.GetAtoms()
        }
        for mol in rxn.GetProducts():
            for atom in mol.GetAtoms():
                original = left_atoms[atom.GetAtomMapNum()]
                if (
                    atom.GetAtomicNum()
                    and original.GetAtomicNum()
                    and atom.GetAtomicNum() != original.GetAtomicNum()
                ):
                    raise ValueError(
                        "Mapped operator atoms retain their element identity"
                    )
                if atom.GetIsotope() and original.GetIsotope() != atom.GetIsotope():
                    raise ValueError(
                        "Mapped operator atoms retain their explicit isotope identity"
                    )
        operator = rdChemReactions.ReactionToSmarts(rxn)
        if rule["id"] in ids or operator in operators:
            raise ValueError(
                "Use one distinct rule per operator and combine its reference evidence"
            )
        ids.add(rule["id"])
        operators.add(operator)


def trace_routes(request, compounds, reactions, limits):
    """Trace finite inventories through the generated graph at a declared extent."""
    by_smiles = {m["canonical_smiles"]: m["chemical_id"] for m in compounds}
    by_id = {m["chemical_id"]: m["canonical_smiles"] for m in compounds}
    target = by_smiles.get(canonical(request["target"]["smiles"]))
    if target is None:
        return [], 0
    initial = {
        by_smiles[canonical(f["smiles"])]: coefficient(f["amount"])
        for f in request["feedstocks"]
    }
    needed = coefficient(request["target"]["amount"])
    recovered = {
        by_smiles[canonical(f["smiles"])]: coefficient(f["amount"])
        for f in request["feedstocks"]
        if f["id"] in request["recover_feedstocks"]
    }
    extent = coefficient(request["extent_per_step"])
    bounds = request["bounds"]
    pending = deque([(initial, [])])
    seen = {(tuple(sorted(initial.items())), ())}
    routes = []
    visited = 0
    while pending:
        if visited >= bounds["max_route_states"]:
            limits.add("max_route_states")
            break
        inventory, steps = pending.popleft()
        visited += 1
        if inventory.get(target, 0) >= needed and all(
            inventory.get(k, 0) >= v for k, v in recovered.items()
        ):
            net = {
                k: inventory.get(k, Fraction(0)) - initial.get(k, Fraction(0))
                for k in set(inventory) | set(initial)
            }
            net = {k: v for k, v in sorted(net.items()) if v}
            routes.append(
                {
                    "id": "route-"
                    + digest(
                        {
                            "steps": steps,
                            "initial": {k: str(v) for k, v in sorted(initial.items())},
                        }
                    ),
                    "steps": steps,
                    "net_stoichiometry": {k: str(v) for k, v in net.items()},
                    "initial_inventory": {
                        k: str(v) for k, v in sorted(initial.items())
                    },
                    "final_inventory": {
                        k: str(v) for k, v in sorted(inventory.items()) if v
                    },
                    "target_amount": str(inventory[target]),
                    "balance_residuals": balance({by_id[k]: v for k, v in net.items()}),
                }
            )
            if len(routes) >= bounds["max_routes"]:
                if pending:
                    limits.add("max_routes")
                break
            continue
        if len(steps) >= bounds["max_route_steps"]:
            limits.add("max_route_steps")
            continue
        for reaction in reactions:
            required = Counter(reaction["reactants"])
            if any(inventory.get(k, 0) < v * extent for k, v in required.items()):
                continue
            updated = defaultdict(Fraction, inventory)
            for key, amount in reaction["stoichiometry"].items():
                updated[key] += Fraction(amount) * extent
            updated = {k: v for k, v in updated.items() if v}
            next_steps = steps + [
                {"reaction_id": reaction["id"], "extent": str(extent)}
            ]
            # Equivalent permutations retain their first reproducible ordering.
            state_key = (
                tuple(sorted(updated.items())),
                tuple(sorted(Counter(s["reaction_id"] for s in next_steps).items())),
            )
            if state_key in seen:
                continue
            if len(seen) >= bounds["max_route_states"]:
                limits.add("max_route_states")
                continue
            seen.add(state_key)
            pending.append((updated, next_steps))
    return routes, visited


def explore_routes(request):
    validate_request(request)
    observed = importlib.metadata.version("doranet")
    import doranet as dn
    from doranet.datatypes import OpDatBasic
    from doranet.metadata import ReactionFilterBase

    Chem = chemistry()
    bounds = request["bounds"]
    limits, rejected, operator_events = set(), [], []
    engine = dn.create_engine()
    network = engine.new_network()
    known = {canonical(f["smiles"]) for f in request["feedstocks"]}
    for smiles in sorted(known):
        network.add_mol(engine.mol.rdkit(smiles))
    rules = sorted(request["rules"], key=lambda r: r["id"])
    accepted = set()
    calls = 0
    rounds = 0
    candidates = 0
    candidate_limit = bounds["max_recipes"] * bounds["max_nodes"]

    class CandidateBudgetReached(Exception):
        pass

    class CandidateBudget(dn.interfaces.RecipeFilter):
        @property
        def meta_required(self):
            return dn.interfaces.MetaKeyPacket()

        def __call__(self, recipe):
            nonlocal candidates
            if candidates >= candidate_limit:
                raise CandidateBudgetReached
            candidates += 1
            return True

    class BoundedOperator(OpDatBasic):
        """Apply DORAnet's public RDKit operator interface with an outcome cap."""

        def __call__(self, *reactants):
            nonlocal calls
            calls += 1
            if calls >= bounds["max_recipes"]:
                limits.add("max_recipes")
            # One extra outcome establishes that a bounded enumeration was truncated.
            results = self.rdkitrxn.RunReactants(
                tuple(r.rdkitmol for r in reactants),
                maxProducts=bounds["max_products_per_recipe"] + 1,
            )
            if len(results) > bounds["max_products_per_recipe"]:
                limits.add("max_products_per_recipe")
                operator_events.append(
                    {
                        "smarts": self.smarts,
                        "reactants": [r.uid for r in reactants],
                        "status": "product-enumeration-truncated",
                    }
                )
            generated = []
            for products in results[: bounds["max_products_per_recipe"]]:
                try:
                    generated.append(tuple(engine.mol.rdkit(mol) for mol in products))
                except (ValueError, RuntimeError) as exc:
                    operator_events.append(
                        {
                            "smarts": self.smarts,
                            "reactants": [r.uid for r in reactants],
                            "products": [Chem.MolToSmiles(mol) for mol in products],
                            "status": "product-sanitization-failed",
                            "diagnostic": str(exc),
                        }
                    )
            return tuple(generated)

    for rule in rules:
        operator = BoundedOperator(rule["smarts"], engine)
        operator.rdkitrxn.GetSubstructParams().useChirality = True
        network.add_op(operator)

    class CompleteEquation(ReactionFilterBase):
        def __call__(self, recipe):
            reactants = [canonical(r.item.uid) for r in recipe.reactants]
            products = [canonical(p.item.uid) for p in recipe.products]
            net = equation(reactants, products)
            residual = balance(net)
            key = (recipe.operator.i, tuple(sorted(reactants)), tuple(sorted(products)))
            reason = None
            if residual:
                reason = "unbalanced-equation"
            elif not net:
                reason = "zero-net-equation"
            elif any(molecule(s).GetNumAtoms() > bounds["max_atoms"] for s in products):
                reason = "max_atoms"
                limits.add(reason)
            elif len(known | set(products)) > bounds["max_nodes"]:
                reason = "max_nodes"
                limits.add(reason)
            elif key not in accepted and len(accepted) >= bounds["max_edges"]:
                reason = "max_edges"
                limits.add(reason)
            if reason:
                rejected.append(
                    {
                        "rule_id": rules[recipe.operator.i]["id"],
                        "reactants": reactants,
                        "products": products,
                        "status": reason,
                        "balance_residuals": residual,
                    }
                )
                return False
            known.update(products)
            accepted.add(key)
            return True

    class GenerationBudget(dn.interfaces.GlobalUpdateHook):
        def __call__(self, network):
            nonlocal rounds
            rounds += 1
            if rounds >= bounds["generations"]:
                limits.add("generations")
                return dn.interfaces.GlobalHookReturnValue.STOP
            return dn.interfaces.GlobalHookReturnValue.CONTINUE

    network_strategy = engine.strat.pq(network)
    try:
        network_strategy.expand(
            max_recipes=bounds["max_recipes"],
            heap_size=bounds["max_recipes"],
            beam_size=None,
            batch_size=bounds["max_recipes"],
            global_hooks=[GenerationBudget()],
            recipe_filter=CandidateBudget(),
            reaction_plan=CompleteEquation(),
            save_unreactive=False,
        )
    except CandidateBudgetReached:
        # Retain completed generations; the in-progress ranking batch is incomplete.
        limits.add("recipe_candidates")
    compounds = [inspect_compound(s) for s in sorted(known)]
    compound_ids = {c["canonical_smiles"]: c["chemical_id"] for c in compounds}
    reactions = {}
    for rxn in network.rxns:
        rule = rules[rxn.operator]
        reactants = sorted(canonical(network.mols[i].uid) for i in rxn.reactants)
        products = sorted(canonical(network.mols[i].uid) for i in rxn.products)
        identifier = "reaction-" + digest(
            {"rule_id": rule["id"], "reactants": reactants, "products": products}
        )
        evidence = [
            copy.deepcopy(e)
            for e in rule["enzyme_evidence"]
            if sorted(canonical(s) for s in e["reactants"]) == reactants
            and sorted(canonical(s) for s in e["products"]) == products
        ]
        reactions[identifier] = {
            "id": identifier,
            "rule_id": rule["id"],
            "rule_name": rule["name"],
            "origin": "rule-generated",
            "rule_source": copy.deepcopy(rule["source"]),
            "reactants": [compound_ids[s] for s in reactants],
            "products": [compound_ids[s] for s in products],
            "stoichiometry": {
                compound_ids[s]: str(v)
                for s, v in equation(reactants, products).items()
            },
            "balance_residuals": {},
            "catalyst_requirements": copy.deepcopy(rule["catalyst_requirements"]),
            "enzyme_evidence": evidence,
        }
    reaction_records = [reactions[k] for k in sorted(reactions)]
    routes, visited = trace_routes(request, compounds, reaction_records, limits)
    target_smiles = canonical(request["target"]["smiles"])
    return {
        "schema_version": "0.1",
        "profile": PROFILE,
        "request": copy.deepcopy(request),
        "request_sha256": digest(request),
        "backend": {"doranet": observed, "rdkit": rdkit_version()},
        "rule_set_sha256": digest(request["rules"]),
        "compounds": compounds,
        "reactions": reaction_records,
        "rejected_reactions": rejected,
        "operator_events": operator_events,
        "routes": routes,
        "target": {
            "smiles": target_smiles,
            "chemical_id": compound_ids.get(target_smiles),
            "status": "inventory-route-found"
            if routes
            else "network-reachable"
            if target_smiles in known
            else "unreached",
        },
        "limits_reached": sorted(limits),
        "recipes_evaluated": calls,
        "recipe_candidates_considered": candidates,
        "recipe_candidate_limit": candidate_limit,
        "expansion_rounds": rounds,
        "route_states_visited": visited,
        "method": "Forward DORAnet bounded-batch expansion applies the supplied mapped operators within explicit generation, recipe, product, molecule and reaction bounds. The recipe heap and batch use max_recipes as their size limit; candidate ranking is capped at max_recipes times max_nodes and retains completed generations when exhausted. Balanced products preserve stereochemistry, isotopes, charge and every coproduct. Finite-feed breadth-first traces apply the declared extent per step, require selected feedstocks to be restored at the endpoint, and retain the first ordering of equivalent inventory/step-count states. Catalyst requirements and exact-equation enzyme references accompany each generated step. Target matching uses exact explicit-species identity. The resulting alternatives describe the supplied rules, feeds, extent and search bounds.",
    }
