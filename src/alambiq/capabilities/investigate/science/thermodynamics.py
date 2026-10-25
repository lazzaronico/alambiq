"""Local aqueous component-contribution estimates with correlated route accounting."""

import copy
import hashlib
import importlib.metadata
import math
from collections import defaultdict
from fractions import Fraction
from importlib import resources
from pathlib import Path

from alambiq.capabilities.shared.records import (
    coefficient,
    digest,
    label,
    signed_coefficient,
)

PROFILE = "equilibrator-aqueous-298k-v1"
VERSIONS = {
    "equilibrator-api": "0.8.0",
    "equilibrator-cache": "0.8.1",
    "component-contribution": "0.8.0",
    "equilibrator-cache-data": "0.1.0",
}
ASSETS = {
    "mini_compounds.sqlite": "4770f5d0787e7d12183974404ab3af8ac98b964fecac0b4526453d91a50c561f",
    "cc_params.npz": "6716493d029d6cd2b0ee38b3135da546fcb4bd62e7e8d8e7e73c5a6aa9ffc4c9",
}


def local_model():
    """Load exact installed model/data assets through explicitly local readers."""
    import equilibrator_cache_data
    from component_contribution import CCModelParameters, data
    from component_contribution.predict import GibbsEnergyPredictor
    from equilibrator_api import ComponentContribution
    from equilibrator_cache import create_compound_cache_from_sqlite_file

    cache_path = Path(equilibrator_cache_data.MINI_COMPOUND_CACHE_PATH)
    params_path = Path(str(resources.files(data).joinpath("cc_params.npz")))
    identities = {}
    for path in (cache_path, params_path):
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != ASSETS[path.name]:
            raise ValueError(
                f"Installed {path.name} differs from the profile's exact asset"
            )
        identities[path.name] = {
            "sha256": checksum.hexdigest(),
            "bytes": path.stat().st_size,
        }
    # The explicit local cache and parameter constructor have no download fallback.
    cache = create_compound_cache_from_sqlite_file(
        f"{cache_path.as_uri()}?mode=ro&immutable=1&uri=true"
    )
    predictor = GibbsEnergyPredictor(parameters=CCModelParameters.from_npz(params_path))
    return ComponentContribution(ccache=cache, predictor=predictor), {
        "packages": {name: importlib.metadata.version(name) for name in VERSIONS},
        "assets": identities,
        "compound_source": "doi:10.5281/zenodo.4128543; KEGG/BiGG subset in equilibrator-cache-data 0.1.0",
        "parameter_source": "doi:10.5281/zenodo.19557433; exact bundled storage representation in component-contribution 0.8.0",
        "data_license": "CC-BY-4.0",
        "code_license": "MIT",
    }


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {
        "id",
        "conditions",
        "assumption_note",
        "compounds",
        "reactions",
        "route_weights",
    }:
        raise ValueError(
            "Supply id, conditions, assumption_note, compounds, reactions and route_weights"
        )
    label(request["id"], "id")
    label(request["assumption_note"], "assumption_note", 3000)
    conditions = request["conditions"]
    bounds = {
        "pH": (0, 14),
        "pMg": (0, 14),
        "ionic_strength_M": (0, 0.5),
        "temperature_K": (298.15, 298.15),
    }
    if not isinstance(conditions, dict) or set(conditions) != set(bounds):
        raise ValueError(
            "Conditions require pH, pMg, ionic_strength_M and temperature_K"
        )
    for key, (low, high) in bounds.items():
        value = conditions[key]
        if type(value) not in (float, int) or not low <= value <= high:
            raise ValueError(
                f"{key} requires a finite value in [{low}, {high}] for this aqueous profile"
            )
    compounds = request["compounds"]
    if not isinstance(compounds, list) or not 2 <= len(compounds) <= 200:
        raise ValueError("Supply 2–200 compound records")
    ids = set()
    for compound in compounds:
        if (
            not isinstance(compound, dict)
            or not {"id", "accession", "activity"} <= compound.keys()
            or compound.keys() - {"id", "accession", "activity", "reference_smiles"}
        ):
            raise ValueError(
                "Compounds require id, exact accession and activity, with optional reference_smiles"
            )
        for key in ("id", "accession"):
            label(compound[key], f"compound.{key}")
        if compound["id"] in ids:
            raise ValueError("Compound IDs must be distinct")
        ids.add(compound["id"])
        activity = compound["activity"]
        if activity is not None and (
            type(activity) not in (int, float) or not 1e-15 <= activity <= 10
        ):
            raise ValueError(
                "Supply a finite positive dimensionless activity in [1e-15,10]; buffered protons use null"
            )
    reactions = request["reactions"]
    if not isinstance(reactions, list) or not 1 <= len(reactions) <= 50:
        raise ValueError("Supply 1–50 reaction records")
    reaction_ids = set()
    for reaction in reactions:
        if not isinstance(reaction, dict) or set(reaction) != {
            "id",
            "stoichiometry",
            "source",
        }:
            raise ValueError("Reaction records require id, stoichiometry and source")
        label(reaction["id"], "reaction.id")
        if reaction["id"] in reaction_ids:
            raise ValueError("Reaction IDs must be distinct")
        reaction_ids.add(reaction["id"])
        source = reaction["source"]
        if not isinstance(source, dict) or set(source) != {"kind", "citation"}:
            raise ValueError("Reaction source requires kind and citation")
        if source["kind"] not in {
            "researcher-specified",
            "curated-reference",
            "publication",
            "patent",
            "rule-generated",
            "model-predicted",
        }:
            raise ValueError("Supply a supported explicit source kind")
        label(source["citation"], "source.citation", 3000)
        stoich = reaction["stoichiometry"]
        if not isinstance(stoich, dict) or not stoich or stoich.keys() - ids:
            raise ValueError("Stoichiometry must name supplied compounds")
        values = [signed_coefficient(v) for v in stoich.values()]
        if not any(v < 0 for v in values) or not any(v > 0 for v in values):
            raise ValueError(
                "Supply reactants with negative coefficients and products with positive coefficients"
            )
    weights = request["route_weights"]
    if not isinstance(weights, dict) or not weights or weights.keys() - reaction_ids:
        raise ValueError("Route weights must name supplied reaction IDs")
    for value in weights.values():
        coefficient(value)


def resolve_compounds(request, model):
    from equilibrator_cache import PROTON_INCHI_KEY, WATER_INCHI_KEY

    resolved, records, used = {}, [], set()
    for supplied in request["compounds"]:
        record = {**copy.deepcopy(supplied), "status": "unresolved-accession"}
        compound = model.get_compound(supplied["accession"])
        if compound is not None:
            if compound.inchi_key == "JLVVSXFLKOJNIY-UHFFFAOYSA-N":
                raise ValueError(
                    "This profile specifies free magnesium through pMg; supply magnesium-containing solute ensembles by their own accession"
                )
            if compound.id in used:
                raise ValueError(
                    "Use one local identity/activity for each resolved compound, including accession aliases"
                )
            used.add(compound.id)
            record.update(
                status="resolved",
                cache_id=compound.id,
                inchi_key=compound.inchi_key,
                cache_smiles=compound.smiles,
            )
            proton = compound.inchi_key == PROTON_INCHI_KEY
            if proton and supplied["activity"] is not None:
                raise ValueError(
                    "Buffered protons use activity: null; pH specifies their reservoir"
                )
            if not proton and supplied["activity"] is None:
                raise ValueError("Each solute requires an explicit activity")
            if compound.inchi_key == WATER_INCHI_KEY and supplied["activity"] != 1:
                raise ValueError("This aqueous profile uses unit water activity")
            if "reference_smiles" in supplied:
                from rdkit import Chem

                from alambiq.capabilities.investigate.science.reactions import molecule

                key = Chem.MolToInchiKey(molecule(supplied["reference_smiles"]))
                record["reference_inchi_key"] = key
                # Standard InChI's final block describes protonation; the first
                # two blocks retain molecular, stereochemical and isotope identity.
                if (
                    not key
                    or not compound.inchi_key
                    or key.rsplit("-", 1)[0] != compound.inchi_key.rsplit("-", 1)[0]
                ):
                    record["status"] = "reference-identity-mismatch"
            if record["status"] == "resolved":
                resolved[supplied["id"]] = compound
        records.append(record)
    return resolved, records


def request_from_route(network, solution):
    """Map one saved explicit-species route into supplied aqueous compound identities."""
    from alambiq.capabilities.investigate.science.routes import canonical

    if not isinstance(solution, dict) or set(solution) != {
        "id",
        "route_index",
        "conditions",
        "assumption_note",
        "compounds",
    }:
        raise ValueError(
            "Route solution requires id, route_index, conditions, assumption_note and compounds"
        )
    index = solution["route_index"]
    if type(index) is not int or not 0 <= index < len(network["routes"]):
        raise ValueError("Select an existing zero-based route_index")
    route = network["routes"][index]
    reactions = {r["id"]: r for r in network["reactions"]}
    identities = {c["chemical_id"]: c["canonical_smiles"] for c in network["compounds"]}
    selected, weights = [], defaultdict(Fraction)
    for step in route["steps"]:
        if step["reaction_id"] not in weights:
            selected.append(reactions[step["reaction_id"]])
        weights[step["reaction_id"]] += coefficient(step["extent"])
    if not selected:
        raise ValueError(
            "Select a route containing at least one chemical transformation"
        )
    mapping = {}
    if not isinstance(solution["compounds"], list) or len(solution["compounds"]) > 200:
        raise ValueError("Supply at most 200 route compound mappings")
    for entry in solution["compounds"]:
        if not isinstance(entry, dict) or set(entry) != {
            "smiles",
            "accession",
            "activity",
        }:
            raise ValueError(
                "Each route compound mapping requires smiles, accession and activity"
            )
        smiles = canonical(entry["smiles"])
        if smiles in mapping:
            raise ValueError(
                "Supply one aqueous mapping for each explicit-species identity"
            )
        mapping[smiles] = entry
    required = sorted({k for r in selected for k in r["stoichiometry"]})
    if any(identities[k] not in mapping for k in required):
        raise ValueError(
            "Supply an explicit accession/activity mapping for every route participant"
        )
    result = {
        "id": solution["id"],
        "conditions": copy.deepcopy(solution["conditions"]),
        "assumption_note": solution["assumption_note"],
        "compounds": [
            {
                "id": k,
                "accession": mapping[identities[k]]["accession"],
                "activity": mapping[identities[k]]["activity"],
                "reference_smiles": identities[k],
            }
            for k in required
        ],
        "reactions": [
            {
                "id": r["id"],
                "stoichiometry": copy.deepcopy(r["stoichiometry"]),
                "source": {
                    "kind": "rule-generated",
                    "citation": f"{r['rule_id']}: {r['rule_source']['citation']}",
                },
            }
            for r in selected
        ],
        "route_weights": {k: str(v) for k, v in weights.items()},
    }
    validate_request(result)
    return result


def assess_thermodynamics(request, model=None, model_identity=None):
    validate_request(request)
    import numpy as np
    from equilibrator_api import Q_, R, Reaction
    from equilibrator_cache import PROTON_INCHI_KEY
    from equilibrator_cache.exceptions import MissingDissociationConstantsException

    if model is None:
        model, model_identity = local_model()
    conditions = request["conditions"]
    model.p_h = Q_(conditions["pH"])
    model.p_mg = Q_(conditions["pMg"])
    model.ionic_strength = Q_(conditions["ionic_strength_M"], "M")
    model.temperature = Q_(conditions["temperature_K"], "K")
    rt = float((R * model.temperature).m_as("kJ/mol"))
    compounds, compound_records = resolve_compounds(request, model)
    activities = {c["id"]: c["activity"] for c in request["compounds"]}
    rows, estimates, estimate_indices = [], [], []
    for supplied in request["reactions"]:
        stoich = {
            k: signed_coefficient(v) for k, v in supplied["stoichiometry"].items()
        }
        row = {
            **copy.deepcopy(supplied),
            "status": "unresolved-compounds",
            "unresolved_compounds": sorted(stoich.keys() - compounds.keys()),
            "balance_residuals": {},
            "notes": [],
            "dg0_prime_kj_mol": None,
            "dg_prime_kj_mol": None,
            "model_sd_kj_mol": None,
            "ln_q": None,
            "ln_k_prime": None,
        }
        rows.append(row)
        if row["unresolved_compounds"]:
            continue
        residual = defaultdict(Fraction)
        missing_formula = []
        for key, amount in stoich.items():
            bag = compounds[key].atom_bag
            if bag is None:
                missing_formula.append(key)
                continue
            for atom, count in bag.items():
                if atom != "H":
                    residual[atom] += amount * Fraction(str(count))
        row["balance_residuals"] = {k: str(v) for k, v in sorted(residual.items()) if v}
        if row["balance_residuals"] or missing_formula:
            row["status"] = (
                "unbalanced" if not missing_formula else "unresolved-formula"
            )
            row["notes"] = [
                "Balance retains non-hydrogen elements and electrons; hydrogen exchange is buffered by pH."
            ]
            continue
        reaction = Reaction({compounds[k]: float(v) for k, v in stoich.items()})
        if all(compounds[k].inchi_key == PROTON_INCHI_KEY for k in stoich):
            row["status"] = "buffer-only-equation"
            continue
        try:
            # Upstream multi-reaction estimation catches missing transform data
            # and returns a partial result. Check each transform explicitly first.
            reaction.transform(
                p_h=model.p_h,
                p_mg=model.p_mg,
                ionic_strength=model.ionic_strength,
                temperature=model.temperature,
            )
        except (
            MissingDissociationConstantsException,
            ValueError,
            ArithmeticError,
        ) as exc:
            row.update(status="unresolved-transform", notes=[str(exc)])
            continue
        _, _, infinite, unknown = model.predictor.get_reaction_prediction(reaction)
        if unknown or np.linalg.norm(infinite) > 1e-8:
            row.update(
                status="unresolved-model",
                notes=[
                    "This equation retains an unconstrained component-contribution direction."
                ],
            )
            continue
        row["ln_q"] = sum(
            float(v) * math.log(activities[k])
            for k, v in stoich.items()
            if compounds[k].inchi_key != PROTON_INCHI_KEY
        )
        estimate_indices.append(len(rows) - 1)
        estimates.append(reaction)
    covariance = np.zeros((len(rows), len(rows)))
    if estimates:
        means, cov = model.standard_dg_prime_multi(
            estimates, uncertainty_representation="cov"
        )
        means = np.asarray(means.m_as("kJ/mol"))
        cov = np.asarray(cov.m_as("(kJ/mol)**2"))
        if not np.isfinite(means).all() or not np.isfinite(cov).all():
            raise ValueError(
                "The numerical model returned nonfinite energies or covariance"
            )
        covariance[np.ix_(estimate_indices, estimate_indices)] = cov
        for i, index in enumerate(estimate_indices):
            row = rows[index]
            row.update(
                status="estimated",
                dg0_prime_kj_mol=float(means[i]),
                dg_prime_kj_mol=float(means[i] + rt * row["ln_q"]),
                model_sd_kj_mol=math.sqrt(max(0.0, float(cov[i, i]))),
                ln_k_prime=float(-means[i] / rt),
            )
    weights = [
        coefficient(request["route_weights"][r["id"]])
        if r["id"] in request["route_weights"]
        else Fraction(0)
        for r in rows
    ]
    net = defaultdict(Fraction)
    for row, weight in zip(rows, weights, strict=True):
        for key, value in row["stoichiometry"].items():
            net[key] += weight * signed_coefficient(value)
    required = [
        row["id"]
        for row, weight in zip(rows, weights, strict=True)
        if weight and row["status"] != "estimated"
    ]
    route = {
        "status": "unresolved-steps" if required else "estimated",
        "weights": copy.deepcopy(request["route_weights"]),
        "unresolved_steps": required,
        "net_stoichiometry": {k: str(v) for k, v in sorted(net.items()) if v},
        "dg0_prime_kj_mol": None,
        "dg_prime_kj_mol": None,
        "model_sd_kj_mol": None,
        "ln_q": None,
        "ln_k_prime": None,
    }
    if not required:
        for key in ("dg0_prime_kj_mol", "dg_prime_kj_mol", "ln_q", "ln_k_prime"):
            route[key] = sum(
                float(w) * row[key] for row, w in zip(rows, weights, strict=True) if w
            )
        w = np.asarray(weights, dtype=float)
        variance = float(w @ covariance @ w)
        if variance < -1e-9:
            raise ValueError("Route covariance produced a negative variance")
        route["model_sd_kj_mol"] = math.sqrt(max(0.0, variance))
    # Null rows express unresolved estimates; a numeric zero denotes a computed covariance.
    public_cov = [
        [
            float(covariance[i, j])
            if i in estimate_indices and j in estimate_indices
            else None
            for j in range(len(rows))
        ]
        for i in range(len(rows))
    ]
    return {
        "schema_version": "0.1",
        "profile": PROFILE,
        "request": copy.deepcopy(request),
        "request_sha256": digest(request),
        "model": copy.deepcopy(model_identity),
        "rt_kj_mol": rt,
        "compounds": compound_records,
        "reactions": rows,
        "covariance_order": [row["id"] for row in rows],
        "covariance_kj2_mol2": public_cov,
        "route": route,
        "method": "Component-contribution standard transformed Gibbs energies at 298.15 K with explicit pH, pMg and ionic strength. Activities are dimensionless relative to the 1 M standard state; water has unit activity and protons exchange with the fixed-pH reservoir. Delta G prime equals Delta G zero prime plus RT ln Q. Route uncertainty uses the full shared-model covariance, conditional on the supplied activities and conditions. Standard InChI comparison retains connectivity, stereochemistry and isotopes while allowing proton-ensemble identity.",
    }
