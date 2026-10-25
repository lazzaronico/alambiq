"""Turn measured Michaelis-Menten kinetics into the rate parameters of an operating definition.

An operating definition (see `simulation.py`) states each rate channel's capacity in M/s
at its catalyst's reference amount, full activity and the initial volume, and the
concentration at which a saturation factor is half-way. A fit from an assay states a
volumetric rate and a Km for the enzyme concentration that assay used. This module converts
between them.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from alambiq.capabilities.measure.science.balance import kinetics
from alambiq.capabilities.operate.science.simulation import validate_request

MASS = {"kg": 1e3, "g": 1.0, "mg": 1e-3, "ug": 1e-6, "ng": 1e-9}
VOLUME = {"L": 1.0, "dL": 1e-1, "mL": 1e-3, "uL": 1e-6}


def per_litre(value: float, unit: str, catalyst_unit: str) -> float:
    """An enzyme concentration in the catalyst's own amount unit per liter.

    Mass per volume converts when the catalyst is counted in a mass; any other catalyst
    unit (an activity unit, moles) has to be stated per liter in that same unit.
    """
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "The assay enzyme concentration must be a finite number above 0."
        )
    amount, _, volume = unit.partition("/")
    if volume not in VOLUME:
        raise ValueError(
            f"The assay enzyme unit {unit!r} must be an amount per volume, for example "
            f"mg/mL or {catalyst_unit}/L."
        )
    if amount == catalyst_unit:
        return value / VOLUME[volume]
    if amount in MASS and catalyst_unit in MASS:
        return value * MASS[amount] / MASS[catalyst_unit] / VOLUME[volume]
    raise ValueError(
        f"The assay enzyme unit {unit!r} cannot be converted to the unit "
        f"{catalyst_unit!r} of the catalyst. State the concentration in "
        f"{catalyst_unit}/L."
    )


def _one(rows: list[dict], key: str, value: str, what: str) -> dict:
    found = [row for row in rows if row[key] == value]
    if len(found) != 1:
        raise ValueError(f"The template has no {what} {value!r}")
    return found[0]


def _rational_pair(definition, equilibrium):
    """The forward and reverse channels of a reversible pair, checked to share one form.

    Both carry a mass-action factor on their own substrate and the same denominator
    `1 + S/Ks + P/Kp`, so that setting Ks, Kp and the two capacities sets the
    reversible Michaelis-Menten rate law.
    """
    forward = _one(definition["channels"], "id", equilibrium["forward"], "channel")
    reverse = _one(definition["channels"], "id", equilibrium["reverse"], "channel")
    substrate, product = equilibrium["substrate"], equilibrium["product"]
    for channel, source, target in (
        (forward, substrate, product),
        (reverse, product, substrate),
    ):
        stoichiometry = channel["stoichiometry"]
        if (
            float(stoichiometry.get(source, 0)) >= 0
            or float(stoichiometry.get(target, 0)) <= 0
            or "catalyst" not in channel
        ):
            raise ValueError(
                f"Channel {channel['id']!r} must consume {source!r}, produce {target!r} "
                "and name a catalyst for a reversible pair."
            )
        factor = next((f for f in channel["factors"] if f["species"] == source), None)
        terms = (channel.get("denominator") or {}).get("terms", [])
        species = {part["species"] for term in terms for part in term["factors"]}
        if (
            factor is None
            or factor["kind"] != "mass-action"
            or factor.get("power", 1) != 1
            or species != {substrate, product}
            or len(terms) != 2
            or channel["denominator"]["constant"] != 1
        ):
            raise ValueError(
                f"Channel {channel['id']!r} must be a mass-action factor on {source!r} "
                f"over the denominator 1 + {substrate}/Ks + {product}/Kp for a "
                "reversible pair."
            )
    if forward["catalyst"] != reverse["catalyst"]:
        raise ValueError(
            "The forward and reverse channels of a reversible pair must name the same "
            "catalyst."
        )
    return forward, reverse


def _set_scales(channel, scales):
    """Write each species' scale into a channel's factors and denominator terms."""
    for factor in channel["factors"]:
        if factor["species"] in scales:
            factor["scale_M"] = scales[factor["species"]]
    for term in channel["denominator"]["terms"]:
        for part in term["factors"]:
            part["scale_M"] = scales[part["species"]]


TIME_SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0}


def _fit_parameters(fit: dict[str, Any], model: str, what: str) -> dict[str, float]:
    """The coefficients of a stored `assay.fit` record, checked to be the model needed."""
    if (
        not isinstance(fit, dict)
        or fit.get("model") != model
        or fit.get("optimizer", {}).get("success") is not True
    ):
        raise ValueError(f"{what} requires a converged {model} fit from assay.fit.")
    return {p["name"]: float(p["value"]) for p in fit["parameters"]}


def _inactivate(definition, item):
    """Set a catalyst's deactivation rate, and its temperature dependence, from fits."""
    catalyst = _one(definition["catalysts"], "id", item["catalyst"], "catalyst")
    decay = _fit_parameters(item["decay_fit"], "first-order-decay-v1", "Inactivation")
    unit = item["decay_fit"]["request"]["x_unit"]
    if unit not in TIME_SECONDS:
        raise ValueError(
            "A decay fit for inactivation requires x_unit to be s, min or h."
        )
    rate = decay["k"] / TIME_SECONDS[unit]
    catalyst["deactivation_per_s"] = rate
    row = {
        "catalyst": catalyst["id"],
        "deactivation_per_s": rate,
        "half_life_s": math.log(2.0) / rate if rate > 0 else None,
        "decay_temperature_K": item["decay_temperature_K"],
        "activation_energy_J_mol": None,
    }
    if item.get("temperature_fit") is not None:
        arrhenius = _fit_parameters(
            item["temperature_fit"],
            "temperature-arrhenius-v1",
            "Inactivation temperature",
        )
        catalyst["deactivation_temperature"] = {
            "activation_energy_J_mol": arrhenius["Ea"],
            "reference_K": item["decay_temperature_K"],
        }
        row["activation_energy_J_mol"] = arrhenius["Ea"]
    return row


def define(
    template: dict[str, Any],
    entries: list[dict[str, Any]],
    equilibria: list[dict[str, Any]] | None = None,
    inactivations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fill the named channels of a template from fits; return the definition and a log."""
    validate_request(template)
    definition = copy.deepcopy(template)
    seen: set[str] = set()
    log: list[dict[str, Any]] = []
    pairs = {e["forward"]: e for e in equilibria or []}
    reversible: dict[str, dict[str, Any]] = {}
    for equilibrium in equilibria or []:
        if equilibrium["forward"] == equilibrium["reverse"]:
            raise ValueError(
                "The forward and reverse channels of a reversible pair must differ."
            )
        forward, reverse = _rational_pair(definition, equilibrium)
        reversible[equilibrium["forward"]] = {
            **equilibrium,
            "forward": forward,
            "reverse": reverse,
        }
    for entry in entries:
        name = entry["channel"]
        if name in seen:
            raise ValueError(f"Channel {name!r} is listed twice in kinetics.")
        seen.add(name)
        channel = _one(definition["channels"], "id", name, "channel")
        if "catalyst" not in channel:
            raise ValueError(
                f"Channel {name!r} has no catalyst; kinetics measured per amount of "
                "enzyme require a catalyst to scale to."
            )
        catalyst = _one(definition["catalysts"], "id", channel["catalyst"], "catalyst")
        species = entry["species"]
        coefficient = channel["stoichiometry"].get(species)
        if coefficient is None or float(coefficient) >= 0:
            raise ValueError(
                f"{species!r} is not consumed by channel {name!r}; species must name "
                "the substrate that the assay measured."
            )
        factor = next((f for f in channel["factors"] if f["species"] == species), None)
        if name in pairs:
            if species != pairs[name]["substrate"]:
                raise ValueError(
                    f"Channel {name!r} is the forward channel of a reversible pair on "
                    f"{pairs[name]['substrate']!r}, not {species!r}."
                )
        elif (
            factor is None
            or factor["kind"] != "saturation"
            or factor.get("power", 1) != 1
            or "denominator" in channel
        ):
            raise ValueError(
                f"Channel {name!r} must have a saturation factor of power 1 on "
                f"{species!r} and no denominator to receive the Km and the capacity of "
                "a Michaelis-Menten fit."
            )
        fraction = entry["assay_active_fraction"]
        if not 0 < fraction <= 1:
            raise ValueError(
                f"assay_active_fraction of channel {name!r} must be above 0 and at most "
                f"1; found {fraction:g}."
            )
        km, substrate_vmax, _unit, low, high, _ratio = kinetics(entry["fit"])
        if km <= 0:
            raise ValueError(
                f"A saturation factor requires a positive Km; the fit for channel "
                f"{name!r} gives {km:g} M."
            )
        enzyme = per_litre(
            entry["assay_enzyme"]["value"],
            entry["assay_enzyme"]["unit"],
            catalyst["amount_unit"],
        )
        if entry.get("temperature_fit") is not None:
            if entry.get("assay_temperature_K") is None:
                raise ValueError(
                    f"Channel {name!r}: assay_temperature_K, the temperature of the "
                    "kinetics assay, must be set when temperature_fit is given."
                )
            arrhenius = _fit_parameters(
                entry["temperature_fit"],
                "temperature-arrhenius-v1",
                "A temperature fit",
            )
            channel["temperature"] = {
                "activation_energy_J_mol": arrhenius["Ea"],
                "reference_K": entry["assay_temperature_K"],
            }
        if entry.get("ph_fit") is not None:
            if entry.get("assay_pH") is None:
                raise ValueError(
                    f"Channel {name!r}: assay_pH, the pH of the kinetics assay, must be "
                    "set when ph_fit is given."
                )
            bell = _fit_parameters(entry["ph_fit"], "ph-bell-v1", "A pH fit")
            channel["ph"] = {
                "pKa1": bell["pKa1"],
                "pKa2": bell["pKa2"],
                "reference_pH": entry["assay_pH"],
            }
        extent_rate = substrate_vmax / abs(float(coefficient))
        per_enzyme = extent_rate / fraction / enzyme
        capacity = per_enzyme * catalyst["reference_amount"] / definition["volume_L"]
        if name in pairs:
            pair = reversible[name]
            ks, kp = km, pair["product_km_M"]
            ratio = pair["equilibrium_constant"]
            _set_scales(pair["forward"], {pair["substrate"]: ks, pair["product"]: kp})
            _set_scales(pair["reverse"], {pair["substrate"]: ks, pair["product"]: kp})
            # Haldane: K = (Vf / Ks) / (Vr / Kp), so the reverse capacity follows.
            reverse_capacity = capacity * kp / (ratio * ks)
            pair["reverse"]["capacity_M_s"] = reverse_capacity
            pair["reverse"]["source"] = (
                f"Reverse of {name} by the Haldane relation: equilibrium constant "
                f"{ratio:.6g}, product Km {kp:.6g} M, substrate Km {ks:.6g} M"
            )[:3000]
        else:
            factor["scale_M"] = km
        channel["capacity_M_s"] = capacity
        channel["source"] = (
            f"Michaelis-Menten fit {entry['fit_sha256'][:12]}: Vmax "
            f"{substrate_vmax:.6g} M/s and Km {km:.6g} M at {entry['assay_enzyme']['value']:g} "
            f"{entry['assay_enzyme']['unit']} enzyme, active fraction {fraction:g}; "
            f"scaled to {catalyst['reference_amount']:g} {catalyst['amount_unit']} in "
            f"{definition['volume_L']:g} L"
        )[:3000]
        reached = [
            row["initial_M"] for row in definition["species"] if row["id"] == species
        ] + [phase["feed_M"].get(species, 0.0) for phase in definition["phases"]]
        log.append(
            {
                "channel": name,
                "catalyst": catalyst["id"],
                "species": species,
                "capacity_M_s": capacity,
                "scale_M": km,
                "vmax_substrate_equivalent_M_s": substrate_vmax,
                "assay_enzyme_per_L": enzyme,
                "assay_active_fraction": fraction,
                "fit_domain_M": [low, high],
                "substrate_maximum_M": max(reached),
                "extrapolates": max(reached) > high,
                "activation_energy_J_mol": channel.get("temperature", {}).get(
                    "activation_energy_J_mol"
                ),
                "ph_pKa": [channel["ph"]["pKa1"], channel["ph"]["pKa2"]]
                if "ph" in channel
                else None,
            }
        )
    equilibrium_log = []
    for name, pair in reversible.items():
        if name not in seen:
            raise ValueError(
                f"The forward channel {name!r} of a reversible pair has no kinetics entry"
            )
        forward, reverse = pair["forward"], pair["reverse"]
        scales = {
            term["factors"][0]["species"]: term["factors"][0]["scale_M"]
            for term in forward["denominator"]["terms"]
        }
        implied = (forward["capacity_M_s"] / scales[pair["substrate"]]) / (
            reverse["capacity_M_s"] / scales[pair["product"]]
        )
        if not math.isclose(implied, pair["equilibrium_constant"], rel_tol=1e-12):
            raise ValueError(
                f"The rate law of the reversible pair {name!r} gives an equilibrium "
                "constant that differs from the supplied constant."
            )
        equilibrium_log.append(
            {
                "forward": name,
                "reverse": reverse["id"],
                "equilibrium_constant": pair["equilibrium_constant"],
                "product_km_M": pair["product_km_M"],
                "reverse_capacity_M_s": reverse["capacity_M_s"],
                "implied_equilibrium_constant": implied,
            }
        )
        seen.add(reverse["id"])
    inactivation_log = [_inactivate(definition, item) for item in inactivations or []]
    validate_request(definition)
    return {
        "definition": definition,
        "channels": log,
        "equilibria": equilibrium_log,
        "inactivation": inactivation_log,
        "unchanged": [c["id"] for c in definition["channels"] if c["id"] not in seen],
    }
