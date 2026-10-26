"""A constant-condition, single-substrate batch balance."""

import hashlib
import json
import math

from alambiq.capabilities.measure.science.fitting import (
    validate_rate_basis,
)
from alambiq.capabilities.measure.science.tables import UNITS, finite

PROFILE = "constant-condition-batch-target-v1"
QUANTITIES = {
    "substrate_initial": {k: v[1] for k, v in UNITS.items() if v[0] == "concentration"},
    "duration": {"s": 1, "min": 60, "h": 3600},
    "volume": {"L": 1, "mL": 0.001},
    "product_molar_mass": {"g/mol": 1},
    "target_product": {"g": 1, "mg": 0.001, "kg": 1000},
}
FRACTIONS = {"formation_fraction", "recovery_fraction", "accepted_fraction"}
FIELDS = tuple(QUANTITIES) + (
    "product_per_substrate",
    "formation_fraction",
    "recovery_fraction",
    "accepted_fraction",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def numerical_parameters(parameters):
    values = {}
    for name in FIELDS:
        value = parameters[name]
        if name in QUANTITIES:
            if (
                not isinstance(value, dict)
                or set(value) != {"value", "unit"}
                or value["unit"] not in QUANTITIES[name]
            ):
                raise ValueError(
                    f"{name} must have exactly the fields value and unit, with a unit "
                    f"from {', '.join(QUANTITIES[name])}."
                )
            if type(value["value"]) not in (int, float):
                raise ValueError(f"The value of {name} must be a number.")
            value = finite(value["value"], name) * QUANTITIES[name][value["unit"]]
        if type(value) not in (int, float):
            raise ValueError(f"{name} must be a number.")
        value = finite(value, name)
        valid = 0 <= value <= 1 if name in FRACTIONS else value > 0
        if not valid:
            raise ValueError(
                f"{name} must be {'from 0 to 1' if name in FRACTIONS else 'positive'}; "
                f"found {value:g}."
            )
        values[name] = value
    return values


def validate_parameters(parameters):
    if (
        not isinstance(parameters, dict)
        or not set(FIELDS) | {"assumption_note"} <= parameters.keys()
        or parameters.keys() - set(FIELDS) - {"assumption_note", "sensitivity"}
    ):
        raise ValueError(
            "The balance parameters are "
            + ", ".join(FIELDS)
            + ", assumption_note and optionally sensitivity."
        )
    note = parameters["assumption_note"]
    if not isinstance(note, str) or not note.strip() or len(note) > 3000:
        raise ValueError(
            "assumption_note must be nonblank text of at most 3,000 characters that "
            "states the assumptions of transferring the experiment to the application."
        )
    numerical_parameters(parameters)
    scenarios = parameters.get("sensitivity", [])
    if not isinstance(scenarios, list) or len(scenarios) > 20:
        raise ValueError("sensitivity must be a list of at most 20 scenarios.")
    ids = {"base"}
    for s in scenarios:
        if (
            not isinstance(s, dict)
            or set(s) != {"id", "overrides"}
            or not isinstance(s["id"], str)
            or not s["id"].strip()
            or len(s["id"]) > 200
            or s["id"] in ids
        ):
            raise ValueError(
                "Each sensitivity scenario has exactly the fields id and overrides. "
                "Its id is a nonblank string of at most 200 characters, distinct "
                "from the other ids and from base."
            )
        ids.add(s["id"])
        if (
            not isinstance(s["overrides"], dict)
            or not s["overrides"]
            or s["overrides"].keys() - set(FIELDS)
        ):
            raise ValueError(
                "The overrides of a scenario are a nonempty mapping of balance "
                "quantities or fractions. The fitted coefficients cannot be overridden."
            )
        numerical_parameters({**parameters, **s["overrides"]})


def kinetics(fit):
    if (
        not isinstance(fit, dict)
        or fit.get("schema_version") != "0.1"
        or fit.get("model") != "initial-rate-michaelis-menten-v1"
        or fit.get("optimizer", {}).get("success") is not True
    ):
        raise ValueError(
            "fit must be a record of a converged Michaelis–Menten initial-rate fit "
            "(model initial-rate-michaelis-menten-v1)."
        )
    records = fit.get("parameters", [])
    if len(records) != 2 or {p.get("name") for p in records} != {"Km", "Vmax"}:
        raise ValueError(
            "The fit record must contain exactly the coefficients Km and Vmax."
        )
    values, units = {}, {}
    for p in records:
        unit = p.get("unit")
        dimension = "concentration" if p["name"] == "Km" else "concentration-rate"
        if unit not in UNITS or UNITS[unit][0] != dimension:
            raise ValueError(
                f"The unit {unit!r} of {p['name']} is not a unit of {dimension}."
            )
        values[p["name"]] = finite(p.get("value"), p["name"]) * UNITS[unit][1]
        units[p["name"]] = unit
    if values["Km"] < 0 or values["Vmax"] <= 0:
        raise ValueError(
            "The balance requires Km of at least 0 and Vmax above 0; found "
            f"Km {values['Km']:g} and Vmax {values['Vmax']:g} in base units."
        )
    if (
        fit.get("request", {}).get("x_unit") != units["Km"]
        or fit.get("request", {}).get("y_unit") != units["Vmax"]
    ):
        raise ValueError(
            "The x_unit and y_unit of the fit request differ from the units of Km and "
            "Vmax."
        )
    domain = fit.get("domain", {})
    low, high = [
        finite(domain.get(k), "fit domain") * UNITS[units["Km"]][1]
        for k in ("minimum_x", "maximum_x")
    ]
    if not 0 <= low < high:
        raise ValueError(
            "The measured concentration range of the fit must satisfy 0 <= minimum < "
            f"maximum; found {low:g} to {high:g} M."
        )
    ratio = validate_rate_basis(fit.get("request", {}).get("rate_basis"))
    substrate_vmax = values["Vmax"] / ratio
    if not math.isfinite(substrate_vmax) or substrate_vmax <= 0:
        raise ValueError(
            "The substrate-equivalent Vmax must be finite and positive; check the "
            "rate_basis of the fit."
        )
    return values["Km"], substrate_vmax, units["Vmax"], low, high, ratio


def balance(fit, parameters):
    validate_parameters(parameters)
    km, vmax, rate_unit, low, high, rate_ratio = kinetics(fit)
    scenarios = []
    for s in [{"id": "base", "overrides": {}}] + parameters.get("sensitivity", []):
        request = {k: parameters[k] for k in FIELDS} | s["overrides"]
        p = numerical_parameters(request)
        theoretical = (
            p["substrate_initial"]
            * p["volume"]
            * p["product_molar_mass"]
            * p["product_per_substrate"]
        )
        formed = theoretical * p["formation_fraction"]
        recovered = formed * p["recovery_fraction"]
        accepted = recovered * p["accepted_fraction"]
        if accepted == 0 and all(p[k] > 0 for k in FRACTIONS):
            raise ValueError(
                "The product amount underflows to 0 although all product fractions "
                "are above 0; check the units and the ranges of the inputs."
            )
        conversion = p["target_product"] / accepted if accepted else None
        row = {
            "id": s["id"],
            "inputs": request,
            "available_substrate_mol": p["substrate_initial"] * p["volume"],
            "maximum_formed_product_g": formed,
            "maximum_recovered_product_g": recovered,
            "maximum_accepted_product_g": accepted,
            "required_conversion": conversion,
            "formed_product_g": None,
            "recovered_product_g": None,
            "accepted_product_g": None,
            "remaining_substrate_M": None,
            "required_vmax": None,
            "activity_multiple": None,
            "time_at_measured_vmax": None,
            "concentration_extrapolation": p["substrate_initial"] > high,
            "limitation": None,
        }
        if conversion is None:
            row["limitation"] = (
                "A product fraction of 0 leaves no accepted product at any catalyst activity."
            )
        elif conversion > 1:
            row["limitation"] = (
                "The target exceeds the mass of accepted product available at any catalyst activity."
            )
        elif conversion == 1 and km > 0:
            row["limitation"] = (
                "A conversion of exactly 1 takes infinite time when Km is above 0."
            )
        else:
            remaining = p["substrate_initial"] * (1 - conversion)
            # Integrate substrate disappearance using substrate-equivalent Vmax.
            integral = p["substrate_initial"] * conversion - (
                km * math.log1p(-conversion) if km else 0
            )
            required = integral / p["duration"]
            row.update(
                formed_product_g=formed * conversion,
                recovered_product_g=recovered * conversion,
                accepted_product_g=accepted * conversion,
                remaining_substrate_M=remaining,
                required_vmax={
                    "value": required * rate_ratio / UNITS[rate_unit][1],
                    "unit": rate_unit,
                },
                activity_multiple=required / vmax,
                time_at_measured_vmax={
                    "value": integral
                    / vmax
                    / QUANTITIES["duration"][request["duration"]["unit"]],
                    "unit": request["duration"]["unit"],
                },
                concentration_extrapolation=remaining < low
                or p["substrate_initial"] > high,
            )
        scenarios.append(row)
    result = {
        "schema_version": "0.1",
        "profile": PROFILE,
        "request": parameters,
        "source": {
            "fit_sha256": digest(fit),
            "context": fit.get("source"),
            "fit_request": fit["request"],
            "parameters": fit["parameters"],
            "diagnostics": fit.get("diagnostics", []),
            "rate_basis": fit["request"]["rate_basis"],
            "substrate_vmax": {"value": vmax, "unit": "M/s"},
        },
        "scenarios": scenarios,
        "method": "Integrated irreversible single-substrate Michaelis–Menten depletion at constant enzyme activity, constant volume and conditions compatible with the assay. The measured rate is converted to substrate consumption with the molar mapping of the assay. The required Vmax is expressed on the basis of the originally measured species. The formation, recovery and accepted-product fractions describe the application scenario. Differences between scenarios show the sensitivity to the inputs. The uncertainty of the fitted coefficients is in the source fit.",
        "equation": "required_Vmax_measured = measured_per_substrate * (S0*conversion - Km*ln(1-conversion))/duration",
    }
    try:
        json.dumps(result, allow_nan=False)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            "The balance contains values that are not finite; check the units and the "
            "ranges of the inputs."
        ) from exc
    return result
