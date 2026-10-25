"""Explicit small response fits over source-checked assay observations."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict

from alambiq.capabilities.measure.science.inspection import inspect_assay
from alambiq.capabilities.measure.science.tables import UNITS, finite

PROFILES = (
    "response-linear-v1",
    "initial-rate-michaelis-menten-v1",
    "temperature-arrhenius-v1",
    "first-order-decay-v1",
    "ph-bell-v1",
)
LINEAR, MICHAELIS_MENTEN, ARRHENIUS, DECAY, PH_BELL = PROFILES

# Gas constant, J/(mol K): the activation energy is reported in joules per mole.
GAS_CONSTANT = 8.314462618

# The coefficients of each model, the lower bound each must have (None: unconstrained by
# default), and which of them may not be given a negative lower bound.
NAMES = {
    LINEAR: ["intercept", "slope"],
    MICHAELIS_MENTEN: ["Km", "Vmax"],
    ARRHENIUS: ["k_ref", "Ea"],
    DECAY: ["A0", "k"],
    PH_BELL: ["ymax", "pKa1", "pKa2"],
}
DEFAULT_BOUNDS = {
    LINEAR: {"intercept": [None, None], "slope": [None, None]},
    MICHAELIS_MENTEN: {"Km": [0, None], "Vmax": [0, None]},
    ARRHENIUS: {"k_ref": [0, None], "Ea": [None, None]},
    DECAY: {"A0": [0, None], "k": [0, None]},
    PH_BELL: {"ymax": [0, None], "pKa1": [None, None], "pKa2": [None, None]},
}
NONNEGATIVE = {
    LINEAR: (),
    MICHAELIS_MENTEN: ("Km", "Vmax"),
    ARRHENIUS: ("k_ref",),
    DECAY: ("A0", "k"),
    PH_BELL: ("ymax",),
}
EQUATIONS = {
    LINEAR: "intercept + slope * x",
    MICHAELIS_MENTEN: "Vmax * x / (Km + x)",
    ARRHENIUS: "k_ref * exp(-Ea / R * (1 / T - 1 / T_ref))",
    DECAY: "A0 * exp(-k * x)",
    PH_BELL: "ymax / (1 + 10^(pKa1 - pH) + 10^(pH - pKa2))",
}
TEMPERATURE_UNITS = {"K": 0.0, "degC": 273.15}
REQUIRED = {
    "sample_id",
    "metric",
    "x_quantity",
    "x_unit",
    "y_unit",
    "observation_basis",
}


def validate_parameters(profile, parameters):
    if profile not in PROFILES:
        raise ValueError(
            f"{profile!r} is not a fitting profile; the profiles are {', '.join(PROFILES)}."
        )
    if (
        not isinstance(parameters, dict)
        or not REQUIRED <= parameters.keys()
        or not parameters.keys()
        <= REQUIRED
        | {
            "exclude_measurement_ids",
            "prediction_x",
            "condition_ids",
            "pool_over",
            "parameter_bounds",
            "fixed_parameters",
            "uncertainty",
            "rate_basis",
            "reference_x",
        }
    ):
        raise ValueError(
            "The fit parameters must include sample_id, metric, x_quantity, x_unit, "
            "y_unit and observation_basis, and may include exclude_measurement_ids, "
            "prediction_x, condition_ids, pool_over, parameter_bounds, "
            "fixed_parameters, uncertainty, rate_basis and reference_x."
        )
    for key in REQUIRED:
        value = parameters[key]
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise ValueError(
                f"{key} must be a nonempty string of at most 200 characters."
            )
    if parameters["observation_basis"] not in {"supplied-values", "independent-means"}:
        raise ValueError(
            "observation_basis must be supplied-values or independent-means; found "
            f"{parameters['observation_basis']!r}."
        )
    for key in ("exclude_measurement_ids", "condition_ids", "pool_over"):
        if key in parameters:
            values = parameters[key]
            if (
                not isinstance(values, list)
                or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)
                or len(values) > 10000
            ):
                raise ValueError(
                    f"{key} must be a list of at most 10,000 distinct nonempty "
                    "identifiers."
                )
    if "prediction_x" in parameters:
        values = parameters["prediction_x"]
        if not isinstance(values, list) or not 1 <= len(values) <= 200:
            raise ValueError("prediction_x must be a list of 1 to 200 finite numbers.")
        for x in values:
            if isinstance(x, str):
                raise ValueError("Each value of prediction_x must be a number.")
            finite(x, "prediction_x")
            if profile == MICHAELIS_MENTEN and x < 0:
                raise ValueError(
                    "Each value of prediction_x of a Michaelis–Menten fit is a "
                    f"concentration and must be at least 0; found {x:g}."
                )
    parameter_constraints(profile, parameters)
    if "reference_x" in parameters:
        if profile != ARRHENIUS:
            raise ValueError("reference_x is accepted only with the model arrhenius.")
        finite(parameters["reference_x"], "reference_x")
    if "rate_basis" in parameters:
        if profile != MICHAELIS_MENTEN:
            raise ValueError(
                "rate_basis is accepted only with the model michaelis_menten."
            )
        validate_rate_basis(parameters["rate_basis"])
        if (
            UNITS.get(parameters["x_unit"], (None,))[0] != "concentration"
            or UNITS.get(parameters["y_unit"], (None,))[0] != "concentration-rate"
        ):
            raise ValueError(
                "rate_basis requires x_unit to be a concentration unit and y_unit to be "
                "a concentration-rate unit."
            )
    options = parameters.get("uncertainty", {"method": "local"})
    if not isinstance(options, dict) or options.get("method") not in {
        "local",
        "cluster-bootstrap",
    }:
        raise ValueError("The uncertainty method must be local or cluster-bootstrap.")
    if options["method"] == "local":
        if set(options) != {"method"}:
            raise ValueError("The local uncertainty has the field method only.")
    elif (
        set(options) != {"method", "resamples", "seed"}
        or type(options["resamples"]) is not int
        or not 100 <= options["resamples"] <= 2000
        or type(options["seed"]) is not int
        or not 0 <= options["seed"] <= 2147483647
    ):
        raise ValueError(
            "The cluster bootstrap requires exactly the fields method, resamples and "
            "seed, with resamples an integer from 100 to 2,000 and seed an integer "
            "from 0 to 2,147,483,647."
        )


def validate_rate_basis(basis):
    """Validate the assay's measured species and substrate-equivalent rate mapping."""
    if (
        not isinstance(basis, dict)
        or set(basis)
        != {"kind", "measured_species", "substrate", "measured_per_substrate"}
        or basis["kind"] not in {"substrate-consumption", "product-formation"}
    ):
        raise ValueError(
            "rate_basis must have exactly the fields kind, measured_species, substrate "
            "and measured_per_substrate, with kind substrate-consumption or "
            "product-formation."
        )
    for key in ("measured_species", "substrate"):
        if (
            not isinstance(basis[key], str)
            or not basis[key].strip()
            or len(basis[key]) > 200
        ):
            raise ValueError(
                f"rate_basis.{key} must be a nonempty string of at most 200 characters."
            )
    ratio = basis["measured_per_substrate"]
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("measured_per_substrate must be a finite molar ratio above 0.")
    if basis["kind"] == "substrate-consumption" and (
        ratio != 1 or basis["measured_species"] != basis["substrate"]
    ):
        raise ValueError(
            "With kind substrate-consumption, measured_species must equal substrate "
            "and measured_per_substrate must be 1."
        )
    return ratio


def parameter_constraints(profile, parameters):
    names = NAMES[profile]
    supplied, fixed = (
        parameters.get("parameter_bounds", {}),
        parameters.get("fixed_parameters", {}),
    )
    if (
        not isinstance(supplied, dict)
        or not isinstance(fixed, dict)
        or set(supplied) - set(names)
        or set(fixed) - set(names)
        or len(fixed) > len(names) - 1
        or (len(fixed) > 1 and len(names) == 2)
    ):
        raise ValueError(
            "parameter_bounds and fixed_parameters must be mappings that name only "
            f"coefficients of the model ({', '.join(names)}), and at least 1 "
            "coefficient must remain free."
        )
    bounds = {name: supplied.get(name, DEFAULT_BOUNDS[profile][name]) for name in names}
    for name, pair in bounds.items():
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(v is not None and type(v) not in (int, float) for v in pair)
        ):
            raise ValueError(
                f"The bounds of {name} must be a list [lower, upper] of numbers or null "
                "in the unit of the coefficient; null means unbounded."
            )
        lo, hi = [finite(v, "parameter bound") if v is not None else None for v in pair]
        if (lo is not None and hi is not None and lo >= hi) or (
            name in NONNEGATIVE[profile] and (lo is None or lo < 0)
        ):
            raise ValueError(
                f"The bounds of {name} must be ordered, lower < upper"
                + (
                    ", and the lower bound must be at least 0"
                    if name in NONNEGATIVE[profile]
                    else ""
                )
                + "."
            )
        if name in fixed:
            value = fixed[name]
            if type(value) not in (int, float):
                raise ValueError(
                    f"The fixed value of {name} must be a number in the unit of the "
                    "coefficient."
                )
            value = finite(value, "fixed parameter")
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                raise ValueError(f"The fixed value of {name} lies outside its bounds.")
    return names, bounds, fixed


def condition_variation(data, points, parameters):
    """Compare supplied non-x quantities."""
    contexts = [
        data.get("condition_values", {}).get(p["condition_id"], {}) for p in points
    ]
    names = set().union(*(c.keys() for c in contexts)) - {parameters["x_quantity"]}
    pooled = set(parameters.get("pool_over", []))
    if pooled - names:
        raise ValueError(
            "pool_over must name condition quantities of the assay record other than "
            f"x_quantity; unknown: {', '.join(sorted(pooled - names))}."
        )

    def normalized(quantity):
        if quantity is None:
            return None
        dimension, factor, offset = UNITS.get(
            quantity["unit"], (quantity["unit"], 1, 0)
        )
        return dimension, quantity["value"] * factor + offset

    changing = []
    for name in sorted(names):
        values = [normalized(c.get(name)) for c in contexts]
        first = values[0]
        if any(
            (v is None) != (first is None)
            or (
                v is not None
                and first is not None
                and (
                    v[0] != first[0]
                    or not math.isclose(v[1], first[1], rel_tol=1e-12, abs_tol=0)
                )
            )
            for v in values[1:]
        ):
            changing.append(name)
    if set(changing) - pooled:
        raise ValueError(
            "The condition quantities "
            + ", ".join(sorted(set(changing) - pooled))
            + " differ between the selected conditions or are missing in some of them. "
            "Select conditions with equal quantities in condition_ids, or list the "
            "quantities in pool_over; pooled measurements are fitted with one set of coefficients."
        )
    return changing


def checked_assay(data):
    """Reconstruct from the recorded source and settings."""
    try:
        if data["schema_version"] != "0.1":
            raise ValueError(
                "The assay input is not an assay.inspect record (schema_version 0.1)."
            )
        text = data["source_text"]
        if hashlib.sha256(text.encode()).hexdigest() != data["source_sha256"]:
            raise ValueError(
                "The source_text of the assay record differs from its source_sha256."
            )
        intake = data.get("intake", {})
        rebuilt = inspect_assay(
            text,
            data["conditions"],
            data["independent_unit"],
            table_format=intake.get("format", "csv"),
            columns=intake.get("columns"),
            defaults=intake.get("defaults"),
            units=intake.get("units"),
            condition_values=data.get("condition_values"),
            sample_descriptions=data.get("sample_descriptions"),
        )
        # Earlier dossiers omit the newly optional status fields. Every other
        # source-derived field and the complete record order must still agree.
        for key, optional in (
            ("rows", {"status", "limit"}),
            ("groups", {"status_counts"}),
        ):
            if len(data[key]) != len(rebuilt[key]):
                raise ValueError(
                    f"The number of {key} in the assay record differs from the number "
                    "rebuilt from its source_text."
                )
            for supplied, exact in zip(data[key], rebuilt[key], strict=True):
                expected = {
                    k: v for k, v in exact.items() if k not in optional or k in supplied
                }
                if supplied != expected:
                    raise ValueError(
                        f"The {key} of the assay record differ from the {key} rebuilt "
                        "from its source_text."
                    )
        if "source" in data:
            rebuilt["source"] = data["source"]
        return rebuilt
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            "The assay record is incomplete or has fields of the wrong type; it must be "
            "a complete assay.inspect record with its source and settings."
        ) from exc


def observations(data, profile, parameters):
    validate_parameters(profile, parameters)
    data = checked_assay(data)
    rows = [
        r
        for r in data["rows"]
        if r["sample_id"] == parameters["sample_id"]
        and r["metric"] == parameters["metric"]
    ]
    if not rows:
        raise ValueError(
            f"The assay record has no measurements of {parameters['metric']} for the "
            f"sample {parameters['sample_id']}."
        )
    excluded_ids = set(parameters.get("exclude_measurement_ids", []))
    if excluded_ids - {r["measurement_id"] for r in rows}:
        raise ValueError(
            "exclude_measurement_ids names measurements that are not in the series of "
            "the sample and the metric: "
            + ", ".join(sorted(excluded_ids - {r["measurement_id"] for r in rows}))
            + "."
        )
    conditions = parameters.get("condition_ids", list(data["conditions"]))
    if set(conditions) - data["conditions"].keys() or not conditions:
        raise ValueError(
            "condition_ids must be a nonempty list of conditions of the assay record."
        )
    included, excluded = [], []
    for row in rows:
        identifier = row["measurement_id"]
        quantity = (
            data.get("condition_values", {})
            .get(row["condition_id"], {})
            .get(parameters["x_quantity"])
        )
        reason = (
            "researcher-excluded"
            if identifier in excluded_ids
            else "condition-not-selected"
            if row["condition_id"] not in conditions
            else row["status"]
            if row["value"] is None
            else "missing-x-quantity"
            if quantity is None
            else None
        )
        if reason:
            excluded.append({"measurement_id": identifier, "reason": reason})
            continue
        if row["unit"] != parameters["y_unit"]:
            raise ValueError(
                f"y_unit is {parameters['y_unit']!r} but measurement {identifier} is in "
                f"{row['unit']!r}. The units must match exactly: convert the table before "
                "fitting, or change y_unit."
            )
        if quantity["unit"] != parameters["x_unit"]:
            raise ValueError(
                f"x_unit is {parameters['x_unit']!r} but condition "
                f"{row['condition_id']} gives {parameters['x_quantity']} in "
                f"{quantity['unit']!r}. The units must match exactly: convert the "
                "condition values before fitting, or change x_unit."
            )
        x = finite(quantity["value"], "condition value")
        if profile == MICHAELIS_MENTEN and x < 0:
            raise ValueError(
                f"The concentration of condition {row['condition_id']} is {x:g} "
                f"{quantity['unit']}; Michaelis–Menten concentrations must be at least 0."
            )
        if profile == ARRHENIUS:
            if parameters["x_unit"] not in TEMPERATURE_UNITS:
                raise ValueError("An Arrhenius fit requires x_unit to be K or degC.")
            if x + TEMPERATURE_UNITS[parameters["x_unit"]] <= 0:
                raise ValueError(
                    f"The temperature {x:g} {parameters['x_unit']} of condition "
                    f"{row['condition_id']} is not above absolute zero."
                )
            if row["value"] <= 0:
                raise ValueError(
                    "The rate constants of an Arrhenius fit must be positive; measurement "
                    f"{identifier} has {row['value']:g}."
                )
        if profile == DECAY and x < 0:
            raise ValueError(
                f"The time {x:g} of condition {row['condition_id']} is negative; times "
                "of a first-order decay must be at least 0."
            )
        included.append(
            {
                "condition_id": row["condition_id"],
                "x": x,
                "y": row["value"],
                "measurement_ids": [identifier],
                "independent_id": row["independent_id"] or None,
            }
        )
    if parameters["observation_basis"] == "independent-means":
        grouped = defaultdict(list)
        for obs in included:
            if not obs["independent_id"]:
                raise ValueError(
                    "observation_basis independent-means requires an independent_id "
                    "and a technical_id for every included measurement."
                )
            grouped[obs["condition_id"], obs["independent_id"]].append(obs)
        included = [
            {
                **members[0],
                "y": statistics.mean(m["y"] for m in members),
                "measurement_ids": [i for m in members for i in m["measurement_ids"]],
            }
            for members in grouped.values()
        ]
    elif any(obs["independent_id"] for obs in included):
        raise ValueError(
            "Measurements with an independent_id require observation_basis "
            "independent-means; supplied-values counts them as separate observations."
        )
    if len(included) < 4 or len({p["x"] for p in included}) < 3:
        raise ValueError(
            "A fit requires at least 4 included observations at 3 distinct x values; "
            f"found {len(included)} observations at {len({p['x'] for p in included})} "
            "distinct x values."
        )
    condition_variation(data, included, parameters)
    return data, included, excluded


def _correlation(np, covariance):
    if covariance is None:
        return None
    spread = np.sqrt(np.diag(covariance))
    if np.any(spread <= 0):
        return None
    return np.clip(covariance / np.outer(spread, spread), -1, 1).tolist()


def _numerics(np, profile, parameters, x, y):
    """The model of a profile in the scaled variables the optimiser works in.

    Returns the scales, the map from the x the user gave to the x the model takes, the
    factors that turn scaled coefficients into physical ones, the model and its
    Jacobian, a starting point and the physical units of each coefficient.
    """
    y_scale = max(float(np.max(np.abs(y))), 1e-100)
    x_unit, y_unit = parameters["x_unit"], parameters["y_unit"]

    def identity(values):
        return values

    base = {"y_scale": y_scale, "to_fit_x": identity}
    if profile == LINEAR:
        x_scale = max(float(np.max(np.abs(x))), 1e-100)

        def model(q, v):
            return q[0] + q[1] * v

        def jacobian(q, v):
            return np.column_stack((np.ones_like(v), v))

        return {
            **base,
            "x_scale": x_scale,
            "factors": np.array([y_scale, y_scale / x_scale]),
            "model": model,
            "jacobian": jacobian,
            "initial": lambda xs, ys: [float(np.mean(ys)), 0.0],
            "units": [y_unit, f"({y_unit})/({x_unit})"],
        }
    if profile == MICHAELIS_MENTEN:
        x_scale = max(float(np.max(np.abs(x))), 1e-100)

        def model(q, v):
            return np.divide(
                q[1] * v, q[0] + v, out=np.zeros_like(v), where=q[0] + v > 0
            )

        def jacobian(q, v):
            den = q[0] + v
            return np.column_stack(
                (
                    np.divide(-q[1] * v, den**2, out=np.zeros_like(v), where=den > 0),
                    np.divide(v, den, out=np.zeros_like(v), where=den > 0),
                )
            )

        return {
            **base,
            "x_scale": x_scale,
            "factors": np.array([x_scale, y_scale]),
            "model": model,
            "jacobian": jacobian,
            "initial": lambda xs, ys: [
                float(np.median(xs[xs > 0])),
                float(np.max(ys)),
            ],
            "units": [x_unit, y_unit],
        }
    if profile == ARRHENIUS:
        shift = TEMPERATURE_UNITS[x_unit]
        temperature = x + shift
        if parameters.get("reference_x") is not None:
            reference = float(parameters["reference_x"]) + shift
            if reference <= 0:
                raise ValueError("reference_x must be above absolute zero (0 K).")
        else:
            reference = float(np.mean(temperature))

        def model(q, t):
            return q[0] * np.exp(-q[1] * (reference / t - 1.0))

        def jacobian(q, t):
            decay = np.exp(-q[1] * (reference / t - 1.0))
            return np.column_stack((decay, -q[0] * (reference / t - 1.0) * decay))

        def initial(ts, ys):
            u = reference / ts - 1.0
            if np.ptp(u) > 0 and np.all(ys > 0):
                slope, intercept = np.polyfit(u, np.log(ys), 1)
                return [float(np.exp(intercept)), float(-slope)]
            return [float(np.mean(ys)), 1.0]

        return {
            **base,
            "x_scale": 1.0,
            "to_fit_x": lambda v: v + shift,
            "factors": np.array([y_scale, GAS_CONSTANT * reference]),
            "model": model,
            "jacobian": jacobian,
            "initial": initial,
            "units": [y_unit, "J/mol"],
            "reference_temperature_K": reference,
        }
    if profile == DECAY:
        x_scale = max(float(np.max(np.abs(x))), 1e-100)

        def model(q, v):
            return q[0] * np.exp(-q[1] * v)

        def jacobian(q, v):
            decay = np.exp(-q[1] * v)
            return np.column_stack((decay, -q[0] * v * decay))

        def initial(xs, ys):
            if np.ptp(xs) > 0 and np.all(ys > 0):
                slope, intercept = np.polyfit(xs, np.log(ys), 1)
                return [float(np.exp(intercept)), float(max(-slope, 1e-6))]
            return [float(np.max(ys)), 1.0]

        return {
            **base,
            "x_scale": x_scale,
            "factors": np.array([y_scale, 1.0 / x_scale]),
            "model": model,
            "jacobian": jacobian,
            "initial": initial,
            "units": [y_unit, f"1/{x_unit}"],
        }

    def model(q, ph):
        return q[0] / (1.0 + 10.0 ** (q[1] - ph) + 10.0 ** (ph - q[2]))

    def jacobian(q, ph):
        a, b = 10.0 ** (q[1] - ph), 10.0 ** (ph - q[2])
        d = 1.0 + a + b
        ln10 = math.log(10.0)
        return np.column_stack(
            (1.0 / d, -q[0] * ln10 * a / d**2, q[0] * ln10 * b / d**2)
        )

    def initial(xs, ys):
        peak = float(xs[int(np.argmax(ys))])
        return [float(np.max(ys)) * 1.2, peak - 1.0, peak + 1.0]

    return {
        **base,
        "x_scale": 1.0,
        "factors": np.array([y_scale, 1.0, 1.0]),
        "model": model,
        "jacobian": jacobian,
        "initial": initial,
        "units": [y_unit, "pH", "pH"],
    }


def _derived(np, profile, names, values, covariance, numerics, parameters, quantile):
    """Quantities that follow from the coefficients, with delta-method uncertainty."""
    x_unit = parameters["x_unit"]
    if profile == DECAY:
        quantities = {
            "half_life": (lambda v: math.log(2.0) / v[1], x_unit),
        }
    elif profile == ARRHENIUS:
        reference = numerics["reference_temperature_K"]
        quantities = {
            "activation_energy_kJ_mol": (lambda v: v[1] / 1000.0, "kJ/mol"),
            "pre_exponential_factor": (
                lambda v: v[0] * math.exp(v[1] / (GAS_CONSTANT * reference)),
                parameters["y_unit"],
            ),
        }
    elif profile == PH_BELL:
        quantities = {
            "optimum_pH": (lambda v: 0.5 * (v[1] + v[2]), "pH"),
            "activity_at_optimum": (
                lambda v: v[0] / (1.0 + 2.0 * 10.0 ** (0.5 * (v[1] - v[2]))),
                parameters["y_unit"],
            ),
        }
    else:
        return []
    point = np.asarray(values, dtype=float)
    rows = []
    for name, (function, unit) in quantities.items():
        value = float(function(point))
        error = None
        if covariance is not None and all(
            np.isfinite(covariance[i, i]) for i in range(len(names))
        ):
            gradient = np.zeros(len(names))
            for i in range(len(names)):
                step = 1e-6 * max(abs(point[i]), 1e-12)
                up, down = point.copy(), point.copy()
                up[i] += step
                down[i] -= step
                gradient[i] = (function(up) - function(down)) / (2.0 * step)
            variance = float(gradient @ covariance @ gradient)
            error = math.sqrt(max(variance, 0.0))
        rows.append(
            {
                "name": name,
                "value": value,
                "unit": unit,
                "standard_error": error,
                "interval_95": (
                    [value - quantile * error, value + quantile * error]
                    if error is not None
                    else None
                ),
            }
        )
    if profile == ARRHENIUS:
        rows.append(
            {
                "name": "reference_temperature",
                "value": numerics["reference_temperature_K"]
                - TEMPERATURE_UNITS[x_unit],
                "unit": x_unit,
                "standard_error": None,
                "interval_95": None,
            }
        )
    return rows


def fit_assay(data, profile, parameters):
    data, points, excluded = observations(data, profile, parameters)
    try:
        import numpy as np
        import scipy
        from scipy.optimize import least_squares
        from scipy.stats import t
    except ImportError as exc:
        raise ValueError("Install alambiq[quantitative] for response fitting") from exc

    x, y = (np.array([p[k] for p in points], dtype=float) for k in ("x", "y"))
    names, bounds, fixed = parameter_constraints(profile, parameters)
    numerics = _numerics(np, profile, parameters, x, y)
    x_scale, y_scale = numerics["x_scale"], numerics["y_scale"]
    xs, ys = numerics["to_fit_x"](x) / x_scale, y / y_scale
    factors = numerics["factors"]
    model, jacobian = numerics["model"], numerics["jacobian"]
    free = [i for i, name in enumerate(names) if name not in fixed]
    lower = (
        np.array([bounds[n][0] if bounds[n][0] is not None else -np.inf for n in names])
        / factors
    )
    upper = (
        np.array([bounds[n][1] if bounds[n][1] is not None else np.inf for n in names])
        / factors
    )

    def expand(q):
        full = np.array([fixed.get(name, 0) for name in names], dtype=float) / factors
        full[free] = q
        return full

    if profile == MICHAELIS_MENTEN and not np.any(y > 0):
        raise ValueError(
            "Michaelis–Menten fitting requires at least 1 positive response value."
        )

    initial = np.clip(numerics["initial"](xs, ys), lower, upper)[free]

    def solve(xx, yy, start):
        return least_squares(
            lambda q: model(expand(q), xx) - yy,
            start,
            jac=lambda q: jacobian(expand(q), xx)[:, free],
            bounds=(lower[free], upper[free]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
            method="trf",
            loss="linear",
            x_scale=1.0,
        )

    fit = solve(xs, ys, initial)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"The numerical fit did not converge: {fit.message}")
    qhat = expand(fit.x)
    n, df = len(points), len(points) - len(free)
    jac = jacobian(qhat, xs)[:, free]
    rank = int(np.linalg.matrix_rank(jac))
    condition = float(np.linalg.cond(jac))
    residual = y - model(qhat, xs) * y_scale
    variance_scaled = float(np.sum(fit.fun**2) / df)
    flat = float(np.ptp(ys)) < 1e-10
    reliable = (
        rank == len(free)
        and np.isfinite(condition)
        and condition < 1e8
        and not (profile != LINEAR and flat and len(free) >= 2)
        and not np.any(fit.active_mask)
    )
    inverse_jac = np.linalg.pinv(jac) if reliable else None
    covariance = np.zeros((len(names), len(names))) if reliable else None
    if reliable:
        covariance[np.ix_(free, free)] = inverse_jac @ inverse_jac.T * variance_scaled
    values = qhat * factors
    uncertainty = dict(parameters.get("uncertainty", {"method": "local"}))
    bootstrap = uncertainty["method"] == "cluster-bootstrap"
    draws = None
    if bootstrap:
        if any(not p["independent_id"] for p in points):
            raise ValueError(
                "The cluster bootstrap requires a named independent unit for every "
                "observation (observation_basis independent-means)."
            )
        groups = defaultdict(list)
        for i, point in enumerate(points):
            groups[point["independent_id"]].append(i)
        if len(groups) < 4:
            raise ValueError(
                "The cluster bootstrap requires at least 4 independent units; found "
                f"{len(groups)}."
            )
        units = list(groups.values())
        rng = np.random.default_rng(uncertainty["seed"])
        samples, failures, boundary = [], [], 0
        for attempt in range(uncertainty["resamples"]):
            indices = np.concatenate(
                [units[i] for i in rng.integers(0, len(units), len(units))]
            )
            try:
                if len(set(xs[indices])) < 3:
                    raise ValueError("fewer than 3 distinct x values")
                sampled = solve(xs[indices], ys[indices], fit.x)
                if not sampled.success or not np.isfinite(sampled.x).all():
                    raise ValueError(
                        "optimizer did not converge to finite coefficients"
                    )
                if np.linalg.matrix_rank(
                    jacobian(expand(sampled.x), xs[indices])[:, free]
                ) < len(free):
                    raise ValueError("rank-deficient resampled Jacobian")
                samples.append(expand(sampled.x))
                boundary += int(np.any(sampled.active_mask))
            except (ValueError, ArithmeticError) as exc:
                failures.append({"resample": attempt, "reason": str(exc)})
        uncertainty.update(
            units=len(units),
            successful_resamples=len(samples),
            failed_resamples=failures,
            boundary_resamples=boundary,
            parameter_draws=(np.asarray(samples) * factors).tolist() if samples else [],
        )
        # Retain every draw/failure; do not silently report a selected successful subset.
        reliable = reliable and not failures
        draws = np.asarray(samples) if reliable else None
        covariance = np.cov(draws, rowvar=False, ddof=1) if reliable else None
    errors = (
        np.sqrt(np.maximum(np.diag(covariance), 0)) * factors
        if covariance is not None
        else None
    )
    quantile = float(t.ppf(0.975, df))
    units = numerics["units"]
    diagnostics = []
    varying = condition_variation(data, points, parameters)
    if varying:
        diagnostics.append(
            "Quantities listed in pool_over that differ between conditions or are "
            "partly missing: " + ", ".join(varying) + ". The measurements are pooled across them with one set of "
            "coefficients."
        )
    if fixed:
        diagnostics.append(
            "The uncertainty is conditional on the fixed coefficients."
        )
    if bootstrap:
        diagnostics.append(
            f"The bootstrap resamples {uncertainty['units']} named independent units."
        )
        if uncertainty["failed_resamples"]:
            diagnostics.append(
                f"{len(uncertainty['failed_resamples'])} bootstrap resamples failed; "
                "no interval is given."
            )
        if uncertainty["boundary_resamples"]:
            diagnostics.append(
                f"{uncertainty['boundary_resamples']} bootstrap fits reached a parameter "
                "bound."
            )
    if flat:
        diagnostics.append(
            "The supplied response is constant across x; the data cannot determine the coefficients of this model."
            if profile != LINEAR
            else "The supplied response is constant across x."
        )
    if not reliable:
        diagnostics.append(
            "No parameter uncertainty is estimated, because the fit has a coefficient at "
            "a bound, a rank-deficient or ill-conditioned local Jacobian, or failed "
            "resamples."
        )
    if profile == MICHAELIS_MENTEN and not float(np.min(x)) < values[0] < float(
        np.max(x)
    ):
        diagnostics.append(
            "The fitted Km is outside the measured concentration range."
        )
    if profile == MICHAELIS_MENTEN and np.any(y < 0):
        diagnostics.append(
            "Negative recorded rates are included in the fit."
        )
    if variance_scaled < 1e-20:
        diagnostics.append(
            "The scaled residual variance is below 1e-20."
        )
    if excluded:
        diagnostics.append(
            f"{len(excluded)} measurement records are not used in this fit."
        )
    if len({p["independent_id"] for p in points if p["independent_id"]}) < sum(
        p["independent_id"] is not None for p in points
    ):
        diagnostics.append(
            "An independent unit appears under more than one condition."
        )
    requested_x = parameters.get(
        "prediction_x", np.linspace(float(np.min(x)), float(np.max(x)), 51).tolist()
    )
    predictions = []
    for value in requested_x:
        scaled_x = numerics["to_fit_x"](np.array([float(value)])) / x_scale
        predicted = float(model(qhat, scaled_x)[0] * y_scale)
        grad = jacobian(qhat, scaled_x)[0]
        mean_se = (
            float(np.sqrt(max(float(grad @ covariance @ grad), 0)) * y_scale)
            if covariance is not None
            else None
        )
        prediction_se = (
            float(np.sqrt((mean_se / y_scale) ** 2 + variance_scaled) * y_scale)
            if mean_se is not None
            else None
        )
        mean_interval = (
            [predicted - quantile * mean_se, predicted + quantile * mean_se]
            if mean_se is not None
            else None
        )
        if bootstrap:
            prediction_se = None
            samples_y = (
                np.array([model(q, scaled_x)[0] * y_scale for q in draws])
                if draws is not None
                else None
            )
            mean_se = (
                float(np.std(samples_y, ddof=1)) if samples_y is not None else None
            )
            mean_interval = (
                np.quantile(samples_y, [0.025, 0.975]).tolist()
                if samples_y is not None
                else None
            )
        predictions.append(
            {
                "x": float(value),
                "y": predicted,
                "mean_standard_error": mean_se,
                "prediction_standard_error": prediction_se,
                "mean_interval_95": mean_interval,
                "prediction_interval_95": [
                    predicted - quantile * prediction_se,
                    predicted + quantile * prediction_se,
                ]
                if prediction_se is not None
                else None,
                "extrapolation": bool(value < np.min(x) or value > np.max(x)),
            }
        )

    def parameter_interval(i, name):
        if name in fixed or errors is None:
            return None
        if bootstrap:
            return np.quantile(draws[:, i] * factors[i], [0.025, 0.975]).tolist()
        return [
            float(values[i] - quantile * errors[i]),
            float(values[i] + quantile * errors[i]),
        ]

    result = {
        "schema_version": "0.1",
        "model": profile,
        "equation": EQUATIONS[profile],
        "request": parameters,
        "source": {
            "assay_source_sha256": data["source_sha256"],
            "receipt": data.get("source"),
            "conditions": data["conditions"],
            "condition_values": data.get("condition_values", {}),
            "sample_description": data.get("sample_descriptions", {}).get(
                parameters["sample_id"]
            ),
            "independent_unit": data["independent_unit"],
        },
        "observations": [
            {**p, "predicted": float(p["y"] - r), "residual": float(r)}
            for p, r in zip(points, residual, strict=True)
        ],
        "excluded": excluded,
        "parameters": [
            {
                "name": name,
                "value": float(values[i]),
                "unit": units[i],
                "fixed": name in fixed,
                "standard_error": float(errors[i])
                if errors is not None and name not in fixed
                else None,
                "interval_95": parameter_interval(i, name),
            }
            for i, name in enumerate(names)
        ],
        "parameter_dependence": {
            "order": names,
            "covariance": (covariance * np.outer(factors, factors)).tolist()
            if covariance is not None
            else None,
            "correlation": float(
                np.clip(
                    covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]),
                    -1,
                    1,
                )
            )
            if len(names) == 2
            and covariance is not None
            and covariance[0, 0] > 0
            and covariance[1, 1] > 0
            else None,
            "correlation_matrix": _correlation(np, covariance),
        },
        "derived": _derived(
            np,
            profile,
            names,
            values,
            covariance * np.outer(factors, factors) if covariance is not None else None,
            numerics,
            parameters,
            quantile,
        ),
        "optimizer": {
            "success": True,
            "method": "scipy.optimize.least_squares",
            "message": str(fit.message),
            "evaluations": int(fit.nfev),
            "algorithm": "trf",
            "loss": "linear",
            "tolerances": {"xtol": 1e-12, "ftol": 1e-12, "gtol": 1e-12},
            "max_nfev": 2000,
            "normalization": {
                "x_scale": x_scale,
                "y_scale": y_scale,
                "optimizer_x_scale": 1.0,
            },
            "parameter_bounds": bounds,
            "fixed_parameters": fixed,
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        },
        "statistics": {
            "n": n,
            "unique_x": len(set(x)),
            "parameter_count": len(names),
            "free_parameter_count": len(free),
            "residual_degrees_of_freedom": df,
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "residual_standard_deviation": float(np.sqrt(variance_scaled) * y_scale),
            "jacobian_rank": rank,
            "scaled_jacobian_condition_number": condition
            if np.isfinite(condition)
            else None,
        },
        "domain": {"minimum_x": float(np.min(x)), "maximum_x": float(np.max(x))},
        "predictions": predictions,
        "diagnostics": diagnostics,
        "uncertainty": uncertainty,
        "method": "Unweighted least squares on the observation basis selected in the request. Technical readings are averaged within a named independent unit when independent-means is requested. "
        + (
            "Cluster percentile-bootstrap intervals of the coefficients and of the mean resample whole named independent units, with the measurements of a unit kept together. They assume experimentally independent, representative units. Intervals of future observations are available with the residual method. "
            if bootstrap
            else "Local Jacobian covariance and Student-t intervals assume independent, homoskedastic residual errors and an adequate model; nonlinear intervals are approximate and can extend beyond physical parameter bounds. The uncertainty of the mean and the uncertainty of a future observation are separate. "
        )
        + "Fixed coefficients have no estimated interval; the other uncertainty is conditional on their values. Extrapolations are labeled.",
    }
    # Numerical extremes must become a useful failure, never invalid JSON.
    try:
        json.dumps(result, allow_nan=False)
    except ValueError as exc:
        raise ValueError(
            "The numerical result contains values that are not finite; check the units "
            "and the range of the data."
        ) from exc
    return result


def assay_data(value):
    """Accept an inspection record, or supplied JSON text holding one."""
    if isinstance(value, dict) and "text" in value:
        try:
            return json.loads(value["text"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The source text of the assay input is not a JSON assay.inspect record."
            ) from exc
    return value
