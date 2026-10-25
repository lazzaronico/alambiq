"""Bounded progress fitting of explicit well-mixed operation requests."""

from __future__ import annotations

import copy
import math

from alambiq.capabilities.operate.science.simulation import (
    _fields,
    _number,
    _records,
    simulate,
    validate_request,
)
from alambiq.capabilities.shared.records import digest, label

PROFILE = "well-mixed-operation-fit-v1"


class OperationFitError(ValueError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


def _named(rows, identifier, key="id"):
    return next((r for r in rows if r[key] == identifier), None)


def _target(operation, path):
    """Resolve only the operation model's supported physical scalar fields."""
    if (
        not isinstance(path, list)
        or not 3 <= len(path) <= 8
        or any(type(part) not in (str, int) for part in path)
    ):
        raise ValueError(
            "Each target path must be a list of 3 to 8 strings and integers that names "
            "a supported numerical field."
        )
    parent, field, unit, positive, fraction = None, path[-1], None, False, False
    if path[0] == "channels" and isinstance(path[1], str):
        channel = _named(operation["channels"], path[1])
        if channel is not None:
            if len(path) == 3 and field == "capacity_M_s":
                parent, unit = channel, "M/s"
            elif len(path) == 4 and path[2] == "inhibitors_M":
                parent, unit, positive = channel.get("inhibitors_M"), "M", True
            elif path[2] == "factors" and 5 <= len(path) <= 6:
                factor = _named(channel["factors"], path[3], "species")
                if factor is not None:
                    if len(path) == 5 and field in {"scale_M", "self_inhibition_M"}:
                        parent, unit, positive = factor, "M", True
                    elif len(path) == 6 and path[4] in {
                        "competitive_inhibitors_M",
                        "uncompetitive_inhibitors_M",
                    }:
                        parent, unit, positive = factor.get(path[4]), "M", True
            elif path[2:4] == ["denominator", "terms"] and len(path) in (6, 8):
                term = _named(channel.get("denominator", {}).get("terms", []), path[4])
                if term is not None:
                    if len(path) == 6 and field == "coefficient":
                        parent, unit, positive = term, "1", True
                    elif len(path) == 8 and path[5] == "factors" and field == "scale_M":
                        parent = _named(term["factors"], path[6], "species")
                        unit, positive = "M", True
    elif path[0] == "catalysts" and len(path) == 3:
        catalyst = _named(operation["catalysts"], path[1])
        if catalyst is not None:
            if field == "amount":
                parent, unit = catalyst, catalyst["amount_unit"]
            elif field == "deactivation_per_s":
                parent, unit = catalyst, "1/s"
            elif field in {"active_fraction", "washout_fraction"}:
                parent, unit, fraction = catalyst, "fraction", True
    elif (
        path[0] == "events"
        and len(path) == 4
        and type(path[1]) is int
        and 0 <= path[1] < len(operation["events"])
        and path[2] in {"catalyst_retention", "catalyst_activity_retention"}
    ):
        parent, unit, fraction = (
            operation["events"][path[1]].get(path[2]),
            "fraction",
            True,
        )
    if parent is None or not isinstance(field, str) or field not in parent:
        raise ValueError(
            f"The target path {path} does not resolve to an existing supported "
            "numerical field of the operation."
        )
    return parent, field, unit, positive, fraction


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def validate_observation(op, obs):
    """Resolve a measurement's physical basis against an explicit operation."""
    _number(obs["time_s"], "observation time")
    if obs["state"] not in {"before", "after"}:
        raise ValueError(
            "The state of an observation must be before or after an exchange."
        )
    times = (
        [e["time_s"] for e in op["events"]]
        if obs["state"] == "before"
        else [*op["times_s"], *(e["time_s"] for e in op["events"])]
    )
    if obs["quantity"] == "effluent-concentration":
        collection = _named(op.get("collections", []), obs.get("collection_id"))
        if (
            collection is None
            or obs["state"] != "after"
            or obs["time_s"] != collection["end_s"]
        ):
            raise ValueError(
                "An effluent-concentration observation must name a collection_id, the "
                "end time of that collection and the state after."
            )
    elif "collection_id" in obs:
        raise ValueError(
            "collection_id is accepted only for effluent-concentration observations."
        )
    elif obs["time_s"] not in times:
        raise ValueError(
            "The time of an observation must be the time of a recorded state, and the "
            "state before requires the time of an exchange."
        )
    label(obs["target"], "observation target")
    if obs["quantity"] in {
        "concentration",
        "conversion",
        "removed",
        "effluent-concentration",
    }:
        if _named(op["species"], obs["target"]) is None:
            raise ValueError(
                f"The target of a {obs['quantity']} observation must be a declared "
                "species."
            )
        unit = {
            "concentration": "M",
            "conversion": "fraction",
            "removed": "mol",
            "effluent-concentration": "M",
        }[obs["quantity"]]
    elif obs["quantity"] in {"catalyst-total", "catalyst-active"}:
        catalyst = _named(op["catalysts"], obs["target"])
        if catalyst is None:
            raise ValueError(
                f"The target of a {obs['quantity']} observation must be a declared "
                "catalyst."
            )
        unit = catalyst["amount_unit"]
    else:
        raise ValueError(
            "The quantity of an observation must be concentration, "
            "effluent-concentration, conversion, removed, catalyst-total or "
            f"catalyst-active; found {obs['quantity']!r}."
        )
    if obs["unit"] != unit:
        raise ValueError(
            f"The unit of the {obs['quantity']} observation of {obs['target']} must be "
            f"{unit}; found {obs['unit']!r}."
        )
    if obs["quantity"] == "conversion":
        _number(obs.get("reference_M"), "conversion reference_M", positive=True)
    elif "reference_M" in obs:
        raise ValueError("reference_M is accepted only for conversion observations.")
    _finite(obs["value"], "observed value")
    label(obs["source"], "observation source", 3000)


def validate_fit_request(request):
    _fields(
        request,
        {
            "id",
            "experiments",
            "parameters",
            "starts",
            "observations",
            "uncertainty",
            "max_nfev",
        },
    )
    label(request["id"], "fit id")
    _records(request["experiments"], "experiments", 20)
    experiments = {op["id"]: op for op in request["experiments"]}
    for operation in experiments.values():
        validate_request(operation)
    names = _records(request["parameters"], "parameters", 12)
    targets = set()
    for parameter in request["parameters"]:
        _fields(
            parameter,
            {"id", "unit", "lower", "upper", "scale", "targets"},
            {"profile_values"},
        )
        label(parameter["unit"], "parameter unit")
        _number(parameter["lower"], "parameter lower")
        _number(parameter["upper"], "parameter upper", positive=True)
        _number(parameter["scale"], "parameter scale", positive=True)
        lo, hi = parameter["lower"], parameter["upper"]
        if lo >= hi:
            raise ValueError(
                f"The bounds of parameter {parameter['id']} must satisfy lower < upper."
            )
        scaled_lo, scaled_hi = lo / parameter["scale"], hi / parameter["scale"]
        margin = min(1e-4, (scaled_hi - scaled_lo) / 4)
        if (
            not math.isfinite(scaled_lo)
            or not math.isfinite(scaled_hi)
            or not scaled_lo + margin > scaled_lo
            or not scaled_hi - margin < scaled_hi
        ):
            raise ValueError(
                f"The scaled bounds of parameter {parameter['id']} must be finite and "
                "wide enough that a step of at most 1e-4 inside each bound is "
                "representable."
            )
        if (
            not isinstance(parameter["targets"], list)
            or not 1 <= len(parameter["targets"]) <= 200
        ):
            raise ValueError(f"Parameter {parameter['id']} must have 1 to 200 targets.")
        for target in parameter["targets"]:
            _fields(target, {"experiment_id", "path"})
            if target["experiment_id"] not in experiments:
                raise ValueError(
                    f"The experiment_id {target['experiment_id']!r} of a target of "
                    f"parameter {parameter['id']} is not the id of an experiment of the "
                    "fit request."
                )
            parent, field, unit, positive, fraction = _target(
                experiments[target["experiment_id"]], target["path"]
            )
            if (
                unit != parameter["unit"]
                or (positive and lo == 0)
                or (fraction and hi > 1)
            ):
                raise ValueError(
                    f"The unit and the bounds of parameter {parameter['id']} do not "
                    f"match its target {target['path']}: the target is in {unit}"
                    + (", is positive, so lower must be above 0" if positive else "")
                    + (
                        ", is a fraction, so upper must be at most 1"
                        if fraction
                        else ""
                    )
                    + "."
                )
            key = (target["experiment_id"], *target["path"])
            if key in targets:
                raise ValueError(
                    f"The target {target['path']} occurs under more than one parameter; "
                    "each target must belong to exactly 1 parameter."
                )
            targets.add(key)
        grid = parameter.get("profile_values", [])
        if not isinstance(grid, list) or len(grid) > 50:
            raise ValueError(
                f"profile_values of parameter {parameter['id']} must be a list of at "
                "most 50 values."
            )
        for value in grid:
            if not lo <= _finite(value, "profile value") <= hi:
                raise ValueError(
                    f"profile_values of parameter {parameter['id']} must lie within its "
                    "bounds."
                )
        if len(grid) != len(set(grid)):
            raise ValueError(
                f"profile_values of parameter {parameter['id']} must be distinct."
            )
    if not isinstance(request["starts"], list) or not 1 <= len(request["starts"]) <= 10:
        raise ValueError("starts must be a list of 1 to 10 optimizer starts.")
    for start in request["starts"]:
        if not isinstance(start, dict) or set(start) != names:
            raise ValueError(
                "Each start must give a value for every fitted parameter and for no "
                "other."
            )
        for parameter in request["parameters"]:
            if (
                not parameter["lower"]
                <= _finite(start[parameter["id"]], "start")
                <= parameter["upper"]
            ):
                raise ValueError(
                    "Each start value must lie within the bounds of its parameter."
                )
    _records(request["observations"], "observations", 10000)
    fit_count = 0
    for obs in request["observations"]:
        _fields(
            obs,
            {
                "id",
                "experiment_id",
                "time_s",
                "state",
                "quantity",
                "target",
                "unit",
                "value",
                "scale",
                "partition",
                "source",
            },
            {"reference_M", "collection_id"},
        )
        if obs["experiment_id"] not in experiments:
            raise ValueError(
                f"The experiment_id {obs['experiment_id']!r} of observation "
                f"{obs['id']} is not the id of an experiment of the fit request."
            )
        validate_observation(experiments[obs["experiment_id"]], obs)
        _number(obs["scale"], "residual scale", positive=True)
        if obs["partition"] not in {"fit", "evaluation"}:
            raise ValueError(
                "The partition of an observation must be fit or evaluation."
            )
        fit_count += obs["partition"] == "fit"
    if fit_count <= len(names):
        raise ValueError(
            "The fit partition must contain more observations than there are "
            f"parameters; found {fit_count} observations and {len(names)} parameters."
        )
    if request["uncertainty"] not in {"descriptive", "independent-residuals"}:
        raise ValueError("uncertainty must be descriptive or independent-residuals.")
    if type(request["max_nfev"]) is not int or not 1 <= request["max_nfev"] <= 2000:
        raise ValueError("max_nfev must be an integer from 1 to 2000")


def _observed_value(result, observation):
    if observation["quantity"] == "effluent-concentration":
        return _named(result["collections"], observation["collection_id"])[
            "mean_concentrations_M"
        ][observation["target"]]
    time = observation["time_s"]
    event = next((e for e in result["events"] if e["before"]["time_s"] == time), None)
    row = (
        event[observation["state"]]
        if event is not None
        else next(r for r in result["trajectory"] if r["time_s"] == time)
    )
    target, quantity = observation["target"], observation["quantity"]
    if quantity == "concentration":
        return row["concentrations_M"][target]
    if quantity == "conversion":
        return 1 - row["concentrations_M"][target] / observation["reference_M"]
    if quantity == "removed":
        return row["removed_mol"][target]
    return row["catalysts"][target][
        "amount" if quantity == "catalyst-total" else "active_amount"
    ]


def fit_operation(request, *, method="Radau"):
    """Fit only the assigned measurements and retain all explicitly requested solves."""
    validate_fit_request(request)
    if method not in {"Radau", "DOP853"}:
        raise ValueError(
            f"The solver for operation fitting must be Radau or DOP853; found {method!r}."
        )
    import numpy as np
    import scipy
    from scipy.optimize import least_squares
    from scipy.stats import t

    parameters = request["parameters"]
    names = [p["id"] for p in parameters]
    scales = np.array([p["scale"] for p in parameters])
    lower = np.array([p["lower"] for p in parameters]) / scales
    upper = np.array([p["upper"] for p in parameters]) / scales
    starts = [np.array([s[n] for n in names]) / scales for s in request["starts"]]
    fit_obs = [o for o in request["observations"] if o["partition"] == "fit"]

    def operations(q, experiment_ids):
        ops = {
            o["id"]: copy.deepcopy(o)
            for o in request["experiments"]
            if o["id"] in experiment_ids
        }
        for parameter, value in zip(parameters, q * scales, strict=True):
            for target in parameter["targets"]:
                if target["experiment_id"] in ops:
                    parent, field, *_ = _target(
                        ops[target["experiment_id"]], target["path"]
                    )
                    parent[field] = float(value)
        return {key: simulate(op, method=method) for key, op in ops.items()}

    def residual(q):
        predictions = operations(q, {o["experiment_id"] for o in fit_obs})
        return np.array(
            [
                (_observed_value(predictions[o["experiment_id"]], o) - o["value"])
                / o["scale"]
                for o in fit_obs
            ]
        )

    def jacobian(q, free):
        columns = []
        for i in free:
            left, right = q.copy(), q.copy()
            left[i], right[i] = max(lower[i], q[i] - 1e-4), min(upper[i], q[i] + 1e-4)
            columns.append((residual(right) - residual(left)) / (right[i] - left[i]))
        return np.column_stack(columns)

    def solve(start, fixed=None):
        fixed = fixed or {}
        free = [i for i in range(len(names)) if i not in fixed]
        # Give bounded starts a finite trust radius in the declared scaled units.
        # Retain both requested and effective starts, including profile constraints.
        margin = np.minimum(1e-4, (upper - lower) / 4)
        effective = np.clip(start, lower + margin, upper - margin)
        for i, value in fixed.items():
            effective[i] = value
        record = {
            "start": dict(zip(names, map(float, start * scales), strict=True)),
            "effective_start": dict(
                zip(names, map(float, effective * scales), strict=True)
            ),
        }

        def expand(x):
            q = start.copy()
            q[free] = x
            for index, value in fixed.items():
                q[index] = value
            return q

        try:
            if free:
                result = least_squares(
                    lambda x: residual(expand(x)),
                    effective[free],
                    jac=lambda x: jacobian(expand(x), free),
                    bounds=(lower[free], upper[free]),
                    method="trf",
                    loss="linear",
                    x_scale=1.0,
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                    max_nfev=request["max_nfev"],
                )
                q = expand(result.x)
                success, message, evaluations = (
                    bool(result.success),
                    str(result.message),
                    int(result.nfev),
                )
            else:
                q = expand(np.array([]))
                success, message, evaluations = (
                    True,
                    "All parameters fixed at the profile value",
                    1,
                )
            values = residual(q)
            if not np.isfinite(q).all() or not np.isfinite(values).all():
                raise ValueError("The parameters and the residuals must be finite.")
            return {
                **record,
                "success": success,
                "message": message,
                "nfev": evaluations,
                "parameters": dict(zip(names, map(float, q * scales), strict=True)),
                "sse": float(values @ values),
            }
        except (ValueError, ArithmeticError) as exc:
            return {
                **record,
                "success": False,
                "message": str(exc),
                "parameters": None,
                "sse": None,
            }

    attempts = [
        {"start": dict(zip(names, map(float, s * scales), strict=True)), **solve(s)}
        for s in starts
    ]
    usable = [a for a in attempts if a["success"]]
    if not usable:
        raise OperationFitError(
            "No optimizer start converged. The messages of the starts are: "
            + "; ".join(a["message"] for a in attempts),
            attempts,
        )
    best = min(usable, key=lambda a: a["sse"])
    qhat = np.array([best["parameters"][n] for n in names]) / scales
    residuals = residual(qhat)
    jac = jacobian(qhat, list(range(len(names))))
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-7)) if singular[0] else 0
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    bound_distance = np.minimum(qhat - lower, upper - qhat)
    at_bound = [names[i] for i in range(len(names)) if bound_distance[i] <= 1e-4]
    df = len(fit_obs) - len(names)
    covariance, intervals, correlation = None, None, None
    if rank == len(names) and not at_bound:
        inverse = np.linalg.pinv(jac)
        base = inverse @ inverse.T
        std = np.sqrt(np.diag(base))
        correlation = (base / np.outer(std, std)).tolist()
        if request["uncertainty"] == "independent-residuals":
            cov = base * float(residuals @ residuals) / df * np.outer(scales, scales)
            covariance = cov.tolist()
            half = float(t.ppf(0.975, df)) * np.sqrt(np.diag(cov))
            intervals = {
                n: [float(v - h), float(v + h)]
                for n, v, h in zip(names, qhat * scales, half, strict=True)
            }
    profiles = {}
    for i, parameter in enumerate(parameters):
        if "profile_values" not in parameter:
            continue
        records = []
        for value in parameter["profile_values"]:
            trials = [solve(s, {i: value / scales[i]}) for s in [qhat, *starts]]
            valid = [a for a in trials if a["success"]]
            chosen = min(valid, key=lambda a: a["sse"]) if valid else None
            records.append(
                {
                    "value": value,
                    "attempts": trials,
                    "sse": chosen["sse"] if chosen else None,
                    "parameters": chosen["parameters"] if chosen else None,
                }
            )
        profiles[parameter["id"]] = records
    predicted = operations(qhat, {op["id"] for op in request["experiments"]})
    observations = []
    for obs in request["observations"]:
        value = _observed_value(predicted[obs["experiment_id"]], obs)
        observations.append(
            {
                **obs,
                "predicted": value,
                "residual": value - obs["value"],
                "standardized_residual": (value - obs["value"]) / obs["scale"],
            }
        )
    errors = {}
    for partition in ("fit", "evaluation"):
        values = [
            o["standardized_residual"]
            for o in observations
            if o["partition"] == partition
        ]
        errors[partition] = {
            "count": len(values),
            "standardized_rmse": math.sqrt(sum(v * v for v in values) / len(values))
            if values
            else None,
        }
    return {
        "schema_version": "0.1",
        "profile": PROFILE,
        "request": copy.deepcopy(request),
        "request_sha256": digest(request),
        "parameters": best["parameters"],
        "starts": attempts,
        "profiles": profiles,
        "observations": observations,
        "operations": predicted,
        "errors": errors,
        "diagnostics": {
            "rank": rank,
            "relative_rank_tolerance": 1e-7,
            "singular_values": singular.tolist(),
            "scaled_condition_number": condition
            if condition is None or math.isfinite(condition)
            else None,
            "parameter_order": names,
            "parameter_scales": scales.tolist(),
            "parameters_at_bounds": at_bound,
            "residual_degrees_of_freedom": df,
            "parameter_correlation": correlation,
            "local_covariance": covariance,
            "local_95_percent_intervals": intervals,
            "uncertainty_basis": request["uncertainty"],
        },
        "numerical": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": method,
            "optimizer": "TRF",
            "loss": "linear",
            "scaled_derivative_step": 1e-4,
            "ftol": 1e-9,
            "xtol": 1e-9,
            "gtol": 1e-9,
        },
    }
