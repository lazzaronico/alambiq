"""Compare explicit measurements with a saved operating trajectory."""

import copy
import hashlib
import json
import math
from collections import defaultdict

from alambiq.capabilities.operate.science.fitting import (
    _finite,
    _observed_value,
    validate_observation,
)
from alambiq.capabilities.operate.science.simulation import (
    PROFILE as OPERATION_PROFILE,
)
from alambiq.capabilities.operate.science.simulation import (
    _fields,
    _number,
    _operation_input,
    _records,
    source_text,
    validate_request,
)
from alambiq.capabilities.shared.records import digest, label

PROFILE = "recorded-operation-comparison-v1"
ENSEMBLE_PROFILE = "recorded-operation-ensemble-v1"


def validate_saved_operation(model):
    """Check the exact request and the recorded observation coordinates."""
    if (
        not isinstance(model, dict)
        or model.get("schema_version") != "0.1"
        or model.get("profile") != OPERATION_PROFILE
    ):
        raise ValueError(
            "The operation input must be a stored operation.simulate record "
            "(schema_version 0.1)."
        )
    request = model["request"]
    validate_request(request)
    if model.get("request_sha256") != digest(request):
        raise ValueError(
            "The request_sha256 of the stored operation differs from the digest of its "
            "request."
        )
    if [r["time_s"] for r in model["trajectory"]] != request["times_s"]:
        raise ValueError(
            "The trajectory of the stored operation must contain each requested time "
            "exactly once, in the order of the request."
        )
    if [e["request"] for e in model["events"]] != request["events"]:
        raise ValueError(
            "The events of the stored operation must equal the events of its request."
        )
    states = list(model["trajectory"])
    for event in model["events"]:
        for side in ("before", "after"):
            if event[side]["time_s"] != event["request"]["time_s"]:
                raise ValueError(
                    "The states before and after an exchange in the stored operation "
                    "must have the time of the exchange."
                )
            states.append(event[side])
        observed = next(
            (
                r
                for r in model["trajectory"]
                if r["time_s"] == event["request"]["time_s"]
            ),
            None,
        )
        if observed is not None and event["after"] != observed:
            raise ValueError(
                "The state after an exchange differs from the trajectory state at the "
                "same time."
            )
    species = {s["id"] for s in request["species"]}
    catalysts = {c["id"] for c in request["catalysts"]}
    for row in states:
        for key in ("concentrations_M", "removed_mol"):
            if set(row[key]) != species:
                raise ValueError(
                    "Every recorded state must contain every declared species."
                )
            for value in row[key].values():
                _finite(value, "recorded species quantity")
        if set(row["catalysts"]) != catalysts:
            raise ValueError(
                "Every recorded state must contain every declared catalyst."
            )
        for catalyst in row["catalysts"].values():
            for key in ("amount", "active_amount"):
                _finite(catalyst[key], "recorded catalyst amount")
    collections = model.get("collections", [])
    expected = request.get("collections", [])
    if [{k: c[k] for k in ("id", "start_s", "end_s")} for c in collections] != expected:
        raise ValueError(
            "The collection windows of the stored operation must equal those of its "
            "request."
        )
    for c in collections:
        _number(c["volume_L"], "collection volume", positive=True)
        if set(c["mean_concentrations_M"]) != species:
            raise ValueError(
                "Every recorded collection must contain every declared species."
            )
        for value in c["mean_concentrations_M"].values():
            _finite(value, "collected concentration")
    digest(model)


def validate_comparison(request, model=None, *, ensemble=False):
    _fields(
        request,
        {"id", "conditions", "observations"}
        | (set() if ensemble else {"operation_id"}),
    )
    label(request["id"], "comparison id")
    if not ensemble:
        label(request["operation_id"], "operation id")
    label(request["conditions"], "measurement conditions", 3000)
    if model is not None:
        if any(
            request[k] != model["request"][v]
            for k, v in (("operation_id", "id"), ("conditions", "conditions"))
        ):
            raise ValueError(
                "The operation_id and the conditions of the measurements must equal "
                "those of the stored request."
            )
    _records(request["observations"], "observations", 10000)
    for obs in request["observations"]:
        _fields(
            obs,
            {
                "id",
                "time_s",
                "state",
                "quantity",
                "target",
                "unit",
                "value",
                "partition",
                "source",
            },
            {"reference_M", "collection_id", "reported_error"},
        )
        if obs["partition"] not in {"calibration", "evaluation", "reference"}:
            raise ValueError(
                "The partition of an observation must be calibration, evaluation or "
                "reference."
            )
        if "reported_error" in obs:
            error = obs["reported_error"]
            _fields(error, {"kind", "value"})
            if error["kind"] not in {"SD", "SE"}:
                raise ValueError("The kind of a reported error must be SD or SE.")
            _number(error["value"], "reported error")
        if model is not None:
            validate_observation(model["request"], obs)


def compare_operation(model, request):
    """Read saved states, preserving source measurements and their error types."""
    validate_saved_operation(model)
    validate_comparison(request, model)
    rows, grouped = [], defaultdict(list)
    for observation in request["observations"]:
        predicted = _finite(_observed_value(model, observation), "prediction")
        row = copy.deepcopy(observation)
        row.update(
            predicted=predicted,
            residual=_finite(predicted - observation["value"], "residual"),
        )
        if "collection_id" in row:
            collection = next(
                c for c in model["collections"] if c["id"] == row["collection_id"]
            )
            row.update(
                collection_start_s=collection["start_s"],
                collection_volume_L=collection["volume_L"],
            )
        rows.append(row)
        grouped[
            row["partition"],
            row["quantity"],
            row["target"],
            row["unit"],
            row.get("reference_M"),
        ].append(row["residual"])
    return dict(
        schema_version="0.1",
        profile=PROFILE,
        request=copy.deepcopy(request),
        request_sha256=digest(request),
        operation_sha256=digest(model),
        operation_request=copy.deepcopy(model["request"]),
        operation_numerical=copy.deepcopy(model["numerical"]),
        observations=rows,
        errors=_errors(grouped),
        counts=dict(observations=len(rows), compared=len(rows)),
    )


def _errors(grouped):
    errors = []
    for (partition, quantity, target, unit, reference), residuals in sorted(
        grouped.items(), key=lambda item: json.dumps(item[0])
    ):
        maximum = max(abs(v) for v in residuals)
        rmse = (
            maximum
            * math.sqrt(
                math.fsum((v / maximum) ** 2 for v in residuals) / len(residuals)
            )
            if maximum
            else 0.0
        )
        errors.append(
            dict(
                partition=partition,
                quantity=quantity,
                target=target,
                unit=unit,
                reference_M=reference,
                count=len(residuals),
                rmse=rmse,
            )
        )
    return errors


def validate_ensemble(ensemble, request=None):
    _fields(ensemble, {"id", "source", "members"})
    label(ensemble["id"], "ensemble id")
    label(ensemble["source"], "ensemble source", 3000)
    _records(ensemble["members"], "ensemble members", 1000)
    if request is not None:
        validate_comparison(request, ensemble=True)
    for member in ensemble["members"]:
        _fields(member, {"id", "operation"})
        validate_saved_operation(member["operation"])
        if request is not None:
            validate_comparison(
                {**request, "operation_id": member["operation"]["request"]["id"]},
                member["operation"],
            )


def compare_ensemble(ensemble, request):
    """Compare every explicitly supplied member using the same observations."""
    validate_ensemble(ensemble, request)
    members, rows = [], None
    for member in ensemble["members"]:
        compared = compare_operation(
            member["operation"],
            {**request, "operation_id": member["operation"]["request"]["id"]},
        )
        if rows is None:
            rows = copy.deepcopy(compared["observations"])
        for expected, observed in zip(rows, compared["observations"], strict=True):
            if any(
                expected.get(k) != observed.get(k)
                for k in ("collection_start_s", "collection_volume_L")
            ):
                raise ValueError(
                    "Effluent observations of an ensemble require the same collection "
                    "interval and volume in every member."
                )
        members.append(
            dict(
                id=member["id"],
                **{
                    k: compared[k]
                    for k in (
                        "operation_sha256",
                        "operation_request",
                        "operation_numerical",
                        "errors",
                    )
                },
                predictions=[r["predicted"] for r in compared["observations"]],
            )
        )
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        values = sorted(m["predictions"][i] for m in members)
        middle = len(values) // 2
        median = (
            values[middle]
            if len(values) % 2
            else math.fsum(v / 2 for v in values[middle - 1 : middle + 1])
        )
        row.update(
            minimum=values[0],
            maximum=values[-1],
            predicted=median,
            residual=_finite(median - row["value"], "median residual"),
        )
        grouped[
            row["partition"],
            row["quantity"],
            row["target"],
            row["unit"],
            row.get("reference_M"),
        ].append(row["residual"])
    return dict(
        schema_version="0.1",
        profile=ENSEMBLE_PROFILE,
        request=copy.deepcopy(request),
        request_sha256=digest(request),
        ensemble_id=ensemble["id"],
        ensemble_source=ensemble["source"],
        ensemble_sha256=digest(ensemble),
        members=members,
        observations=rows,
        errors=_errors(grouped),
        counts=dict(
            members=len(members),
            observations=len(rows),
            compared=len(members) * len(rows),
        ),
    )


def _ensemble_input(value, *, freeze=False):
    """Resolve ordinary per-member descriptors or freeze their exact bytes."""
    ensemble = copy.deepcopy(_operation_input(value))
    for member in ensemble["members"]:
        operation = member["operation"]
        if isinstance(operation, dict) and "text" in operation:
            text = source_text(operation)
            member["operation"] = (
                {**operation, "sha256": hashlib.sha256(text.encode()).hexdigest()}
                if freeze
                else json.loads(text)
            )
    return ensemble
