"""Explicit output, supplied-enzyme turnover and sampled-threshold accounting."""

import copy
import math

from alambiq.capabilities.operate.science.comparison import validate_saved_operation
from alambiq.capabilities.operate.science.fitting import _finite
from alambiq.capabilities.operate.science.resources import _quantity
from alambiq.capabilities.operate.science.simulation import (
    _fields,
    _number,
    _records,
)
from alambiq.capabilities.shared.records import digest, label, signed_coefficient

PROFILE = "recorded-operating-performance-v1"
MASS_G = {"kg": 1000, "g": 1, "mg": 1e-3, "ug": 1e-6}
AMOUNT_MOL = {"mol": 1, "mmol": 1e-3, "umol": 1e-6, "nmol": 1e-9}


def validate_assessment(request, model=None):
    _fields(
        request,
        {
            "id",
            "operation_id",
            "conditions",
            "product",
            "output",
            "catalysts",
            "lifetime",
        },
        {"additional_time"},
    )
    for key in ("id", "operation_id"):
        label(request[key], key)
    label(request["conditions"], "conditions", 3000)
    product = request["product"]
    _fields(product, {"species", "recovery_fraction", "accepted_fraction", "source"})
    label(product["species"], "target species")
    label(product["source"], "product handling source", 3000)
    for key in ("recovery_fraction", "accepted_fraction"):
        _number(product[key], key, fraction=True)
    selected = request["output"]
    _fields(selected, {"final_reactor", "exchanges_s", "collections"})
    if type(selected["final_reactor"]) is not bool:
        raise ValueError("output.final_reactor must be true or false.")
    for key in ("exchanges_s", "collections"):
        values = selected[key]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError(f"output.{key} must be a list without repeated entries.")
        for value in values:
            if key == "exchanges_s":
                _number(value, "exchange time", positive=True)
            else:
                label(value, "collection id")
    _records(request["catalysts"], "assessed catalysts", 50, empty=True)
    for cat in request["catalysts"]:
        _fields(
            cat, {"id", "active_sites_per_molecule", "source"}, {"molecular_mass_g_mol"}
        )
        _number(
            cat["active_sites_per_molecule"], "active sites per molecule", positive=True
        )
        if "molecular_mass_g_mol" in cat:
            _number(cat["molecular_mass_g_mol"], "molecular mass", positive=True)
        label(cat["source"], "molecular basis source", 3000)
    _records(request["lifetime"], "lifetime thresholds", 50, empty=True)
    for row in request["lifetime"]:
        _fields(row, {"id", "catalyst", "minimum_active_amount", "source"})
        label(row["catalyst"], "threshold catalyst")
        label(row["source"], "threshold source", 3000)
        _number(row["minimum_active_amount"], "useful active amount", positive=True)
    if "additional_time" in request:
        _quantity(request["additional_time"], "s")
    if model is None:
        return
    origin = model["request"]
    if (
        request["operation_id"] != origin["id"]
        or request["conditions"] != origin["conditions"]
    ):
        raise ValueError(
            "The operation_id and the conditions of the assessment must equal those of "
            "the stored operation."
        )
    if product["species"] not in {s["id"] for s in origin["species"]}:
        raise ValueError(
            "product.species must be a species declared in the stored operation."
        )
    catalysts = {c["id"] for c in origin["catalysts"]}
    if any(c["id"] not in catalysts for c in request["catalysts"]) or any(
        r["catalyst"] not in catalysts for r in request["lifetime"]
    ):
        raise ValueError(
            "Every catalyst in catalysts and lifetime must be declared in the stored "
            "operation."
        )
    if set(selected["exchanges_s"]) - {e["time_s"] for e in origin["events"]}:
        raise ValueError(
            "Every time in output.exchanges_s must be the time of an exchange of the "
            "stored operation."
        )
    collections = {c["id"]: c for c in model.get("collections", [])}
    if set(selected["collections"]) - collections.keys():
        raise ValueError(
            "Every identifier in output.collections must be a collection of the stored "
            "operation."
        )
    windows = sorted(
        (collections[k] for k in selected["collections"]), key=lambda c: c["start_s"]
    )
    if any(
        a["end_s"] > b["start_s"] for a, b in zip(windows, windows[1:], strict=False)
    ):
        raise ValueError(
            "The windows of the selected collections overlap; the selected collections "
            "must have disjoint windows."
        )


def _close(actual, expected, name):
    _finite(actual, name)
    _finite(expected, name)
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError(
            f"The recorded {name} does not satisfy the balance within a relative "
            "tolerance of 1e-8."
        )


def _amount(model, state, target):
    volume = state.get("volume_L", model["request"]["volume_L"])
    _number(volume, "recorded liquid volume", positive=True)
    amount = _finite(state["concentrations_M"][target] * volume, "species amount")
    if "amounts_mol" in state:
        _close(state["amounts_mol"][target], amount, "amount/concentration")
    return amount


def _catalyst_inventory(catalyst, row):
    for key in (
        "amount",
        "active_amount",
        "added_amount",
        "added_active_amount",
        "removed_amount",
        "removed_active_amount",
        "deactivated_amount",
    ):
        _number(row[key], key)
    if row["amount_unit"] != catalyst["amount_unit"]:
        raise ValueError(
            "The recorded catalyst amount unit differs from the unit in the request of "
            "the stored operation."
        )
    total = catalyst["amount"] + row["added_amount"]
    active = (
        catalyst["amount"] * catalyst["active_fraction"] + row["added_active_amount"]
    )
    _close(row["amount"] + row["removed_amount"], total, "total catalyst balance")
    _close(
        row["active_amount"] + row["removed_active_amount"] + row["deactivated_amount"],
        active,
        "active catalyst balance",
    )
    for active_amount, total_amount in (
        (active, total),
        (row["active_amount"], row["amount"]),
    ):
        if active_amount - total_amount > max(1e-12, 1e-8 * total_amount):
            raise ValueError(
                "The active catalyst amount exceeds the total catalyst amount."
            )
    return total, active


def _lifetime(model, threshold):
    catalyst = next(
        c for c in model["request"]["catalysts"] if c["id"] == threshold["catalyst"]
    )
    events = {e["request"]["time_s"]: e for e in model["events"]}
    states = {r["time_s"]: r for r in model["trajectory"]}
    points = []
    for time in sorted(states.keys() | events.keys()):
        values = (
            [(side, events[time][side]) for side in ("before", "after")]
            if time in events
            else [("recorded", states[time])]
        )
        for side, state in values:
            _catalyst_inventory(catalyst, state["catalysts"][catalyst["id"]])
            amount = _finite(
                state["catalysts"][threshold["catalyst"]]["active_amount"],
                "active amount",
            )
            points.append(
                dict(
                    time_s=time,
                    state=side,
                    active_amount=amount,
                    above_threshold=amount >= threshold["minimum_active_amount"],
                )
            )
    crossings = [
        dict(
            direction="above" if b["above_threshold"] else "below",
            start_s=a["time_s"],
            end_s=b["time_s"],
            exact_event=a["time_s"] == b["time_s"],
        )
        for a, b in zip(points, points[1:], strict=False)
        if a["above_threshold"] != b["above_threshold"]
    ]
    first = next((c for c in crossings if c["direction"] == "below"), None)
    below_at_start = not points[0]["above_threshold"]
    return dict(
        **copy.deepcopy(threshold),
        amount_unit=next(
            c["amount_unit"]
            for c in model["request"]["catalysts"]
            if c["id"] == threshold["catalyst"]
        ),
        status="below_at_start"
        if below_at_start
        else "observed_below_threshold"
        if first
        else "above_at_all_recorded_states",
        first_below_interval_s=[0, 0]
        if below_at_start
        else [first["start_s"], first["end_s"]]
        if first
        else None,
        horizon_s=points[-1]["time_s"],
        points=points,
        crossings=crossings,
        interpretation="A catalyst is useful while its active amount is at or above the supplied threshold. A transition is bracketed by two adjacent recorded states; a change at a discrete event has the exact event time. Recording more states resolves the behavior between them and further crossings.",
    )


def summarize_operation(model, request):
    validate_saved_operation(model)
    validate_assessment(request, model)
    origin, last = model["request"], model["trajectory"][-1]
    target = request["product"]["species"]
    initial = (
        next(s["initial_M"] for s in origin["species"] if s["id"] == target)
        * origin["volume_L"]
    )
    fed = _number(last["feed_mol"][target], "target fed")
    removed = _number(last["removed_mol"][target], "target removed")
    final = _amount(model, last, target)
    channels = []
    for channel in origin["channels"]:
        extent = _number(last["reaction_extent_mol"][channel["id"]], "reaction extent")
        coefficient = (
            float(signed_coefficient(channel["stoichiometry"][target]))
            if target in channel["stoichiometry"]
            else 0.0
        )
        channels.append(
            dict(
                id=channel["id"],
                catalyst=channel.get("catalyst"),
                extent_mol=extent,
                target_change_mol=coefficient * extent,
            )
        )
    formed = math.fsum(max(0, c["target_change_mol"]) for c in channels)
    consumed = math.fsum(max(0, -c["target_change_mol"]) for c in channels)
    net = formed - consumed
    _close(final + removed, initial + fed + net, "target material balance")
    outputs = []
    if request["output"]["final_reactor"]:
        outputs.append(
            dict(
                kind="reactor",
                id="final-reactor",
                start_s=last["time_s"],
                end_s=last["time_s"],
                amount_mol=final,
            )
        )
    for time in request["output"]["exchanges_s"]:
        event = next(e for e in model["events"] if e["request"]["time_s"] == time)
        amount = (
            _amount(model, event["before"], target)
            * event["request"]["exchange_fraction"]
        )
        _close(
            event["after"]["removed_mol"][target]
            - event["before"]["removed_mol"][target],
            amount,
            "exchange removal",
        )
        outputs.append(
            dict(
                kind="exchange",
                id=str(time),
                start_s=time,
                end_s=time,
                amount_mol=amount,
            )
        )
    for identifier in request["output"]["collections"]:
        c = next(c for c in model["collections"] if c["id"] == identifier)
        amount = _finite(c["amounts_mol"][target], "collected target amount")
        _close(
            amount,
            c["mean_concentrations_M"][target] * c["volume_L"],
            "collected amount/concentration",
        )
        outputs.append(
            dict(
                kind="collection",
                id=identifier,
                start_s=c["start_s"],
                end_s=c["end_s"],
                amount_mol=amount,
            )
        )
    selected = math.fsum(r["amount_mol"] for r in outputs)
    if selected < -1e-12 or selected - (final + removed) > max(
        1e-12, 1e-8 * abs(final + removed)
    ):
        raise ValueError(
            "The selected output exceeds the amount of the target available in the "
            "stored operation, the final amount plus the removed amount."
        )
    selected = max(0, selected)
    useful = (
        selected
        * request["product"]["recovery_fraction"]
        * request["product"]["accepted_fraction"]
    )
    time = last["time_s"]
    total_time = (
        time + _quantity(request["additional_time"], "s")
        if "additional_time" in request
        else None
    )
    result = dict(
        schema_version="0.1",
        profile=PROFILE,
        request=copy.deepcopy(request),
        request_sha256=digest(request),
        operation_sha256=digest(model),
        operation_request=copy.deepcopy(origin),
        operation_numerical=copy.deepcopy(model["numerical"]),
        product=dict(
            species=target,
            initial_mol=initial,
            fed_mol=fed,
            final_mol=final,
            removed_mol=removed,
            gross_formed_mol=formed,
            gross_consumed_mol=consumed,
            net_formed_mol=net,
            balance_residual_mol=final + removed - initial - fed - net,
            selected_output_mol=selected,
            useful_output_mol=useful,
            outputs=outputs,
            reaction_elapsed_s=time,
            process_elapsed_s=total_time,
            reaction_time_productivity_mol_s=useful / time,
            process_time_productivity_mol_s=useful / total_time
            if total_time is not None
            else None,
        ),
        catalysts=[],
        lifetimes=[_lifetime(model, r) for r in request["lifetime"]],
        diagnostics=[],
    )
    for molecular in request["catalysts"]:
        catalyst = next(c for c in origin["catalysts"] if c["id"] == molecular["id"])
        row = last["catalysts"][catalyst["id"]]
        unit = catalyst["amount_unit"]
        total, active = _catalyst_inventory(catalyst, row)
        factor = AMOUNT_MOL.get(unit)
        if factor is None and unit in MASS_G and "molecular_mass_g_mol" in molecular:
            factor = MASS_G[unit] / molecular["molecular_mass_g_mol"]
        sites = (
            active * factor * molecular["active_sites_per_molecule"]
            if factor is not None
            else None
        )
        if sites is None or sites == 0:
            result["diagnostics"].append(
                f"{catalyst['id']}: the molar turnover requires a supported amount unit, "
                "or a mass unit together with molecular_mass_g_mol, and "
                "active_sites_per_molecule above 0."
            )
        result["catalysts"].append(
            dict(
                id=catalyst["id"],
                amount_unit=unit,
                total_supplied_amount=total,
                active_supplied_amount=active,
                final_amount=row["amount"],
                final_active_amount=row["active_amount"],
                supplied_active_site_mol=sites,
                net_target_equivalents_per_supplied_active_site=net / sites
                if sites
                else None,
                useful_output_per_supplied_active_site=useful / sites
                if sites
                else None,
                channel_turnovers=[
                    dict(
                        id=c["id"],
                        extent_mol=c["extent_mol"],
                        turnovers_per_supplied_active_site=c["extent_mol"] / sites
                        if sites
                        else None,
                    )
                    for c in channels
                    if c["catalyst"] == catalyst["id"]
                ],
            )
        )
    if total_time is None:
        result["diagnostics"].append(
            "The process-time productivity requires additional_time, which is not "
            "supplied."
        )
    result["method"] = (
        "The net target formation is the sum of the signed channel extents, which equals the final amount plus the removed amount minus the initial amount and the fed amount of the target. The selected output combines the chosen final contents, exchanges and disjoint continuous collections. The useful output applies the declared recovery and accepted fractions. The catalyst denominators include the initial active catalyst and all added active catalyst, converted to active sites with the declared active_sites_per_molecule. Channel turnovers and net target equivalents are distinct quantities."
    )
    digest(result)
    return result
