"""Explicit well-mixed reaction operation with material and catalyst ledgers."""

from __future__ import annotations

import copy
import json
import math
from fractions import Fraction

from alambiq.capabilities.shared.records import digest, label, signed_coefficient
from alambiq.capabilities.shared.sources import source_text as source_text

PROFILE = "well-mixed-reaction-operation-v1"


def _number(value, name, *, positive=False, fraction=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
        or (fraction and value > 1)
    ):
        raise ValueError(
            f"{name} must be a finite number {'above 0' if positive else 'of at least 0'}"
            + (" and at most 1" if fraction else "")
            + "."
        )
    return float(value)


def _fields(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or value.keys() - set(required) - set(optional)
    ):
        needed = (
            f"must have the fields {', '.join(sorted(required))}" if required else ""
        )
        allowed = (
            f"may have the fields {', '.join(sorted(optional))}" if optional else ""
        )
        raise ValueError(
            "A record "
            + (needed + (" and " + allowed if allowed else "") if needed else allowed)
            + ". No other fields are accepted."
        )


def _records(values, name, maximum, *, empty=False):
    if (
        not isinstance(values, list)
        or not (0 if empty else 1) <= len(values) <= maximum
    ):
        raise ValueError(
            f"{name} must be a list of {0 if empty else 1} to {maximum} records."
        )
    names = []
    for record in values:
        if not isinstance(record, dict):
            raise ValueError(f"Each entry of {name} must be a record with an id.")
        label(record.get("id"), f"{name}.id")
        names.append(record["id"])
    if len(names) != len(set(names)):
        raise ValueError(f"The id values in {name} must be unique.")
    return set(names)


def _quantities(value, allowed, name, *, positive=False):
    if not isinstance(value, dict) or value.keys() - allowed:
        raise ValueError(f"{name} must map known identifiers to numbers.")
    for key, quantity in value.items():
        _number(quantity, f"{name}.{key}", positive=positive)


GAS_CONSTANT = 8.314462618


def _arrhenius_fields(value, name):
    _fields(value, {"activation_energy_J_mol", "reference_K"})
    if type(value["activation_energy_J_mol"]) not in (int, float) or not math.isfinite(
        value["activation_energy_J_mol"]
    ):
        raise ValueError(f"{name} activation_energy_J_mol must be a finite number.")
    _number(value["reference_K"], f"{name} reference_K", positive=True)


def validate_conditions(request):
    """Check the stated conditions and the dependences that use them."""
    state = request.get("state")
    needs_temperature = any("temperature" in c for c in request["channels"]) or any(
        "deactivation_temperature" in c for c in request["catalysts"]
    )
    needs_ph = any("ph" in c for c in request["channels"])
    if state is None:
        if needs_temperature or needs_ph:
            raise ValueError(
                "A temperature or pH dependence requires the operating conditions in "
                "state."
            )
        return
    _fields(state, set(), {"temperature_K", "pH"})
    if "temperature_K" in state:
        _number(state["temperature_K"], "state temperature_K", positive=True)
    if "pH" in state and (
        type(state["pH"]) not in (int, float) or not math.isfinite(state["pH"])
    ):
        raise ValueError("state pH must be a finite number.")
    if needs_temperature and "temperature_K" not in state:
        raise ValueError("A temperature dependence requires state.temperature_K.")
    if needs_ph and "pH" not in state:
        raise ValueError("A pH dependence requires state.pH.")
    for channel in request["channels"]:
        if "temperature" in channel:
            _arrhenius_fields(channel["temperature"], f"channel {channel['id']}")
        if "ph" in channel:
            _fields(channel["ph"], {"pKa1", "pKa2", "reference_pH"})
            for key, value in channel["ph"].items():
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(
                        f"channel {channel['id']} ph.{key} must be a finite number."
                    )
    for catalyst in request["catalysts"]:
        if "deactivation_temperature" in catalyst:
            _arrhenius_fields(
                catalyst["deactivation_temperature"], f"catalyst {catalyst['id']}"
            )


def ph_activity(ph, pka1, pka2):
    """The bell-shaped relative activity 1 / (1 + 10^(pKa1 - pH) + 10^(pH - pKa2))."""
    return 1.0 / (1.0 + 10.0 ** (pka1 - ph) + 10.0 ** (ph - pka2))


def apply_conditions(request):
    """The request with every dependence on the stated conditions applied.

    A channel's capacity and a catalyst's deactivation rate are written at reference
    conditions. Where a dependence is declared, the stated temperature or pH scales
    them: an Arrhenius factor exp(-Ea/R (1/T - 1/Tref)) for temperature, and the ratio of
    the bell-shaped activity at the stated pH to that at the reference pH. Returns the
    resolved request and a record of every multiplier.
    """
    state = request.get("state")
    applied = {"state": copy.deepcopy(state), "multipliers": []}
    if state is None:
        return request, applied
    resolved = copy.deepcopy(request)
    for channel in resolved["channels"]:
        multiplier = 1.0
        if "temperature" in channel:
            t = channel["temperature"]
            factor = math.exp(
                -t["activation_energy_J_mol"]
                / GAS_CONSTANT
                * (1.0 / state["temperature_K"] - 1.0 / t["reference_K"])
            )
            applied["multipliers"].append(
                {"channel": channel["id"], "kind": "temperature", "multiplier": factor}
            )
            multiplier *= factor
        if "ph" in channel:
            p = channel["ph"]
            factor = ph_activity(state["pH"], p["pKa1"], p["pKa2"]) / ph_activity(
                p["reference_pH"], p["pKa1"], p["pKa2"]
            )
            applied["multipliers"].append(
                {"channel": channel["id"], "kind": "pH", "multiplier": factor}
            )
            multiplier *= factor
        channel["capacity_M_s"] = channel["capacity_M_s"] * multiplier
    for catalyst in resolved["catalysts"]:
        if "deactivation_temperature" in catalyst:
            t = catalyst["deactivation_temperature"]
            factor = math.exp(
                -t["activation_energy_J_mol"]
                / GAS_CONSTANT
                * (1.0 / state["temperature_K"] - 1.0 / t["reference_K"])
            )
            applied["multipliers"].append(
                {
                    "catalyst": catalyst["id"],
                    "kind": "deactivation temperature",
                    "multiplier": factor,
                }
            )
            catalyst["deactivation_per_s"] = catalyst["deactivation_per_s"] * factor
    return resolved, applied


def flow_schedule(request):
    """Resolve each constant-flow phase at its actual starting liquid volume."""
    rows, start, volume = [], 0.0, request["volume_L"]
    for phase in request["phases"]:
        if "dilution_per_s" in phase:
            inlet = outlet = phase["dilution_per_s"] * volume
        else:
            inlet, outlet = phase["inlet_L_s"], phase["outlet_L_s"]
        end_volume = volume + (inlet - outlet) * (phase["end_s"] - start)
        _number(inlet, "inlet flow_L_s")
        _number(outlet, "outlet flow_L_s")
        _number(end_volume, "phase final volume_L", positive=True)
        rows.append(
            dict(
                start_s=start,
                end_s=phase["end_s"],
                start_volume_L=volume,
                end_volume_L=end_volume,
                inlet_L_s=inlet,
                outlet_L_s=outlet,
            )
        )
        start, volume = phase["end_s"], end_volume
    return rows


def collection_volume(request, collection):
    """Exact piecewise-constant outlet volume over a declared window, in liters."""
    volumes = []
    for phase in flow_schedule(request):
        overlap = max(
            0.0,
            min(collection["end_s"], phase["end_s"])
            - max(collection["start_s"], phase["start_s"]),
        )
        volumes.append(phase["outlet_L_s"] * overlap)
    return math.fsum(volumes)


def validate_request(request):
    """Validate exact channel stoichiometry, physical units and event ordering."""
    _fields(
        request,
        {
            "id",
            "volume_L",
            "conditions",
            "species",
            "catalysts",
            "channels",
            "phases",
            "events",
            "times_s",
        },
        {"collections", "state"},
    )
    label(request["id"], "id")
    label(request["conditions"], "conditions", 3000)
    _number(request["volume_L"], "volume_L", positive=True)
    species = _records(request["species"], "species", 200)
    catalysts = _records(request["catalysts"], "catalysts", 50, empty=True)
    channels = _records(request["channels"], "channels", 200)
    pools = {}
    for row in request["species"]:
        _fields(row, {"id", "initial_M", "pools"})
        _number(row["initial_M"], "initial_M")
        if not isinstance(row["pools"], dict) or len(row["pools"]) > 100:
            raise ValueError(
                f"The pools of species {row['id']} must be a mapping of at most 100 "
                "pool names to conserved-moiety amounts."
            )
        for name, value in row["pools"].items():
            label(name, "pool")
            _number(value, "pool weight")
            pools.setdefault(name, {})[row["id"]] = Fraction(str(value))
    for row in request["catalysts"]:
        _fields(
            row,
            {
                "id",
                "amount",
                "amount_unit",
                "active_fraction",
                "reference_amount",
                "deactivation_per_s",
                "washout_fraction",
            },
            {"deactivation_temperature"},
        )
        label(row["amount_unit"], "amount_unit", 100)
        _number(row["amount"], "catalyst amount")
        _number(row["reference_amount"], "reference_amount", positive=True)
        _number(row["active_fraction"], "active_fraction", fraction=True)
        _number(row["deactivation_per_s"], "deactivation_per_s")
        _number(row["washout_fraction"], "washout_fraction", fraction=True)
    for row in request["channels"]:
        _fields(
            row,
            {"id", "reaction_id", "stoichiometry", "capacity_M_s", "factors", "source"},
            {"catalyst", "inhibitors_M", "denominator", "temperature", "ph"},
        )
        label(row["reaction_id"], "reaction_id")
        label(row["source"], "channel source", 3000)
        _number(row["capacity_M_s"], "capacity_M_s")
        if "catalyst" in row and row["catalyst"] not in catalysts:
            raise ValueError(
                f"The catalyst of channel {row['id']} must be a declared catalyst."
            )
        stoich = row["stoichiometry"]
        if not isinstance(stoich, dict) or len(stoich) < 2 or stoich.keys() - species:
            raise ValueError(
                f"The stoichiometry of channel {row['id']} must name at least 2 "
                "declared species."
            )
        coefficients = {k: signed_coefficient(v) for k, v in stoich.items()}
        consumed = {k for k, v in coefficients.items() if v < 0}
        if not consumed or not any(v > 0 for v in coefficients.values()):
            raise ValueError(
                f"Channel {row['id']} must consume at least 1 species and produce at "
                "least 1 species."
            )
        for name, weights in pools.items():
            if sum(weights.get(k, 0) * v for k, v in coefficients.items()) != 0:
                raise ValueError(
                    f"Channel {row['id']} changes the declared conserved pool {name}."
                )
        factors = row["factors"]
        if not isinstance(factors, list) or not 1 <= len(factors) <= len(species):
            raise ValueError(
                "Each consumed species requires a concentration-dependent factor"
            )
        factor_names = []
        for factor in factors:
            _fields(
                factor,
                {"species", "kind", "scale_M"},
                {
                    "power",
                    "denominator_power",
                    "competitive_inhibitors_M",
                    "uncompetitive_inhibitors_M",
                    "self_inhibition_M",
                },
            )
            if factor["species"] not in species or factor["kind"] not in {
                "mass-action",
                "saturation",
            }:
                raise ValueError(
                    f"Each factor of channel {row['id']} must name a declared species "
                    "and have the kind mass-action or saturation."
                )
            factor_names.append(factor["species"])
            _number(factor["scale_M"], "factor scale_M", positive=True)
            if type(factor.get("power", 1)) is not int or factor.get(
                "power", 1
            ) not in (1, 2):
                raise ValueError(
                    f"The power of a factor of channel {row['id']} must be 1 or 2."
                )
            if factor["kind"] == "mass-action" and (
                set(factor)
                & {
                    "competitive_inhibitors_M",
                    "uncompetitive_inhibitors_M",
                    "self_inhibition_M",
                    "denominator_power",
                }
            ):
                raise ValueError(
                    f"Inhibition terms and denominator_power are accepted only for "
                    f"saturation factors; check channel {row['id']}."
                )
            if type(factor.get("denominator_power", 1)) is not int or factor.get(
                "denominator_power", 1
            ) not in (1, 2):
                raise ValueError(
                    f"The denominator_power of a factor of channel {row['id']} must be "
                    "1 or 2."
                )
            for kind in ("competitive", "uncompetitive"):
                _quantities(
                    factor.get(f"{kind}_inhibitors_M", {}),
                    species,
                    f"{kind} inhibitor Ki_M",
                    positive=True,
                )
            if "self_inhibition_M" in factor:
                _number(
                    factor["self_inhibition_M"],
                    "substrate inhibition Ki_M",
                    positive=True,
                )
        if len(factor_names) != len(set(factor_names)) or consumed - set(factor_names):
            raise ValueError(
                f"Channel {row['id']} must have one factor per named species and a "
                "factor for every consumed species."
            )
        _quantities(
            row.get("inhibitors_M", {}), species, "rate inhibitor Ki_M", positive=True
        )
        if "denominator" in row:
            if any(f["kind"] != "mass-action" for f in factors):
                raise ValueError(
                    f"Channel {row['id']} has a denominator, so all its factors must "
                    "have the kind mass-action."
                )
            denominator = row["denominator"]
            _fields(denominator, {"constant", "terms"})
            _number(denominator["constant"], "denominator constant", positive=True)
            _records(denominator["terms"], "denominator terms", 100)
            for term in denominator["terms"]:
                _fields(term, {"id", "coefficient", "factors"})
                _number(term["coefficient"], "denominator coefficient", positive=True)
                parts = term["factors"]
                if not isinstance(parts, list) or not 1 <= len(parts) <= len(species):
                    raise ValueError(
                        f"Each denominator term of channel {row['id']} must have 1 to "
                        f"{len(species)} concentration factors."
                    )
                names = []
                for part in parts:
                    _fields(part, {"species", "scale_M"}, {"power"})
                    if part["species"] not in species:
                        raise ValueError(
                            "Each denominator factor must name a declared species."
                        )
                    names.append(part["species"])
                    _number(part["scale_M"], "denominator scale_M", positive=True)
                    if type(part.get("power", 1)) is not int or part.get(
                        "power", 1
                    ) not in (1, 2):
                        raise ValueError(
                            "The power of a denominator factor must be 1 or 2."
                        )
                if len(names) != len(set(names)):
                    raise ValueError(
                        "Each denominator term must have one factor per species."
                    )
    validate_conditions(request)
    times = request["times_s"]
    if not isinstance(times, list) or not 2 <= len(times) <= 10000:
        raise ValueError("times_s must be a list of 2 to 10,000 observation times.")
    for t in times:
        _number(t, "time_s")
    if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("times_s must start at 0 and increase strictly.")
    phases = request["phases"]
    if not isinstance(phases, list) or not 1 <= len(phases) <= 100:
        raise ValueError("phases must be a list of 1 to 100 consecutive phases.")
    prior = 0
    for phase in phases:
        _fields(
            phase,
            {"end_s", "feed_M", "active_channels"},
            {
                "dilution_per_s",
                "inlet_L_s",
                "outlet_L_s",
                "catalyst_feed",
                "solute_input_mol_s",
            },
        )
        _number(phase["end_s"], "phase end_s", positive=True)
        if "dilution_per_s" in phase:
            if "inlet_L_s" in phase or "outlet_L_s" in phase:
                raise ValueError(
                    "A phase has either dilution_per_s or the fields inlet_L_s and "
                    "outlet_L_s, not both."
                )
            _number(phase["dilution_per_s"], "dilution_per_s")
        else:
            _number(phase.get("inlet_L_s"), "inlet_L_s")
            _number(phase.get("outlet_L_s"), "outlet_L_s")
        _quantities(phase["feed_M"], species, "feed_M")
        direct = phase.get("solute_input_mol_s", {})
        _quantities(direct, species, "solute input mol/s")
        for value in direct.values():
            _number(value / request["volume_L"], "normalized solute input M/s")
        catalyst_inlet = phase.get("catalyst_feed", {})
        if not isinstance(catalyst_inlet, dict) or catalyst_inlet.keys() - catalysts:
            raise ValueError(
                "catalyst_feed must map declared catalysts to feed amounts."
            )
        for value in catalyst_inlet.values():
            _fields(value, {"amount_per_L", "active_fraction"})
            _number(value["amount_per_L"], "catalyst feed amount_per_L")
            _number(
                value["active_fraction"], "catalyst feed active_fraction", fraction=True
            )
        active = phase["active_channels"]
        if (
            not isinstance(active, list)
            or any(not isinstance(a, str) for a in active)
            or set(active) - channels
            or len(active) != len(set(active))
        ):
            raise ValueError(
                "active_channels must be a list of distinct declared channel ids."
            )
        if phase["end_s"] <= prior:
            raise ValueError("The end_s values of the phases must increase strictly.")
        prior = phase["end_s"]
    if prior != times[-1]:
        raise ValueError(
            "The end_s of the final phase must equal the last time of times_s."
        )
    flow_schedule(request)
    collections = request.get("collections", [])
    _records(collections, "collections", 200, empty=True)
    for collection in collections:
        _fields(collection, {"id", "start_s", "end_s"})
        a = _number(collection["start_s"], "collection start_s")
        b = _number(collection["end_s"], "collection end_s")
        if not a < b <= times[-1]:
            raise ValueError(
                "Each collection window must satisfy start_s < end_s <= the last time "
                "of times_s."
            )
        _number(
            collection_volume(request, collection), "collection volume_L", positive=True
        )
    events = request["events"]
    if not isinstance(events, list) or len(events) > 200:
        raise ValueError("events must be a list of at most 200 liquid-exchange events.")
    prior = 0
    for event in events:
        _fields(
            event,
            {
                "time_s",
                "exchange_fraction",
                "feed_M",
                "catalyst_retention",
                "catalyst_additions",
            },
            {"catalyst_activity_retention"},
        )
        t = _number(event["time_s"], "event time_s", positive=True)
        if not prior < t <= times[-1]:
            raise ValueError(
                "The time_s values of the events must increase strictly, above 0 and "
                "up to the last time of times_s."
            )
        prior = t
        _number(event["exchange_fraction"], "exchange_fraction", fraction=True)
        _quantities(event["feed_M"], species, "event feed_M")
        retention = event["catalyst_retention"]
        if not isinstance(retention, dict) or set(retention) != catalysts:
            raise ValueError(
                "catalyst_retention of each event must give the physical retention of "
                "every catalyst."
            )
        for value in retention.values():
            _number(value, "catalyst retention", fraction=True)
        if "catalyst_activity_retention" in event:
            activity_retention = event["catalyst_activity_retention"]
            if (
                not isinstance(activity_retention, dict)
                or set(activity_retention) != catalysts
            ):
                raise ValueError(
                    "catalyst_activity_retention of an event must give the activity "
                    "retention of every catalyst."
                )
            for value in activity_retention.values():
                _number(value, "catalyst activity retention", fraction=True)
        additions = event["catalyst_additions"]
        if not isinstance(additions, dict) or additions.keys() - catalysts:
            raise ValueError(
                "catalyst_additions must map declared catalysts to amounts in the "
                "amount unit of the catalyst."
            )
        for value in additions.values():
            _fields(value, {"amount", "active_fraction"})
            _number(value["amount"], "added catalyst amount")
            _number(value["active_fraction"], "added active_fraction", fraction=True)


def simulate(request, *, method="Radau"):
    """Integrate chemical amounts per initial volume and absolute catalyst amounts.

    Enzyme-linked capacity is M/s at the declared reference amount and initial
    volume. Rate factors use actual concentration/scale ratios and explicit
    inhibition terms. Each
    forward/reverse channel retains its own source and integrated gross extent.
    Observation rows at exchanges describe the post-event state.
    """
    validate_request(request)
    if method not in {"Radau", "DOP853"}:
        raise ValueError(f"The solver must be Radau or DOP853; found {method!r}.")
    stated, applied = request, None
    request, applied = apply_conditions(request)
    import numpy as np
    import scipy
    from scipy.integrate import solve_ivp
    from scipy.optimize import approx_fprime

    species = [s["id"] for s in request["species"]]
    catalysts = [c["id"] for c in request["catalysts"]]
    channels = request["channels"]
    si, ci = (
        {v: i for i, v in enumerate(species)},
        {v: i for i, v in enumerate(catalysts)},
    )
    ns, nc, nr = len(species), len(catalysts), len(channels)
    total, active = slice(ns, ns + nc), slice(ns + nc, ns + 2 * nc)
    physical_loss = slice(ns + 2 * nc, ns + 3 * nc)
    active_loss = slice(ns + 3 * nc, ns + 4 * nc)
    deactivated = slice(ns + 4 * nc, ns + 5 * nc)
    catalyst_feed = slice(ns + 5 * nc, ns + 6 * nc)
    active_feed = slice(ns + 6 * nc, ns + 7 * nc)
    ext = slice(ns + 7 * nc, ns + 7 * nc + nr)
    inlet = slice(ext.stop, ext.stop + ns)
    outlet = slice(inlet.stop, inlet.stop + ns)
    collections = request.get("collections", [])
    sample = slice(outlet.stop, outlet.stop + (ns if collections else 0))
    state = np.zeros(sample.stop)
    initial = np.array([s["initial_M"] for s in request["species"]], float)
    state[:ns] = initial
    state[total] = [c["amount"] for c in request["catalysts"]]
    state[active] = [c["amount"] * c["active_fraction"] for c in request["catalysts"]]
    initial_total, initial_active = state[total].copy(), state[active].copy()
    matrix = np.zeros((ns, nr))
    for j, channel in enumerate(channels):
        for name, value in channel["stoichiometry"].items():
            matrix[si[name], j] = float(signed_coefficient(value))
    volume = request["volume_L"]
    schedule = flow_schedule(request)
    variable = any("inlet_L_s" in p for p in request["phases"])

    def liquid_volume(t):
        phase = next(p for p in schedule if p["end_s"] >= t)
        return phase["start_volume_L"] + (phase["inlet_L_s"] - phase["outlet_L_s"]) * (
            t - phase["start_s"]
        )

    atol = np.full(len(state), 1e-12)
    if variable:
        # Amounts are normalized to the initial volume. Scaling to the smallest
        # liquid volume bounds the concentration absolute tolerance throughout.
        minimum_volume = min(volume, *(p["end_volume_L"] for p in schedule))
        chemical_atol = 1e-12 * minimum_volume / volume
        _number(chemical_atol, "resolvable chemical amount tolerance", positive=True)
        for part in (slice(0, ns), ext, inlet, outlet):
            atol[part] = chemical_atol
    catalyst_scale = np.array([c["reference_amount"] for c in request["catalysts"]])
    for part in (
        total,
        active,
        physical_loss,
        active_loss,
        deactivated,
        catalyst_feed,
        active_feed,
    ):
        atol[part] = 1e-12 * catalyst_scale
    diagnostics = []

    def checked_snapshot(t, y):
        if not np.isfinite(y).all() or np.any(y < -100 * atol):
            raise ValueError(
                "Accepted operating states require finite amounts that are nonnegative "
                "within the numerical tolerance."
            )
        if np.any(y[active] - y[total] > 100 * atol[total]):
            raise ValueError(
                "The active catalyst amount exceeds the retained physical amount."
            )
        ratio = liquid_volume(t) / volume if variable else 1.0
        expected = (initial + matrix @ y[ext] + y[inlet] - y[outlet]) / ratio
        residual = y[:ns] / ratio - expected
        if np.max(abs(residual)) > 1e-9 * max(1.0, np.max(abs(expected))):
            raise ValueError(
                "The integrated species and external-flow ledger does not close."
            )
        amount_residual = y[total] + y[physical_loss] - initial_total - y[catalyst_feed]
        activity_residual = (
            y[active]
            + y[active_loss]
            + y[deactivated]
            - initial_active
            - y[active_feed]
        )
        if np.any(abs(amount_residual) > 1e-9 * catalyst_scale) or np.any(
            abs(activity_residual) > 1e-9 * catalyst_scale
        ):
            raise ValueError(
                "The integrated catalyst amount and activity ledgers do not close."
            )
        return {
            "time_s": float(t),
            **(
                {
                    "volume_L": float(liquid_volume(t)),
                    "amounts_mol": dict(
                        zip(species, map(float, y[:ns] * volume), strict=True)
                    ),
                }
                if variable
                else {}
            ),
            "concentrations_M": dict(
                zip(species, map(float, y[:ns] / ratio), strict=True)
            ),
            "catalysts": {
                name: {
                    "amount": float(y[total][i]),
                    "active_amount": float(y[active][i]),
                    "amount_unit": request["catalysts"][i]["amount_unit"],
                    "removed_amount": float(y[physical_loss][i]),
                    "removed_active_amount": float(y[active_loss][i]),
                    "deactivated_amount": float(y[deactivated][i]),
                    "added_amount": float(y[catalyst_feed][i]),
                    "added_active_amount": float(y[active_feed][i]),
                    "amount_balance_residual": float(amount_residual[i]),
                    "activity_balance_residual": float(activity_residual[i]),
                }
                for i, name in enumerate(catalysts)
            },
            "reaction_extent_mol": {
                r["id"]: float(v * volume)
                for r, v in zip(channels, y[ext], strict=True)
            },
            "feed_mol": dict(zip(species, map(float, y[inlet] * volume), strict=True)),
            "removed_mol": dict(
                zip(species, map(float, y[outlet] * volume), strict=True)
            ),
            "balance_residual_M": dict(zip(species, map(float, residual), strict=True)),
            "maximum_balance_error_M": float(np.max(abs(residual))),
        }

    def right_hand_side(phase, flows, collection_duration=None):
        enabled = set(phase["active_channels"])
        feed = np.array([phase["feed_M"].get(s, 0.0) for s in species])
        inlet_dilution = (
            flows["inlet_L_s"] / volume if variable else phase["dilution_per_s"]
        )
        direct_input = (
            np.array([phase.get("solute_input_mol_s", {}).get(s, 0.0) for s in species])
            / volume
        )
        has_direct_input = np.any(direct_input)
        supplied = phase.get("catalyst_feed", {})
        catalyst_influx = np.array(
            [supplied.get(c, {}).get("amount_per_L", 0.0) for c in catalysts]
        ) * (flows["inlet_L_s"] if variable else phase["dilution_per_s"] * volume)
        active_influx = catalyst_influx * np.array(
            [supplied.get(c, {}).get("active_fraction", 0.0) for c in catalysts]
        )

        def rhs(t, y):
            # Trial stages use the continuous nonnegative boundary extension;
            # accepted states retain their actual solver values and diagnostics.
            volume_ratio = liquid_volume(t) / volume if variable else 1.0
            concentrations = np.maximum(y[:ns], 0.0) / volume_ratio
            outlet_dilution = (
                flows["outlet_L_s"] / (volume * volume_ratio)
                if variable
                else phase["dilution_per_s"]
            )
            activity = np.maximum(y[active], 0.0)
            flux = np.zeros(nr)
            for j, r in enumerate(channels):
                if r["id"] not in enabled:
                    continue
                value = r["capacity_M_s"]
                if "catalyst" in r:
                    k = ci[r["catalyst"]]
                    value *= activity[k] / catalyst_scale[k] / volume_ratio
                for f in r["factors"]:
                    c = concentrations[si[f["species"]]]
                    ratio = c / f["scale_M"]
                    term = ratio ** f.get("power", 1)
                    if f["kind"] == "saturation":
                        denominator = (
                            1
                            + ratio
                            + sum(
                                concentrations[si[k]] / v
                                for k, v in f.get(
                                    "competitive_inhibitors_M", {}
                                ).items()
                            )
                        )
                        if "self_inhibition_M" in f:
                            denominator += ratio * c / f["self_inhibition_M"]
                        denominator += ratio * sum(
                            concentrations[si[k]] / v
                            for k, v in f.get("uncompetitive_inhibitors_M", {}).items()
                        )
                        term /= denominator ** f.get("denominator_power", 1)
                    value *= term
                if "denominator" in r:
                    denominator = r["denominator"]["constant"]
                    for entry in r["denominator"]["terms"]:
                        term = entry["coefficient"]
                        for part in entry["factors"]:
                            term *= (
                                concentrations[si[part["species"]]] / part["scale_M"]
                            ) ** part.get("power", 1)
                        denominator += term
                    value /= denominator
                value /= 1 + sum(
                    concentrations[si[k]] / v
                    for k, v in r.get("inhibitors_M", {}).items()
                )
                flux[j] = value * volume_ratio
            dy = np.zeros_like(y)
            dy[ext] = flux
            dy[inlet] = inlet_dilution * feed
            if has_direct_input:
                dy[inlet] += direct_input
            # Exact zero terms stay zero during adaptive Jacobian probes of
            # constant states, including probes that overflow an unused column.
            if outlet_dilution:
                dy[outlet] = outlet_dilution * concentrations * volume_ratio
            if collection_duration is not None:
                # Segment-local mean keeps short late fractions numerically resolved.
                dy[sample] = concentrations / collection_duration
            dy[:ns] = matrix @ flux + dy[inlet] - dy[outlet]
            dy[catalyst_feed] = catalyst_influx
            dy[active_feed] = active_influx
            for i, c in enumerate(request["catalysts"]):
                wash = outlet_dilution * c["washout_fraction"]
                if wash:
                    dy[physical_loss][i] = wash * max(y[total][i], 0.0)
                    dy[active_loss][i] = wash * activity[i]
                if c["deactivation_per_s"]:
                    dy[deactivated][i] = c["deactivation_per_s"] * activity[i]
                dy[total][i] = catalyst_influx[i] - dy[physical_loss][i]
                dy[active][i] = (
                    active_influx[i] - dy[active_loss][i] - dy[deactivated][i]
                )
            return dy

        return rhs

    observations = {0.0: checked_snapshot(0.0, state)}
    event_records = []
    collection_segments = []
    event_map = {e["time_s"]: e for e in request["events"]}
    boundaries = sorted(
        {
            0.0,
            *(p["end_s"] for p in request["phases"]),
            *event_map,
            *(c[k] for c in collections for k in ("start_s", "end_s")),
        }
    )
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        phase = next(p for p in request["phases"] if p["end_s"] >= right)
        flows = next(p for p in schedule if p["end_s"] >= right)
        collecting = flows["outlet_L_s"] > 0 and any(
            c["start_s"] <= left and right <= c["end_s"] for c in collections
        )
        state[sample] = 0
        sample_times = sorted(
            {t for t in request["times_s"] if left < t <= right} | {right}
        )
        rhs = right_hand_side(phase, flows, right - left if collecting else None)

        def jacobian(t, y, function=rhs):
            # Recompute a scale-aware forward step at each state. Flat columns
            # keep bounded probes instead of accumulating adaptive step growth.
            step = np.sqrt(np.finfo(float).eps) * np.maximum(np.abs(y), atol)
            return approx_fprime(y, lambda value: function(t, value), epsilon=step)

        sol = solve_ivp(
            rhs,
            (left, right),
            state,
            method=method,
            t_eval=sample_times,
            rtol=1e-9,
            atol=atol,
            **({"jac": jacobian} if method == "Radau" else {}),
        )
        if not sol.success or sol.t[-1] != right:
            raise ValueError(f"The operating integration failed: {sol.message}")
        diagnostics.append(
            dict(
                start_s=left,
                end_s=right,
                nfev=sol.nfev,
                njev=sol.njev,
                nlu=sol.nlu,
                message=sol.message,
            )
        )
        for t, y in zip(sol.t, sol.y.T, strict=True):
            observed = checked_snapshot(t, y)
            if t in request["times_s"]:
                observations[t] = observed
        state = sol.y[:, -1].copy()
        if collecting:
            collection_segments.append(
                (
                    left,
                    right,
                    flows["outlet_L_s"] * (right - left),
                    state[sample].copy(),
                )
            )
        if right in event_map:
            e = event_map[right]
            before = checked_snapshot(right, state)
            fraction = e["exchange_fraction"]
            removed = fraction * state[:ns].copy()
            added = fraction * np.array([e["feed_M"].get(s, 0.0) for s in species])
            if variable:
                added *= liquid_volume(right) / volume
            state[:ns] += added - removed
            state[inlet] += added
            state[outlet] += removed
            for name, i in ci.items():
                retention = e["catalyst_retention"][name]
                activity_retention = e.get("catalyst_activity_retention", {}).get(
                    name, 1.0
                )
                addition = e["catalyst_additions"].get(
                    name, {"amount": 0.0, "active_fraction": 0.0}
                )
                state[physical_loss][i] += state[total][i] * (1 - retention)
                state[active_loss][i] += state[active][i] * (1 - retention)
                retained_activity = state[active][i] * retention
                state[deactivated][i] += retained_activity * (1 - activity_retention)
                state[catalyst_feed][i] += addition["amount"]
                state[active_feed][i] += (
                    addition["amount"] * addition["active_fraction"]
                )
                state[total][i] = state[total][i] * retention + addition["amount"]
                state[active][i] = (
                    retained_activity * activity_retention
                    + addition["amount"] * addition["active_fraction"]
                )
            after = checked_snapshot(right, state)
            event_records.append(
                dict(request=copy.deepcopy(e), before=before, after=after)
            )
            if right in request["times_s"]:
                observations[right] = after
    result = dict(
        schema_version="0.1",
        profile=PROFILE,
        request=copy.deepcopy(stated),
        request_sha256=digest(stated),
        conditions=applied,
        trajectory=[observations[t] for t in request["times_s"]],
        events=event_records,
        numerical=dict(
            method=method,
            jacobian=(
                "state-scaled-forward-difference-v1"
                if method == "Radau"
                else "explicit-method"
            ),
            rtol=1e-9,
            concentration_atol_M=1e-12,
            catalyst_relative_reference_atol=1e-12,
            segments=diagnostics,
            versions=dict(numpy=np.__version__, scipy=scipy.__version__),
        ),
    )
    if "collections" in request:
        result["collections"] = []
        for collection in collections:
            parts = [
                (v, mean)
                for a, b, v, mean in collection_segments
                if collection["start_s"] <= a and b <= collection["end_s"]
            ]
            collected_volume = collection_volume(request, collection)
            amounts = {
                name: math.fsum(v * mean[i] for v, mean in parts)
                for i, name in enumerate(species)
            }
            result["collections"].append(
                {
                    **collection,
                    "volume_L": collected_volume,
                    "amounts_mol": amounts,
                    "mean_concentrations_M": {
                        name: amount / collected_volume
                        for name, amount in amounts.items()
                    },
                }
            )
    return result


def _operation_input(value):
    """Accept an operating definition, or supplied JSON text holding one."""
    if isinstance(value, dict) and "text" in value:
        return json.loads(source_text(value))
    return value
