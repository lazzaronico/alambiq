"""Explicit bounded candidate selection."""

import math
import random
from collections import Counter

from alambiq.capabilities.engineer.science.candidates import (
    candidate_panel,
    content_digest,
    feature_matrix,
)
from alambiq.capabilities.measure.science.tables import finite

STRATEGIES = {
    "random",
    "diverse",
    "highest-predicted",
    "lowest-predicted",
    "uncertain",
    "highest-predicted-diverse",
    "pareto",
}
MODELED = {
    "highest-predicted",
    "lowest-predicted",
    "uncertain",
    "highest-predicted-diverse",
}
DIVERSE = {"diverse", "highest-predicted-diverse"}


def selection_parameters(parameters, inputs):
    required = {"strategy", "count", "seed"}
    optional = {
        "eligible_ids",
        "exclude_ids",
        "max_per_group",
        "costs",
        "budget",
        "cost_unit",
        "diversity",
        "diversity_weight",
        "objectives",
    }
    if (
        not isinstance(parameters, dict)
        or not required <= parameters.keys()
        or not parameters.keys() <= required | optional
    ):
        raise ValueError(
            "The selection parameters must include strategy, count and seed, and may "
            "include only eligible_ids, exclude_ids, max_per_group, costs, budget, "
            "cost_unit, diversity, diversity_weight and objectives in addition."
        )
    if parameters["strategy"] not in STRATEGIES:
        raise ValueError(
            f"strategy {parameters['strategy']!r} is not a selection strategy; the "
            f"strategies are {', '.join(sorted(STRATEGIES))}."
        )
    for key, low, high in [
        ("count", 1, 500),
        ("seed", 0, 2147483647),
        ("max_per_group", 1, 500),
    ]:
        value = parameters.get(key, 500)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(
                f"{key} must be an integer from {low} to {high}; found {value!r}."
            )
    strategy = parameters["strategy"]
    if (strategy in MODELED) != ("model" in inputs):
        raise ValueError(
            f"strategy {strategy} {'requires' if strategy in MODELED else 'does not take'} "
            "model; model is accepted only with highest-predicted, lowest-predicted, "
            "uncertain and highest-predicted-diverse."
        )
    if (strategy == "pareto") != ("models" in inputs):
        raise ValueError(
            f"strategy {strategy} {'does not take' if strategy != 'pareto' else 'requires'} "
            "models; models and objectives are accepted only with pareto."
        )
    if strategy == "pareto":
        objectives = parameters.get("objectives")
        if (
            not isinstance(objectives, list)
            or not 2 <= len(objectives) <= 5
            or any(o not in {"maximize", "minimize"} for o in objectives)
        ):
            raise ValueError(
                "objectives must be a list of 2 to 5 objectives, each maximize or minimize."
            )
    elif "objectives" in parameters:
        raise ValueError("objectives is accepted only with the strategy pareto.")
    if strategy in DIVERSE:
        if parameters.get("diversity") not in {"sequence", "embedding-cosine"}:
            raise ValueError(
                f"strategy {strategy} requires diversity, which must be sequence or "
                f"embedding-cosine; found {parameters.get('diversity')!r}."
            )
        if (parameters["diversity"] == "embedding-cosine") != ("embeddings" in inputs):
            raise ValueError(
                "embeddings is required with diversity embedding-cosine and accepted "
                "only with it."
            )
    elif "diversity" in parameters or "embeddings" in inputs:
        raise ValueError(
            "diversity and embeddings are accepted only with the strategies diverse "
            "and highest-predicted-diverse."
        )
    if "diversity_weight" in parameters:
        weight = parameters["diversity_weight"]
        if strategy != "highest-predicted-diverse":
            raise ValueError(
                "diversity_weight is accepted only with the strategy "
                "highest-predicted-diverse."
            )
        if type(weight) not in (int, float) or not 0 <= weight <= 10:
            raise ValueError(
                f"diversity_weight must be a number from 0 to 10; found {weight!r}."
            )


def select_candidates(panel, parameters, *, model=None, models=None, embeddings=None):
    candidates = candidate_panel(panel)
    selection_parameters(
        parameters,
        {"panel"}
        | ({"model"} if model is not None else set())
        | ({"models"} if models is not None else set())
        | ({"embeddings"} if embeddings is not None else set()),
    )
    by_id = {c["sample_id"]: c for c in candidates}
    for key in ("eligible_ids", "exclude_ids"):
        ids = parameters.get(key, [])
        if (
            not isinstance(ids, list)
            or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)
            or set(ids) - by_id.keys()
        ):
            raise ValueError(
                f"{key} must be a list of distinct sample_id values of the candidate "
                "panel."
            )
    allowed = set(parameters.get("eligible_ids", list(by_id))) - set(
        parameters.get("exclude_ids", [])
    )
    eligible = [
        c for c in candidates if c["partition"] == "pool" and c["sample_id"] in allowed
    ]
    cost_keys = {"costs", "budget", "cost_unit"} & parameters.keys()
    costs, budget = {}, None
    if cost_keys:
        if (
            len(cost_keys) != 3
            or not isinstance(parameters["costs"], dict)
            or not isinstance(parameters["cost_unit"], str)
            or not parameters["cost_unit"].strip()
        ):
            raise ValueError(
                "costs, budget and cost_unit must be set together; costs is a mapping "
                "and cost_unit is a nonblank text."
            )
        costs = {
            key: finite(value, "candidate cost")
            for key, value in parameters["costs"].items()
        }
        budget = finite(parameters["budget"], "budget")
        if (
            budget <= 0
            or any(v <= 0 for v in costs.values())
            or set(costs) - by_id.keys()
            or {c["sample_id"] for c in eligible} - costs.keys()
        ):
            raise ValueError(
                "budget and every cost must be positive, costs must have an entry for "
                "every eligible pool candidate, and every key of costs must be a "
                "sample_id of the candidate panel."
            )
    strategy = parameters["strategy"]
    predicted = {}

    def load_predictions(record):
        found = {}
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != "0.1"
            or record.get("source", {}).get("panel_sha256") != content_digest(panel)
        ):
            raise ValueError(
                "The model record is not a candidate.fit record of this candidate panel "
                "(schema_version 0.1 and the same panel digest)."
            )
        if record.get("evaluation", {}).get("model", {}).get("n", 0) < 4:
            raise ValueError(
                "The model record has fewer than 4 evaluation observations; selection "
                "from predictions requires at least 4."
            )
        for p in record.get("predictions", []):
            identifier = p.get("sample_id")
            if (
                identifier not in by_id
                or identifier in found
                or any(
                    p.get(k) != by_id[identifier][k]
                    for k in ("sequence_sha256", "partition", "group_id")
                )
            ):
                raise ValueError(
                    "A prediction of the model record has a sample_id, "
                    "sequence_sha256, partition or group_id that differs from the "
                    "candidate panel; refit the model on this panel."
                )
            finite(p.get("predicted"), "predicted response")
            if finite(p.get("predictive_standard_deviation"), "predictive SD") < 0:
                raise ValueError(
                    "The predictive standard deviation of a prediction is negative; it "
                    "must be at least 0."
                )
            found[identifier] = p
        if set(found) != set(by_id):
            raise ValueError(
                "The predictions of the model record must cover every candidate of the "
                "candidate panel."
            )
        return found

    objective_sets = []
    if strategy in MODELED:
        predicted = load_predictions(model)
    elif strategy == "pareto":
        if len(models) != len(parameters["objectives"]):
            raise ValueError(
                f"objectives lists {len(parameters['objectives'])} entries for "
                f"{len(models)} models; the list must have one objective for each model."
            )
        objective_sets = [load_predictions(m) for m in models]
    elif model is not None:
        raise ValueError(f"strategy {strategy} does not take model.")
    distance = None
    if strategy in DIVERSE:
        space = parameters.get("diversity")
        if space == "sequence":
            if (
                embeddings is not None
                or len({len(c["sequence"]) for c in eligible}) > 1
            ):
                raise ValueError(
                    "diversity sequence requires eligible candidates with sequences of "
                    "equal length and no embeddings."
                )

            def distance(a, b):
                left, right = by_id[a]["sequence"], by_id[b]["sequence"]
                return sum(x != y for x, y in zip(left, right, strict=True)) / len(left)
        elif space == "embedding-cosine":
            from sklearn.metrics.pairwise import cosine_distances

            _, matrix, _ = feature_matrix(candidates, "pooled-embeddings", embeddings)
            # The representation validator checks hypot (overflow-safe) norms.
            # Normalize before sklearn squares values when calculating distance.
            unit = [[v / math.hypot(*row) for v in row] for row in matrix]
            distances = cosine_distances(unit)
            indexes = {c["sample_id"]: i for i, c in enumerate(candidates)}

            def distance(a, b):
                return float(distances[indexes[a], indexes[b]])
        else:
            raise ValueError(
                f"diversity must be sequence or embedding-cosine; found {space!r}."
            )
    elif "diversity" in parameters or embeddings is not None:
        raise ValueError(
            "diversity and embeddings are accepted only with the strategies diverse "
            "and highest-predicted-diverse."
        )
    shuffled = sorted(c["sample_id"] for c in eligible)
    random.Random(parameters["seed"]).shuffle(shuffled)
    ties = {identifier: i for i, identifier in enumerate(shuffled)}
    selected, remaining, groups = [], set(shuffled), Counter()

    def fits_budget(identifier):
        if budget is None:
            return True
        total = math.fsum(
            [costs[s["sample_id"]] for s in selected] + [costs[identifier]]
        )
        return total <= budget or math.isclose(total, budget, rel_tol=1e-12, abs_tol=0)

    pool_ids = [c["sample_id"] for c in eligible]

    def scaled(values):
        """Min-max scaled to 0..1 over the eligible pool, so scores can be added."""
        found = [values[i] for i in pool_ids]
        low, high = min(found), max(found)
        return {
            i: (values[i] - low) / (high - low) if high > low else 0.5 for i in pool_ids
        }

    if strategy == "highest-predicted-diverse":
        gain = scaled({i: predicted[i]["predicted"] for i in pool_ids})
    fronts: dict[str, int] = {}
    pareto_sum: dict[str, float] = {}
    if strategy == "pareto":
        oriented = []
        for found, direction in zip(
            objective_sets, parameters["objectives"], strict=True
        ):
            sign = 1.0 if direction == "maximize" else -1.0
            oriented.append(scaled({i: sign * found[i]["predicted"] for i in pool_ids}))
        points = {i: tuple(o[i] for o in oriented) for i in pool_ids}
        pareto_sum = {i: math.fsum(p) for i, p in points.items()}
        left = set(pool_ids)
        level = 0
        while left:
            front = {
                i
                for i in left
                if not any(
                    all(points[j][k] >= points[i][k] for k in range(len(points[i])))
                    and any(points[j][k] > points[i][k] for k in range(len(points[i])))
                    for j in left
                    if j != i
                )
            }
            for i in front:
                fronts[i] = level
            left -= front
            level += 1

    def value(identifier):
        if strategy == "random" or strategy == "diverse" and not selected:
            return 0
        if strategy == "diverse":
            return min(distance(identifier, c["sample_id"]) for c in selected)
        if strategy == "highest-predicted-diverse":
            nearest = (
                min(distance(identifier, c["sample_id"]) for c in selected)
                if selected
                else 0.0
            )
            return gain[identifier] + parameters.get("diversity_weight", 1.0) * nearest
        if strategy == "pareto":
            return -1e6 * fronts[identifier] + pareto_sum[identifier]
        p = predicted[identifier]
        return (
            p["predictive_standard_deviation"]
            if strategy == "uncertain"
            else (-p["predicted"] if strategy == "lowest-predicted" else p["predicted"])
        )

    while len(selected) < parameters["count"]:
        possible = [
            i
            for i in remaining
            if groups[by_id[i]["group_id"]] < parameters.get("max_per_group", 500)
            and fits_budget(i)
        ]
        if not possible:
            break
        chosen = min(possible, key=lambda i: (-value(i), ties[i]))
        selected.append(
            {
                **by_id[chosen],
                "rank": len(selected) + 1,
                "selection_value": value(chosen),
                "cost": costs.get(chosen),
            }
        )
        remaining.remove(chosen)
        groups[by_id[chosen]["group_id"]] += 1
    chosen_ids = {c["sample_id"] for c in selected}
    rejected = []
    for c in candidates:
        identifier = c["sample_id"]
        if identifier in chosen_ids:
            continue
        reasons = []
        if c["partition"] != "pool":
            reasons.append(f"partition-{c['partition']}")
        elif identifier not in allowed:
            reasons.append("researcher-eligibility")
        else:
            if groups[c["group_id"]] >= parameters.get("max_per_group", 500):
                reasons.append("group-cap")
            if not fits_budget(identifier):
                reasons.append("remaining-budget")
            if not reasons:
                reasons.append("not-selected-in-requested-batch")
        rejected.append({"sample_id": identifier, "reasons": reasons})
    diagnostics = list(model.get("diagnostics", [])) if model is not None else []
    for extra in models or []:
        diagnostics.extend(extra.get("diagnostics", []))
    if len(selected) < parameters["count"]:
        diagnostics.append(
            f"{len(selected)} of {parameters['count']} requested candidates were chosen "
            "under the supplied eligibility, group caps and budget."
        )
    return {
        "schema_version": "0.1",
        "request": parameters,
        "source": {
            "panel_sha256": content_digest(panel),
            "model_sha256": content_digest(model) if model is not None else None,
            "embeddings_sha256": content_digest(embeddings)
            if embeddings is not None
            else None,
        },
        "fronts": fronts or None,
        "selected": selected,
        "rejected": rejected,
        "total_cost": math.fsum(c["cost"] for c in selected) if costs else None,
        "diagnostics": diagnostics,
        "method": "Greedy selection over pool candidates only, with the strategy and the constraints chosen by the researcher. The seeded order resolves random choices and ties in scores. Diversity starts at a seeded candidate, then maximizes the minimum distance to the chosen candidates; the sequence distance is the fraction of positions that differ. Costs share the supplied unit; budget comparisons allow a relative numerical tolerance of 1e-12. A partial batch is recorded as such.",
    }
