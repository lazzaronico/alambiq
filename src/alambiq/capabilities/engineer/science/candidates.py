"""Measured protein properties, train-only regression and explicit evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import warnings

from alambiq.capabilities.characterize.science.embeddings import validate_embeddings
from alambiq.capabilities.measure.science.fitting import checked_assay
from alambiq.capabilities.shared.records import digest as content_digest

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MODEL_PARAMETERS = {
    "max_iter": 500,
    "tol": 1e-6,
    "alpha_1": 1e-6,
    "alpha_2": 1e-6,
    "lambda_1": 1e-6,
    "lambda_2": 1e-6,
    "fit_intercept": True,
    "compute_score": False,
    "copy_X": True,
    "verbose": False,
}


def candidate_panel(panel):
    if not isinstance(panel, dict) or panel.get("schema_version") != "0.1":
        raise ValueError(
            "The candidate panel must be a JSON object with schema_version 0.1."
        )
    candidates = panel.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 500:
        raise ValueError(
            "The candidate panel must contain a list candidates of 1 to 500 protein "
            "candidates."
        )
    ids, groups, sequences = set(), {}, {}
    for c in candidates:
        if not isinstance(c, dict) or set(c) != {
            "sample_id",
            "sequence",
            "sequence_sha256",
            "group_id",
            "partition",
        }:
            raise ValueError(
                "Each candidate must have exactly the fields sample_id, sequence, "
                "sequence_sha256, group_id and partition."
            )
        for key in ("sample_id", "group_id"):
            if (
                not isinstance(c[key], str)
                or not c[key]
                or len(c[key]) > 200
                or any(ch.isspace() for ch in c[key])
            ):
                raise ValueError(
                    f"Each candidate {key} must be a nonempty string of at most 200 "
                    f"characters without whitespace; found {c[key]!r}."
                )
        if c["sample_id"] in ids:
            raise ValueError(
                f"The sample_id {c['sample_id']} occurs more than once; each sample_id "
                "must be unique."
            )
        ids.add(c["sample_id"])
        if c["partition"] not in {"train", "evaluation", "pool"}:
            raise ValueError(
                f"Candidate {c['sample_id']} has partition {c['partition']!r}; the "
                "partition must be train, evaluation or pool."
            )
        sequence = c["sequence"]
        if not isinstance(sequence, str) or not re.fullmatch(
            r"[ACDEFGHIKLMNPQRSTVWY]{1,10000}", sequence
        ):
            raise ValueError(
                f"The sequence of candidate {c['sample_id']} must consist of 1 to 10,000 "
                "of the 20 standard amino acids, without gaps."
            )
        if hashlib.sha256(sequence.encode()).hexdigest() != c["sequence_sha256"]:
            raise ValueError(
                f"The sequence_sha256 of candidate {c['sample_id']} differs from the "
                "SHA-256 digest of its sequence."
            )
        groups.setdefault(c["group_id"], set()).add(c["partition"])
        sequences.setdefault(c["sequence_sha256"], set()).add(c["partition"])
    if any("evaluation" in parts and len(parts) > 1 for parts in groups.values()):
        raise ValueError(
            "A group of the evaluation partition also appears in another partition. "
            "A group of the evaluation partition must not appear in the training or "
            "pool partition."
        )
    if any("evaluation" in parts and len(parts) > 1 for parts in sequences.values()):
        raise ValueError(
            "An identical sequence appears in the evaluation partition and in another "
            "partition. A sequence of the evaluation partition must not appear in the "
            "training or pool partition."
        )
    return candidates


def feature_matrix(candidates, kind, embeddings=None):
    if kind == "sequence-composition-v1":
        if embeddings is not None:
            raise ValueError(
                "The features sequence-composition-v1 take no embeddings. Remove "
                "embeddings, or set features to pooled-embeddings."
            )
        names = ["length"] + [f"fraction_{aa}" for aa in AMINO_ACIDS]
        values = [
            [len(c["sequence"])]
            + [c["sequence"].count(aa) / len(c["sequence"]) for aa in AMINO_ACIDS]
            for c in candidates
        ]
        return (
            names,
            values,
            {
                "kind": kind,
                "basis": "Length of the supplied sequence and its 20 amino-acid fractions",
            },
        )
    if kind != "pooled-embeddings":
        raise ValueError(
            "features must be sequence-composition-v1 or pooled-embeddings; "
            f"found {kind!r}."
        )
    proteins = validate_embeddings(embeddings)
    by_id = {p["id"]: p for p in proteins}
    if set(by_id) != {c["sample_id"] for c in candidates}:
        raise ValueError(
            "The identifiers of the embedding record differ from the sample_id values of "
            "the candidate panel. The embedding record must contain every candidate and "
            "no other protein."
        )
    if any(by_id[c["sample_id"]]["sequence"] != c["sequence"] for c in candidates):
        raise ValueError(
            "The sequence of an embedding differs from the sequence of the candidate "
            "with the same identifier."
        )
    meta = {
        k: embeddings[k]
        for k in ("model", "revision", "layer", "pooling", "input_tracks", "dimension")
    }
    return (
        [f"embedding_{i}" for i in range(embeddings["dimension"])],
        [by_id[c["sample_id"]]["pooled_vector"] for c in candidates],
        {"kind": kind, **meta, "record_sha256": content_digest(embeddings)},
    )


def validate_parameters(parameters, has_embeddings):
    required = {"metric", "condition_id", "unit", "observation_basis", "features"}
    if not isinstance(parameters, dict) or set(parameters) != required:
        raise ValueError(
            "The model parameters must be exactly metric, condition_id, unit, "
            "observation_basis and features."
        )
    if any(
        not isinstance(v, str) or not v.strip() or len(v) > 200
        for v in parameters.values()
    ):
        raise ValueError(
            "metric, condition_id, unit, observation_basis and features must each be a "
            "nonempty string of at most 200 characters."
        )
    if parameters["observation_basis"] not in {"supplied-values", "independent-means"}:
        raise ValueError(
            "observation_basis must be supplied-values or independent-means; found "
            f"{parameters['observation_basis']!r}."
        )
    if parameters["features"] not in {"sequence-composition-v1", "pooled-embeddings"}:
        raise ValueError(
            "features must be sequence-composition-v1 or pooled-embeddings; found "
            f"{parameters['features']!r}."
        )
    if (parameters["features"] == "pooled-embeddings") != has_embeddings:
        raise ValueError(
            "embeddings is required with the features pooled-embeddings and accepted "
            "only with them."
        )


def modeling_data(panel, assay, parameters, embeddings=None):
    validate_parameters(parameters, embeddings is not None)
    candidates = candidate_panel(panel)
    data = checked_assay(assay)
    if parameters["condition_id"] not in data["conditions"]:
        raise ValueError(
            f"The assay record has no condition {parameters['condition_id']}."
        )
    names, matrix, representation = feature_matrix(
        candidates, parameters["features"], embeddings
    )
    by_id = {c["sample_id"]: c for c in candidates}
    for link in assay.get("sample_links", {}).get("samples", []):
        if (
            link["sample_id"] in by_id
            and link["sequence"] != by_id[link["sample_id"]]["sequence"]
        ):
            raise ValueError(
                f"The sequence of sample {link['sample_id']} in the assay record differs "
                "from the sequence of the candidate with the same sample_id."
            )
    rows = [
        r
        for r in data["rows"]
        if r["metric"] == parameters["metric"]
        and r["condition_id"] == parameters["condition_id"]
    ]
    if not rows:
        raise ValueError(
            f"The assay record has no measurements of {parameters['metric']} under "
            f"condition {parameters['condition_id']}."
        )
    if any(r["sample_id"] not in by_id for r in rows):
        raise ValueError(
            "The selected measurements include sample_id values that are not in the "
            "candidate panel."
        )
    if any(r["unit"] != parameters["unit"] for r in rows):
        raise ValueError(
            f"unit {parameters['unit']} differs from the unit of a selected "
            f"measurement; every selected measurement must have unit {parameters['unit']}."
        )
    if any(
        by_id[r["sample_id"]]["partition"] == "pool" and r["value"] is not None
        for r in rows
    ):
        raise ValueError(
            "The assay record contains a measured value for a candidate of the pool "
            "partition, which must be unmeasured. Set the partition of candidates with "
            "returned outcomes to train."
        )
    values, excluded = {}, []
    for c in candidates:
        records = [r for r in rows if r["sample_id"] == c["sample_id"]]
        if c["partition"] == "pool":
            continue
        measured = [r for r in records if r["value"] is not None]
        excluded.extend(
            {
                "sample_id": c["sample_id"],
                "measurement_id": r["measurement_id"],
                "reason": r["status"],
            }
            for r in records
            if r["value"] is None
        )
        if not measured:
            excluded.append(
                {
                    "sample_id": c["sample_id"],
                    "measurement_id": None,
                    "reason": "no-measured-response",
                }
            )
            continue
        if parameters["observation_basis"] == "supplied-values":
            if len(measured) != 1 or measured[0]["independent_id"]:
                raise ValueError(
                    f"observation_basis supplied-values requires exactly 1 measured "
                    f"record with no independent unit assigned for each candidate; "
                    f"candidate {c['sample_id']} has {len(measured)}. Use "
                    "independent-means for candidates with replicas."
                )
            value = measured[0]["value"]
        else:
            group = next(
                g
                for g in data["groups"]
                if g["sample_id"] == c["sample_id"]
                and g["condition_id"] == parameters["condition_id"]
                and g["metric"] == parameters["metric"]
            )
            if group["mean"] is None:
                raise ValueError(
                    "observation_basis independent-means requires an independent unit "
                    f"assigned to every measured record; candidate {c['sample_id']} has "
                    "none."
                )
            value = group["mean"]
        values[c["sample_id"]] = {
            "value": value,
            "measurement_ids": [r["measurement_id"] for r in measured],
        }
    train = [
        i
        for i, c in enumerate(candidates)
        if c["partition"] == "train" and c["sample_id"] in values
    ]
    evaluation = [
        i
        for i, c in enumerate(candidates)
        if c["partition"] == "evaluation" and c["sample_id"] in values
    ]
    if len(train) < 8 or len({candidates[i]["group_id"] for i in train}) < 3:
        raise ValueError(
            "The training partition needs at least 8 measured candidates in at least 3 "
            f"groups; found {len(train)} candidates in "
            f"{len({candidates[i]['group_id'] for i in train})} groups."
        )
    if len(evaluation) < 4 or len({candidates[i]["group_id"] for i in evaluation}) < 2:
        raise ValueError(
            "The evaluation partition needs at least 4 measured candidates in at least "
            f"2 groups; found {len(evaluation)} candidates in "
            f"{len({candidates[i]['group_id'] for i in evaluation})} groups."
        )
    return (
        candidates,
        data,
        names,
        matrix,
        representation,
        values,
        excluded,
        train,
        evaluation,
    )


def fit_candidates(panel, assay, parameters, *, embeddings=None):
    (
        candidates,
        data,
        names,
        matrix,
        representation,
        values,
        excluded,
        train,
        evaluation,
    ) = modeling_data(panel, assay, parameters, embeddings)
    try:
        import numpy as np
        import sklearn
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ValueError(
            "Install alambiq[quantitative] for candidate modeling"
        ) from exc
    x = np.array(matrix, dtype=float)
    y = np.array([values[candidates[i]["sample_id"]]["value"] for i in train])
    if not np.isfinite(x).all():
        raise ValueError("The candidate features include a value that is not finite.")
    scaler = StandardScaler().fit(x[train])
    scaled = scaler.transform(x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = BayesianRidge(**MODEL_PARAMETERS).fit(scaled[train], y)
        predicted, std = model.predict(scaled, return_std=True)
    baseline = float(np.mean(y))
    diagnostics = [str(w.message) for w in caught]
    same_features = {}
    for candidate, row in zip(candidates, matrix, strict=True):
        same_features.setdefault(tuple(row), []).append(candidate["sample_id"])
    aliases = [ids for ids in same_features.values() if len(ids) > 1]
    representation["indistinguishable_candidates"] = aliases
    if aliases:
        diagnostics.append(
            f"{sum(map(len, aliases))} candidates share identical input features. The "
            f"number of sets of identical features is {len(aliases)}."
        )
    if model.n_iter_ >= MODEL_PARAMETERS["max_iter"]:
        raise ValueError(
            "The Bayesian ridge fit reached its iteration limit of "
            f"{MODEL_PARAMETERS['max_iter']} iterations and is rejected."
        )
    if not np.isfinite(predicted).all() or not np.isfinite(std).all():
        raise ValueError("The fitted model produced predictions that are not finite.")

    def metrics(observed, fitted):
        error = observed - fitted
        total = float(np.sum((observed - np.mean(observed)) ** 2))
        return {
            "n": len(observed),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "r_squared": 1 - float(np.sum(error**2)) / total if total > 0 else None,
        }

    observed = np.array(
        [values[candidates[i]["sample_id"]]["value"] for i in evaluation]
    )
    score = metrics(observed, predicted[evaluation])
    score.update(
        {
            "interval_95_coverage": float(
                np.mean(
                    np.abs(observed - predicted[evaluation]) <= 1.96 * std[evaluation]
                )
            ),
            "mean_interval_95_width": float(np.mean(3.92 * std[evaluation])),
        }
    )
    baseline_score = metrics(observed, np.repeat(baseline, len(evaluation)))
    if score["rmse"] >= baseline_score["rmse"]:
        diagnostics.append(
            "The evaluation RMSE of the model is not lower than the evaluation RMSE of "
            "the training-mean baseline."
        )
    if score["interval_95_coverage"] < 0.95:
        diagnostics.append(
            "The 95 % predictive intervals contain less than 95 % of the evaluation "
            "observations."
        )
    if len(evaluation) < 30:
        diagnostics.append(
            "Fewer than 30 evaluation observations."
        )
    if np.ptp(y) == 0:
        diagnostics.append(
            "The training response is constant."
        )
    predictions = []
    for i, c in enumerate(candidates):
        distance = np.sqrt(np.mean((scaled[train] - scaled[i]) ** 2, axis=1))
        predictions.append(
            {
                "sample_id": c["sample_id"],
                "sequence_sha256": c["sequence_sha256"],
                "group_id": c["group_id"],
                "partition": c["partition"],
                "predicted": float(predicted[i]),
                "predictive_standard_deviation": float(std[i]),
                "model_interval_95": [
                    float(predicted[i] - 1.96 * std[i]),
                    float(predicted[i] + 1.96 * std[i]),
                ],
                "nearest_training_feature_distance": float(np.min(distance)),
            }
        )
    result = {
        "schema_version": "0.1",
        "request": parameters,
        "source": {
            "panel_sha256": content_digest(panel),
            "panel_receipt": panel.get("source"),
            "assay_source_sha256": data["source_sha256"],
            "assay_receipt": data.get("source"),
            "condition": data["conditions"][parameters["condition_id"]],
            "representation": representation,
        },
        "model": {
            "profile": "bayesian-ridge-v1",
            "library": "scikit-learn",
            "version": sklearn.__version__,
            "numpy_version": np.__version__,
            "settings": MODEL_PARAMETERS,
            "feature_names": names,
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "noise_precision": float(model.alpha_),
            "coefficient_precision": float(model.lambda_),
            "iterations": int(model.n_iter_),
            "baseline_training_mean": baseline,
        },
        "training": [
            {
                "sample_id": candidates[i]["sample_id"],
                "group_id": candidates[i]["group_id"],
                **values[candidates[i]["sample_id"]],
            }
            for i in train
        ],
        "evaluation": {
            "model": score,
            "baseline": baseline_score,
            "groups": len({candidates[i]["group_id"] for i in evaluation}),
            "rows": [
                {
                    **predictions[i],
                    "observed": values[candidates[i]["sample_id"]]["value"],
                    "measurement_ids": values[candidates[i]["sample_id"]][
                        "measurement_ids"
                    ],
                    "baseline": baseline,
                }
                for i in evaluation
            ],
        },
        "predictions": predictions,
        "excluded": excluded,
        "diagnostics": diagnostics,
        "method": "StandardScaler and BayesianRidge are fitted on the training observations only. An overlap of evaluation groups or evaluation sequences with other partitions is rejected, and outcomes of the pool partition are not accepted. The training-mean baseline uses the same evaluation observations. The predictive standard deviation and the normal 95 % intervals come from the fitted linear Gaussian model, including its noise; their coverage is measured on the evaluation partition. Feature distance is the root-mean-square distance after standardization with the training statistics. The stored coefficients and scaling reproduce the means; the uncertainty for a new panel requires fitting again with the recorded inputs.",
    }
    json.dumps(result, allow_nan=False)
    return result
