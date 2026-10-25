"""Source-linked enzyme/reaction observations and explicit train-only baselines."""

import copy
import hashlib
import importlib.metadata
import math
import re
from collections import Counter, defaultdict
from fractions import Fraction

from alambiq.capabilities.investigate.science.reactions import (
    chemistry,
    inspect_reaction,
)
from alambiq.capabilities.shared.records import coefficient, digest, label

PROFILE = "enzyme-reaction-kernel-v1"
STATES = {"measured", "inactive", "preparation-failed", "untested", "unreported"}
AA = "ACDEFGHIKLMNPQRSTVWY"


def fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        raise ValueError(f"{name} requires exactly: {expected}")


def keyed(rows, name, maximum):
    if not isinstance(rows, list) or not 1 <= len(rows) <= maximum:
        raise ValueError(f"Supply 1–{maximum} {name} records")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Each {name} record must be an object")
        key = row.get("id")
        if not isinstance(key, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}", key
        ):
            raise ValueError(
                f"Each {name} id needs a short identifier using letters, digits, _, ., : or -"
            )
        if key in result:
            raise ValueError(f"Duplicate {name} id: {key}")
        result[key] = row
    return result


def numeric(value, name, nonnegative=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or abs(value) > 1e12
        or (nonnegative and value < 0)
    ):
        raise ValueError(
            f"{name} requires a finite {'nonnegative ' if nonnegative else ''}number of magnitude at most 1e12"
        )


def novelty_identity(reaction):
    """Compare directed equations independently of a common positive scaling."""
    net = defaultdict(Fraction)
    for p in reaction["participants"]:
        net[p["chemical_id"]] += coefficient(p["coefficient"]) * (
            -1 if p["side"] == "reactant" else 1
        )
    terms = sorted((k, v) for k, v in net.items() if v)
    scale = abs(terms[0][1])
    return digest({k: str(v / scale) for k, v in terms})


def validate_panel(panel):
    fields(
        panel,
        "schema_version id title endpoint sources enzymes reactions conditions observations",
        "Panel",
    )
    if panel["schema_version"] != "0.1":
        raise ValueError("Panel schema_version must be 0.1")
    for key in ("id", "title"):
        label(panel[key], key)
    fields(panel["endpoint"], "metric unit description", "Endpoint")
    for key, value in panel["endpoint"].items():
        label(value, key, 2000)
    sources = keyed(panel["sources"], "sources", 100)
    for source in sources.values():
        if not {"id", "citation", "license"} <= source.keys():
            raise ValueError("Sources require id, citation and license")
        for key in ("citation", "license"):
            label(source[key], key, 2000)
    enzymes = keyed(panel["enzymes"], "enzymes", 500)
    for enzyme in enzymes.values():
        fields(
            enzyme,
            "id sequence sequence_sha256 form source_id source_locator",
            "Enzyme",
        )
        sequence = enzyme["sequence"]
        if not isinstance(sequence, str) or not re.fullmatch(
            r"[ACDEFGHIKLMNPQRSTVWY]{3,10000}", sequence
        ):
            raise ValueError("Each enzyme needs 3–10000 canonical ungapped residues")
        if hashlib.sha256(sequence.encode()).hexdigest() != enzyme["sequence_sha256"]:
            raise ValueError("Enzyme sequence digest mismatch")
        if enzyme["source_id"] not in sources:
            raise ValueError("Unknown enzyme source_id")
        label(enzyme["form"], "form", 2000)
        label(enzyme["source_locator"], "source_locator", 2000)
    if (
        not isinstance(panel["reactions"], list)
        or not 1 <= len(panel["reactions"]) <= 100
    ):
        raise ValueError("Supply 1–100 reaction/feature records")
    reactions = {}
    for item in panel["reactions"]:
        fields(item, "reaction feature_participants", "Reaction feature record")
        reaction = inspect_reaction(item["reaction"])
        key = reaction["request"]["id"]
        if key in reactions:
            raise ValueError("Duplicate reaction ID")
        if not reaction["balance"]["balanced"] or any(
            d["code"] == "zero-net-transformation" for d in reaction["diagnostics"]
        ):
            raise ValueError("Reaction balance must close for a nonzero transformation")
        selected = item["feature_participants"]
        participants = {p["id"]: p for p in reaction["participants"]}
        if (
            not isinstance(selected, list)
            or not selected
            or any(not isinstance(x, str) for x in selected)
        ):
            raise ValueError("Supply feature participant identifiers")
        if len(set(selected)) != len(selected) or set(selected) - participants.keys():
            raise ValueError(
                "Each feature participant must identify a distinct supplied participant"
            )
        if {participants[k]["side"] for k in selected} != {"reactant", "product"}:
            raise ValueError(
                "Select feature participants on both sides of the reaction"
            )
        reaction["feature_participants"] = selected.copy()
        reaction["novelty_identity"] = novelty_identity(reaction)
        reactions[key] = reaction
    conditions = panel["conditions"]
    if not isinstance(conditions, dict) or not 1 <= len(conditions) <= 100:
        raise ValueError("Supply 1–100 identified conditions")
    for key, condition in conditions.items():
        label(key, "condition_id")
        if not isinstance(condition, dict) or "description" not in condition:
            raise ValueError("Each condition requires a description")
        label(condition["description"], "condition description", 4000)
    observations = keyed(panel["observations"], "observations", 10000)
    grouped = defaultdict(list)
    for row in observations.values():
        fields(
            row,
            "id enzyme_id reaction_id condition_id status value reported_sd independent_id technical_id source_id source_locator note",
            "Observation",
        )
        for key, index in (
            ("enzyme_id", enzymes),
            ("reaction_id", reactions),
            ("condition_id", conditions),
            ("source_id", sources),
        ):
            if not isinstance(row[key], str) or row[key] not in index:
                raise ValueError(f"Unknown observation {key}")
        if row["status"] not in STATES:
            raise ValueError(
                "Observation status must be measured, inactive, preparation-failed, untested or unreported"
            )
        if row["status"] == "measured":
            numeric(row["value"], "Measured response")
            if row["reported_sd"] is not None:
                numeric(row["reported_sd"], "reported_sd", True)
        elif row["value"] is not None or row["reported_sd"] is not None:
            raise ValueError(
                "A finite response belongs to a measured observation; other states hold null values"
            )
        for key in ("source_locator", "note"):
            label(row[key], key, 4000)
        for key in ("independent_id", "technical_id"):
            if row[key] is not None:
                label(row[key], key)
        if row["technical_id"] is not None and row["independent_id"] is None:
            raise ValueError("Technical replicates require an independent replicate ID")
        enzyme = enzymes[row["enzyme_id"]]
        key = (
            enzyme["sequence_sha256"],
            enzyme["form"],
            reactions[row["reaction_id"]]["reaction_id"],
            row["condition_id"],
        )
        grouped[key].append(row)
    groups = []
    for key, rows in sorted(grouped.items()):
        states = {row["status"] for row in rows}
        if len(states) != 1 and any(r["independent_id"] is None for r in rows):
            raise ValueError(
                "Resolve conflicting states for an identical enzyme/form/reaction/condition group"
            )
        independent = defaultdict(list)
        if len(rows) > 1:
            if any(r["independent_id"] is None for r in rows):
                raise ValueError(
                    "Multiple aggregate rows for a pair require explicit independent replicate identities"
                )
            identities = [(r["independent_id"], r["technical_id"]) for r in rows]
            if len(identities) != len(set(identities)):
                raise ValueError("Duplicate independent/technical replicate identity")
            for independent_id in {r["independent_id"] for r in rows}:
                replicates = [r for r in rows if r["independent_id"] == independent_id]
                if len(replicates) > 1 and any(
                    r["technical_id"] is None for r in replicates
                ):
                    raise ValueError(
                        "Nested technical replicates require every technical ID"
                    )
        for r in rows:
            if r["status"] == "measured":
                independent[r["independent_id"] or r["id"]].append(r["value"])
        means = [sum(values) / len(values) for _, values in sorted(independent.items())]
        value = sum(means) / len(means) if means else None
        se = (
            math.sqrt(
                sum((v - value) ** 2 for v in means) / (len(means) * (len(means) - 1))
            )
            if len(means) > 1
            else None
        )
        groups.append(
            {
                "id": "pair-" + digest(key),
                "sequence_sha256": key[0],
                "form": key[1],
                "reaction_identity": key[2],
                "reaction_novelty_identity": reactions[rows[0]["reaction_id"]][
                    "novelty_identity"
                ],
                "condition_id": key[3],
                "enzyme_ids": sorted({r["enzyme_id"] for r in rows}),
                "reaction_ids": sorted({r["reaction_id"] for r in rows}),
                "observation_ids": sorted(r["id"] for r in rows),
                "status": "measured"
                if means
                else next(iter(states))
                if len(states) == 1
                else "mixed",
                "observation_state_counts": dict(
                    sorted(Counter(r["status"] for r in rows).items())
                ),
                "value": value,
                "independent_n": len(means),
                "standard_error": se,
                "reported_sd": {
                    r["id"]: r["reported_sd"]
                    for r in sorted(rows, key=lambda r: r["id"])
                },
            }
        )
    if len(groups) > 2000:
        raise ValueError(
            "This paired-kernel profile accepts at most 2000 enzyme/reaction/condition groups"
        )
    return enzymes, reactions, observations, groups


def validate_parameters(parameters, enzymes, observations, groups):
    fields(parameters, "alpha splits candidate_ids", "Parameters")
    numeric(parameters["alpha"], "alpha")
    if not 1e-8 <= parameters["alpha"] <= 1e6:
        raise ValueError("alpha must be between 1e-8 and 1e6")
    selected = parameters["candidate_ids"]
    if (
        not isinstance(selected, list)
        or not 1 <= len(selected) <= 500
        or any(not isinstance(x, str) for x in selected)
        or len(selected) != len(set(selected))
        or set(selected) - enzymes.keys()
    ):
        raise ValueError("candidate_ids must identify 1–500 distinct supplied enzymes")
    splits = keyed(parameters["splits"], "splits", 10)
    resolved = []
    for split in splits.values():
        fields(split, "id novelty train evaluation", "Split")
        if split["novelty"] not in {"sequence", "reaction", "both", "pair"}:
            raise ValueError("Split novelty must be sequence, reaction, both or pair")
        partitions = {}
        for name in ("train", "evaluation"):
            ids = split[name]
            if (
                not isinstance(ids, list)
                or not ids
                or any(not isinstance(i, str) for i in ids)
                or len(ids) != len(set(ids))
                or set(ids) - observations.keys()
            ):
                raise ValueError(
                    "Split partitions require distinct known observation IDs"
                )
            chosen = set(ids)
            partitions[name] = []
            for group in groups:
                members = set(group["observation_ids"])
                if members & chosen:
                    if not members <= chosen:
                        raise ValueError(
                            "Every replicate of a pair belongs to the same complete partition"
                        )
                    partitions[name].append(group)
        train, evaluation = partitions["train"], partitions["evaluation"]
        for field, required in (
            ("sequence_sha256", split["novelty"] in {"sequence", "both"}),
            ("reaction_novelty_identity", split["novelty"] in {"reaction", "both"}),
        ):
            if required and {g[field] for g in train} & {g[field] for g in evaluation}:
                raise ValueError(
                    f"Declared {field} novelty has train/evaluation overlap"
                )

        def pair(g):
            return g["sequence_sha256"], g["reaction_novelty_identity"]

        if {pair(g) for g in train} & {pair(g) for g in evaluation}:
            raise ValueError(
                "An identical sequence/reaction pair crosses train/evaluation partitions"
            )
        if (
            sum(g["value"] is not None for g in train) < 3
            or sum(g["value"] is not None for g in evaluation) < 2
        ):
            raise ValueError(
                "Each split requires at least three measured training groups and two evaluation groups"
            )
        resolved.append((split, partitions))
    return resolved


def pair_kernels(groups, enzymes, reactions):
    import numpy as np
    from rdkit.Chem import rdFingerprintGenerator

    Chem = chemistry()
    by_id = {e["id"]: e for e in enzymes}
    seqs = sorted({g["sequence_sha256"] for g in groups})
    sequences = {e["sequence_sha256"]: e["sequence"] for e in enzymes}
    indices = {aa: i for i, aa in enumerate(AA)}
    features = np.zeros((len(seqs), 8000))
    for i, key in enumerate(seqs):
        sequence = sequences[key]
        for a, b, c in zip(sequence, sequence[1:], sequence[2:], strict=False):
            features[i, indices[a] * 400 + indices[b] * 20 + indices[c]] += 1
    # Integer-valued binary64 products/sums are exact here: 10,000 residues
    # bound every dot product by 9,998**2, comfortably below 2**53.
    sequence_dot = (features @ features.T).astype(np.int64)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    seq_index = {s: i for i, s in enumerate(seqs)}
    si = [seq_index[by_id[g["enzyme_ids"][0]]["sequence_sha256"]] for g in groups]
    seq_kernel = features @ features.T
    seq_kernel = np.clip(seq_kernel[np.ix_(si, si)], 0, 1)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=512, includeChirality=True
    )
    reaction_bits = {}
    for reaction in reactions:
        bits = np.zeros(1024, dtype=float)
        for p in reaction["participants"]:
            if p["id"] in reaction["feature_participants"]:
                offset = 0 if p["side"] == "reactant" else 512
                fp = generator.GetFingerprint(Chem.MolFromSmiles(p["canonical_smiles"]))
                bits[[offset + int(i) for i in fp.GetOnBits()]] = 1
        key = reaction["reaction_id"]
        if key in reaction_bits and not np.array_equal(bits, reaction_bits[key]):
            raise ValueError(
                "Identical reaction aliases require identical feature selection"
            )
        reaction_bits[key] = bits
    matrix = np.stack([reaction_bits[g["reaction_identity"]] for g in groups])
    shared = matrix @ matrix.T
    total = matrix.sum(axis=1)
    union = total[:, None] + total[None, :] - shared
    reaction_kernel = shared / union
    kernels = {
        "sequence-ridge": seq_kernel,
        "reaction-ridge": reaction_kernel,
        "joint-ridge": seq_kernel * reaction_kernel,
    }
    counts = {
        "sequence_dot": sequence_dot[np.ix_(si, si)],
        "sequence_norm2": sequence_dot.diagonal()[si],
        "reaction_intersection": shared.astype(np.int64),
        "reaction_union": union.astype(np.int64),
    }
    return kernels, counts


def nearest_reference(query, candidates, counts):
    """Maximize exact squared similarity, retaining canonical order for ties."""
    best, best_numerator, best_denominator = 0, -1, 1
    for position, candidate in enumerate(candidates):
        # The query norm is a common positive factor and cancels. Similarities
        # are nonnegative, so squaring preserves order. Python integers retain
        # exact cross-products beyond the fixed-width count arrays.
        dot = int(counts["sequence_dot"][query, candidate])
        shared = int(counts["reaction_intersection"][query, candidate])
        union = int(counts["reaction_union"][query, candidate])
        norm = int(counts["sequence_norm2"][candidate])
        numerator, denominator = (dot * shared) ** 2, norm * union**2
        if numerator * best_denominator > best_numerator * denominator:
            best, best_numerator, best_denominator = position, numerator, denominator
    return best


def metrics(rows, method):
    import numpy as np
    from scipy.stats import spearmanr

    observed = np.array([r["observed"] for r in rows])
    predicted = np.array([r["predictions"][method] for r in rows])
    delta = predicted - observed
    rho = (
        float(spearmanr(observed, predicted).statistic)
        if len(rows) > 1 and np.ptp(observed) > 0 and np.ptp(predicted) > 0
        else None
    )
    return {
        "n": len(rows),
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt((delta**2).mean())),
        "bias": float(delta.mean()),
        "spearman_rho": rho,
    }


def analyze_panel(panel, parameters):
    import numpy as np
    from sklearn.kernel_ridge import KernelRidge

    enzymes, reactions, observations, groups = validate_panel(panel)
    splits = validate_parameters(parameters, enzymes, observations, groups)
    kernels, counts = pair_kernels(
        groups, list(enzymes.values()), list(reactions.values())
    )
    indices = {g["id"]: i for i, g in enumerate(groups)}
    evaluations = []
    for split, partitions in splits:
        training = [g for g in partitions["train"] if g["value"] is not None]
        ti = [indices[g["id"]] for g in training]
        y = np.asarray([g["value"] for g in training])
        mean = float(y.mean())
        models, predictions = {}, {}
        for method, kernel in kernels.items():
            model = KernelRidge(alpha=parameters["alpha"], kernel="precomputed")
            model.fit(kernel[np.ix_(ti, ti)], y - mean)
            # Evaluate each identical feature row once. Batched BLAS can use
            # different summation paths for duplicate rows, creating spurious
            # ranks in a mathematically constant reaction-only prediction.
            unique_rows, inverse = np.unique(kernel[:, ti], axis=0, return_inverse=True)
            predictions[method] = model.predict(unique_rows)[inverse] + mean
            models[method] = {
                "training_mean": mean,
                "alpha": parameters["alpha"],
                "dual_coefficients": model.dual_coef_.tolist(),
                "training_group_ids": [g["id"] for g in training],
            }
        rows, scored = [], []
        for g in partitions["evaluation"]:
            i = indices[g["id"]]
            nearest = nearest_reference(i, ti, counts)
            reference = training[nearest]
            j = ti[nearest]
            prediction = {
                method: float(values[i]) for method, values in predictions.items()
            }
            prediction.update(
                {"training-mean": mean, "nearest-reference": reference["value"]}
            )
            row = {
                "group_id": g["id"],
                "enzyme_ids": g["enzyme_ids"],
                "reaction_ids": g["reaction_ids"],
                "condition_id": g["condition_id"],
                "observation_ids": g["observation_ids"],
                "status": g["status"],
                "observed": g["value"],
                "predictions": prediction,
                "nearest_reference": {
                    "group_id": reference["id"],
                    "enzyme_ids": reference["enzyme_ids"],
                    "reaction_ids": reference["reaction_ids"],
                    "condition_id": reference["condition_id"],
                    "observation_ids": reference["observation_ids"],
                    "sequence_similarity": float(kernels["sequence-ridge"][i, j]),
                    "reaction_similarity": float(kernels["reaction-ridge"][i, j]),
                    "joint_similarity": float(kernels["joint-ridge"][i, j]),
                },
            }
            rows.append(row)
            if g["value"] is not None:
                scored.append(row)
        methods = ["training-mean", "nearest-reference", *kernels]
        by_condition = {
            key: {
                m: metrics([r for r in scored if r["condition_id"] == key], m)
                for m in methods
            }
            for key in sorted({r["condition_id"] for r in scored})
        }
        evaluations.append(
            {
                "id": split["id"],
                "novelty": split["novelty"],
                "training": [
                    {k: g[k] for k in ("id", "value", "observation_ids")}
                    for g in training
                ],
                "training_state_counts": dict(
                    sorted(Counter(g["status"] for g in partitions["train"]).items())
                ),
                "evaluation_state_counts": dict(
                    sorted(
                        Counter(g["status"] for g in partitions["evaluation"]).items()
                    )
                ),
                "training_sha256": digest(training),
                "evaluation_sha256": digest(partitions["evaluation"]),
                "models": models,
                "rows": rows,
                "metrics": {m: metrics(scored, m) for m in methods},
                "metrics_by_condition": by_condition,
            }
        )
    candidates = [
        {
            "sample_id": k,
            "sequence": enzymes[k]["sequence"],
            "sequence_sha256": enzymes[k]["sequence_sha256"],
            "group_id": enzymes[k]["sequence_sha256"],
            "partition": "pool",
        }
        for k in parameters["candidate_ids"]
    ]
    return {
        "schema_version": "0.1",
        "profile": PROFILE,
        "panel": copy.deepcopy(panel),
        "parameters": copy.deepcopy(parameters),
        "request_sha256": digest({"panel": panel, "parameters": parameters}),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("rdkit", "numpy", "scipy", "scikit-learn")
        },
        "counts": dict(
            sorted(Counter(r["status"] for r in observations.values()).items())
        ),
        "groups": groups,
        "reactions": list(reactions.values()),
        "evaluations": evaluations,
        "candidate_panel": {
            "schema_version": "0.1",
            "candidates": candidates,
            "source": {
                "panel_id": panel["id"],
                "panel_sha256": digest(panel),
                "selection": "Researcher-supplied candidate_ids; exported for a separate downstream analysis",
            },
        },
        "method": "Normalized amino-acid 3-mer count cosine and chirality-aware radius-2 Morgan presence fingerprints (512 bits per reaction side) define sequence and reaction kernels. Selected chemical participants define the fingerprint; all participants enter exact reaction accounting. The joint kernel multiplies both similarities. Kernel ridge uses the supplied alpha and training-mean centering; identical kernel rows share one prediction. Nearest reference maximizes exact squared joint similarity using integer count cross-products; exact ties retain canonical sequence/form/reaction/condition group order. Technical means are averaged equally across independent replicates. Reported source uncertainty is stored with the observations; measured-target regressions use equal group weights. Errors describe the supplied held-out numeric subset. Screen conditions and enzyme preparation are recorded with every observation.",
    }
