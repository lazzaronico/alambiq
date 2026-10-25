"""Supplied long-form measurements."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict

from alambiq.capabilities.measure.science.tables import (
    OPTIONAL,
    STATES,
    convert_row,
    finite,
    mapped_csv,
    validate_condition_values,
)
from alambiq.capabilities.shared.sequences import parse_fasta, parse_panel
from alambiq.capabilities.shared.sources import source_text as source_text

COLUMNS = (
    "measurement_id",
    "sample_id",
    "condition_id",
    "metric",
    "value",
    "unit",
    "independent_id",
    "technical_id",
    "role",
)
GROUP = ("sample_id", "condition_id", "metric", "unit", "role")
ROLES = {"sample", "reference", "blank", "control"}


def link_sample_constructs(sources, sequences, sample_ids):
    """Join supplied records."""
    if not isinstance(sources, dict) or not 1 <= len(sources) <= 500:
        raise ValueError(
            "sample_constructs must map 1 to 500 sample_id values of the assay to JSON "
            "sources."
        )
    samples = []
    for sample_id, spec in sorted(sources.items()):
        if sample_id not in sample_ids or sample_id not in sequences:
            raise ValueError(
                f"sample_constructs names the sample {sample_id}, which is not a "
                "sample_id of the assay or has no sequence in the sample FASTA."
            )
        text = source_text(spec)
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"The construct source of sample {sample_id} is not JSON."
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"The construct source of sample {sample_id} must be one JSON object."
            )
        dna, protein = data.get("sequence"), data.get("translation")
        if (
            data.get("schema_version") != "0.1"
            or data.get("topology") != "circular"
            or not isinstance(dna, str)
            or not 1 <= len(dna) <= 26000
            or set(dna) - set("ACGT")
            or type(data.get("length")) is not int
            or data["length"] != len(dna)
            or data.get("sequence_sha256") != hashlib.sha256(dna.encode()).hexdigest()
            or not isinstance(protein, str)
            or not protein
            or set(protein) - set("ACDEFGHIKLMNPQRSTVWY")
            or data.get("protein_sequence_sha256")
            != hashlib.sha256(protein.encode()).hexdigest()
            or protein != sequences[sample_id]
        ):
            raise ValueError(
                f"The construct record of sample {sample_id} is inconsistent: its "
                "schema_version, topology, length, sequence digest and protein digest "
                "must agree with its sequence and translation, and its translation must "
                "equal the sample protein."
            )
        samples.append(
            {
                "sample_id": sample_id,
                "length": len(dna),
                "topology": "circular",
                "sequence_sha256": data["sequence_sha256"],
                "protein_sequence_sha256": data["protein_sequence_sha256"],
                "source": {
                    **{k: v for k, v in spec.items() if k not in {"path", "text"}},
                    "text": text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                },
            }
        )
    return {
        "samples": samples,
        "unlinked_sample_ids": sorted(sample_ids - sources.keys()),
        "method": "The association of samples with constructs is supplied by the researcher. The source bytes and the recorded DNA and protein digests are checked, and the recorded translation equals the supplied sample sequence.",
    }


def link_sample_sequences(sample_ids, fasta, reference_fasta=None):
    """Resolve the exact supplied sample and sequence identities."""
    records = parse_panel(fasta)
    supplied = {r["id"] for r in records}
    if supplied - sample_ids:
        raise ValueError(
            "Every identifier of the sample FASTA must equal a sample_id of the assay; "
            f"unknown: {', '.join(sorted(supplied - sample_ids))}."
        )
    reference = None
    if reference_fasta is not None:
        identifier, sequence = parse_fasta(reference_fasta)
        reference = {
            "id": identifier,
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        }
    samples = []
    for record in records:
        sequence = record["sequence"]
        comparable = reference is not None and len(sequence) == len(
            reference["sequence"]
        )
        differences = (
            [
                {"position": i, "reference": a, "sample": b}
                for i, (a, b) in enumerate(
                    zip(reference["sequence"], sequence, strict=True), 1
                )
                if a != b
            ]
            if comparable
            else []
        )
        ambiguous = (
            [
                i
                for i, (a, b) in enumerate(
                    zip(reference["sequence"], sequence, strict=True), 1
                )
                if a not in "ACDEFGHIKLMNPQRSTVWY" or b not in "ACDEFGHIKLMNPQRSTVWY"
            ]
            if comparable
            else []
        )
        samples.append(
            {
                "sample_id": record["id"],
                "description": record["description"],
                "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "comparison": "same-length-positional"
                if comparable
                else "no-reference"
                if reference is None
                else "different-lengths",
                "differences": differences,
                "ambiguous_positions": ambiguous,
            }
        )
    return {
        "samples": samples,
        "reference": reference,
        "unlinked_sample_ids": sorted(sample_ids - supplied),
    }


def _label(value, name, optional=False):
    if (
        not isinstance(value, str)
        or len(value) > 200
        or any(ord(c) < 32 for c in value)
        or (not optional and not value.strip())
    ):
        raise ValueError(
            f"{name} must be a single-line text of at most 200 characters"
            + ("." if optional else ", not blank.")
        )
    return value.strip()


def _context(conditions, independent_unit):
    if not isinstance(conditions, dict) or not 1 <= len(conditions) <= 100:
        raise ValueError(
            "conditions must map 1 to 100 condition identifiers to descriptions."
        )
    if any(
        _label(k, "condition id") != k
        or not isinstance(v, str)
        or not 1 <= len(v.strip()) <= 4000
        for k, v in conditions.items()
    ):
        raise ValueError(
            "Each condition must have a description of 1 to 4,000 characters."
        )
    if independent_unit is not None:
        independent_unit = _label(independent_unit, "independent_unit")
    return independent_unit


def inspect_assay(
    text,
    conditions,
    independent_unit=None,
    *,
    table_format="csv",
    columns=None,
    defaults=None,
    units=None,
    condition_values=None,
    sample_descriptions=None,
):
    independent_unit = _context(conditions, independent_unit)
    normalized, intake = mapped_csv(text, COLUMNS, table_format, columns, defaults)
    if units is not None and (
        not isinstance(units, dict)
        or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in units.items()
        )
    ):
        raise ValueError("units must map metric names to target unit names.")
    if condition_values is not None:
        validate_condition_values(condition_values, conditions)
    reader = csv.DictReader(io.StringIO(normalized, newline=""), strict=True)
    try:
        header = reader.fieldnames
    except csv.Error as exc:
        raise ValueError(f"The CSV header is malformed: {exc}") from exc
    if (
        header is None
        or len(header) != len(set(header))
        or not set(COLUMNS).issubset(header)
        or set(header) - set(COLUMNS) - set(OPTIONAL)
    ):
        lacking = [c for c in COLUMNS if not header or c not in header]
        unknown = [c for c in (header or []) if c not in COLUMNS and c not in OPTIONAL]
        repeated = sorted({c for c in (header or []) if (header or []).count(c) > 1})
        problems = [
            *([f"missing columns: {', '.join(lacking)}"] if lacking else []),
            *([f"not recognized: {', '.join(unknown)}"] if unknown else []),
            *([f"repeated: {', '.join(repeated)}"] if repeated else []),
        ]
        raise ValueError(
            "The table header is wrong ("
            + "; ".join(problems or ["no header row"])
            + "). The required columns are "
            + ", ".join(COLUMNS)
            + "; the optional columns are "
            + ", ".join(OPTIONAL)
            + "."
        )
    rows, identifiers, technical_keys, group_roles = [], set(), set(), {}
    try:
        for index, raw in enumerate(reader, 1):
            if index > 10000:
                raise ValueError(
                    "The table has more than 10,000 rows; at most 10,000 measurements "
                    "are accepted."
                )
            if None in raw or None in raw.values():
                raise ValueError(
                    f"CSV record {index} has a number of cells that differs from the "
                    "number of columns in the header."
                )
            row = {
                key: _label(
                    raw[key], key, key in {"value", "independent_id", "technical_id"}
                )
                for key in COLUMNS
            }
            if row["measurement_id"] in identifiers:
                raise ValueError(
                    f"measurement_id {row['measurement_id']!r} appears more than once; "
                    "each measurement_id must be unique, and duplicates are neither "
                    "selected nor averaged."
                )
            identifiers.add(row["measurement_id"])
            if row["condition_id"] not in conditions:
                raise ValueError(
                    f"The condition {row['condition_id']!r} of measurement "
                    f"{row['measurement_id']} is not a key of conditions."
                )
            if (
                row["role"] not in ROLES
                or group_roles.setdefault(
                    tuple(row[k] for k in GROUP[:-1]), row["role"]
                )
                != row["role"]
            ):
                raise ValueError(
                    f"Measurement {row['measurement_id']}: role {row['role']!r} must be "
                    "sample, reference, blank or control, and must equal the role of "
                    "the other measurements of the same sample, condition, metric and "
                    "unit."
                )
            if bool(row["independent_id"]) != bool(row["technical_id"]):
                raise ValueError(
                    f"Measurement {row['measurement_id']}: independent_id and "
                    "technical_id must both be set or both be empty."
                )
            value = None
            if row["value"]:
                try:
                    value = float(row["value"])
                except ValueError as exc:
                    raise ValueError(
                        f"The value of measurement {row['measurement_id']} must be a "
                        "finite number or empty."
                    ) from exc
                if not math.isfinite(value) or abs(value) > 1e100:
                    raise ValueError(
                        f"Measurement {row['measurement_id']}: the value must be finite "
                        "with a magnitude of at most 1e100. Express larger quantities "
                        "in a larger unit."
                    )
            if "reported_standard_error" in raw:
                error = _label(
                    raw["reported_standard_error"],
                    "reported_standard_error",
                    optional=True,
                )
                try:
                    error = float(error) if error else None
                except ValueError as exc:
                    raise ValueError(
                        "reported_standard_error must be empty or a finite number from 0 "
                        "to 1e100."
                    ) from exc
                if error is not None and (
                    value is None or not math.isfinite(error) or not 0 <= error <= 1e100
                ):
                    raise ValueError(
                        f"Measurement {row['measurement_id']}: reported_standard_error "
                        "requires a measured value and must be a finite number from 0 "
                        "to 1e100 in the unit of the measurement."
                    )
                row["reported_standard_error"] = error
            status = raw.get("status", "").strip() or (
                "measured" if value is not None else "missing"
            )
            if status not in STATES or (status == "measured") != (value is not None):
                raise ValueError(
                    f"Measurement {row['measurement_id']}: the status must agree with "
                    "the value. A measured record has a number; a missing, failed or "
                    "censored record has an empty value."
                )
            limit = raw.get("limit", "").strip()
            if bool(limit) != (status in {"below-limit", "above-limit"}):
                raise ValueError(
                    f"Measurement {row['measurement_id']}: a below-limit or above-limit "
                    "record has a numeric limit and no value, and no other record has "
                    "a limit."
                )
            row.update(
                value=value,
                status=status,
                limit=finite(limit, "limit") if limit else None,
                source_record=index,
            )
            if "note" in raw:
                row["note"] = _label(raw["note"], "note", optional=True)
            if row["metric"] in (units or {}):
                convert_row(row, units[row["metric"]])
            if row["independent_id"]:
                if independent_unit is None:
                    raise ValueError(
                        "independent_unit must be set when independent_id values are "
                        "supplied."
                    )
                key = tuple(row[k] for k in (*GROUP, "independent_id", "technical_id"))
                if key in technical_keys:
                    raise ValueError(
                        f"Measurement {row['measurement_id']} repeats the independent_id "
                        "and technical_id of another measurement of the same sample, "
                        "condition, metric and unit (after the requested unit "
                        "conversions)."
                    )
                technical_keys.add(key)
            rows.append(row)
    except csv.Error as exc:
        raise ValueError(f"The CSV table is malformed: {exc}") from exc
    if set(units or {}) - {r["metric"] for r in rows}:
        raise ValueError(
            "units names metrics that no measurement has: "
            + ", ".join(sorted(set(units or {}) - {r["metric"] for r in rows}))
            + "."
        )
    if sample_descriptions is not None:
        if not isinstance(sample_descriptions, dict) or set(sample_descriptions) - {
            r["sample_id"] for r in rows
        }:
            raise ValueError(
                "sample_descriptions must map sample_id values of the table to "
                "descriptions."
            )
        for description in sample_descriptions.values():
            if (
                not isinstance(description, str)
                or not 1 <= len(description.strip()) <= 4000
            ):
                raise ValueError(
                    "Each sample description must have 1 to 4,000 characters."
                )
    # Conversion can put originally separate units in the same analysis group.
    group_roles.clear()
    for row in rows:
        if (
            group_roles.setdefault(tuple(row[k] for k in GROUP[:-1]), row["role"])
            != row["role"]
        ):
            raise ValueError(
                "After the unit conversion, measurements of the same sample, condition, "
                "metric and unit have different roles."
            )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in GROUP)].append(row)
    if len(grouped) > 500:
        raise ValueError(
            f"The table has {len(grouped)} groups of sample, condition, metric, unit and "
            "role; at most 500 are accepted."
        )
    groups = []
    for key, members in sorted(grouped.items()):
        by_replicate = defaultdict(list)
        for row in members:
            if row["independent_id"]:
                by_replicate[row["independent_id"]].append(row)
        replicates = []
        for identifier, technical in sorted(by_replicate.items()):
            values = [r["value"] for r in technical if r["value"] is not None]
            replicates.append(
                {
                    "independent_id": identifier,
                    "measurement_ids": [r["measurement_id"] for r in technical],
                    "observed_count": len(values),
                    "missing_count": len(technical) - len(values),
                    "mean": statistics.mean(values) if values else None,
                    "standard_deviation": statistics.stdev(values)
                    if len(values) > 1
                    else None,
                }
            )
        means = [r["mean"] for r in replicates if r["mean"] is not None]
        missing = sum(r["value"] is None for r in members)
        unassigned = sum(
            not r["independent_id"] and r["value"] is not None for r in members
        )
        aggregation = (
            "no-values"
            if missing == len(members)
            else "unassigned-replicates"
            if unassigned
            else "independent-means"
        )
        groups.append(
            {
                "id": "group-"
                + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16],
                **dict(zip(GROUP, key, strict=True)),
                "measurement_ids": [r["measurement_id"] for r in members],
                "measurement_count": len(members),
                "missing_count": missing,
                "status_counts": dict(Counter(r["status"] for r in members)),
                "unassigned_count": unassigned,
                "independent_count": len(means),
                "replicates": replicates,
                "aggregation": aggregation,
                "mean": statistics.mean(means) if means and not unassigned else None,
                "standard_deviation": statistics.stdev(means)
                if len(means) > 1 and not unassigned
                else None,
            }
        )
    data = {
        "schema_version": "0.1",
        "source_text": text,
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "conditions": conditions,
        "independent_unit": independent_unit,
        "rows": rows,
        "groups": groups,
        "intake": {**intake, "units": units or {}},
        "method": "Arithmetic means within named independent units, then equal-weight mean and sample SD (n-1) of those means. Missing, failed and censored records are retained, not imputed or pooled as observations. Unassigned observed values prevent group pooling. "
        + (
            "Explicit unit conversions are recorded beside original values. "
            if units
            else "No unit conversion. "
        )
        + "The summaries use the supplied values.",
    }
    if condition_values is not None:
        data["condition_values"] = condition_values
    if sample_descriptions is not None:
        data["sample_descriptions"] = sample_descriptions
    return data


def reconcile_worklist(text, data):
    """Compare an explicit expected-measurement table."""
    try:
        header = next(csv.reader(io.StringIO(text), strict=True), [])
    except csv.Error as exc:
        raise ValueError(f"The worklist CSV is malformed: {exc}") from exc
    preparation_fields = {"preparation_status", "preparation_note"}
    has_preparation = bool(preparation_fields & set(header))
    if set(header) - {*COLUMNS, *OPTIONAL, *preparation_fields}:
        raise ValueError(
            "The worklist has columns that are not table fields: "
            + ", ".join(
                sorted(set(header) - {*COLUMNS, *OPTIONAL, *preparation_fields})
            )
            + "."
        )
    planned = inspect_assay(
        text,
        data["conditions"],
        data["independent_unit"],
        columns={k: k for k in header if k not in preparation_fields}
        if has_preparation
        else None,
    )
    if not planned["rows"]:
        raise ValueError("The worklist must contain at least 1 expected measurement.")
    preparations, states = {}, {}
    if has_preparation:
        for raw in csv.DictReader(io.StringIO(text), strict=True):
            status = raw.get("preparation_status", "").strip() or "unspecified"
            note = _label(
                raw.get("preparation_note", ""), "preparation_note", optional=True
            )
            if status not in {"unspecified", "available", "unbuilt"}:
                raise ValueError(
                    "preparation_status must be unspecified, available or unbuilt; "
                    f"found {status!r}."
                )
            key = (raw["sample_id"].strip(), raw["independent_id"].strip())
            if states.setdefault(key, status) != status:
                raise ValueError(
                    "A sample and independent unit have different preparation_status "
                    "values in the worklist."
                )
            preparations[raw["measurement_id"].strip()] = {
                "preparation_status": status,
                "preparation_note": note,
            }
    if any(
        row["value"] is not None or row["status"] != "missing"
        for row in planned["rows"]
    ):
        raise ValueError(
            "The value cells of the worklist must be blank; returned measurements "
            "belong in the measurement table."
        )
    observed = {r["measurement_id"]: r for r in data["rows"]}
    identity = tuple(k for k in COLUMNS if k not in {"measurement_id", "value"})
    rows = []
    for expected in planned["rows"]:
        actual = observed.get(expected["measurement_id"])
        preparation = preparations.get(expected["measurement_id"], {})
        if actual is not None and preparation.get("preparation_status") == "unbuilt":
            raise ValueError(
                f"Worklist measurement {expected['measurement_id']} has the "
                "preparation_status unbuilt, but the measurement table contains a "
                "record for it."
            )
        if actual is not None:
            differences = [
                k
                for k in identity
                if expected[k] != actual.get("original", {}).get(k, actual[k])
            ]
            if differences:
                raise ValueError(
                    f"Worklist measurement {expected['measurement_id']} differs from "
                    "the returned record in: " + ", ".join(differences)
                )
        rows.append(
            {
                **expected,
                **preparation,
                "return_status": "not-returned"
                if actual is None
                else "returned-value"
                if actual["value"] is not None
                else "returned-" + actual.get("status", "missing"),
                "measurement_source_record": None
                if actual is None
                else actual["source_record"],
            }
        )
    planned_ids = {r["measurement_id"] for r in rows}
    return {
        "source_text": text,
        "source_sha256": planned["source_sha256"],
        "rows": rows,
        "unplanned_measurement_ids": [
            r["measurement_id"]
            for r in data["rows"]
            if r["measurement_id"] not in planned_ids
        ],
    }
