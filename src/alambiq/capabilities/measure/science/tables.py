"""Small, explicit table mappings and physical unit conversions for assay intake."""

import csv
import io
import json
import math

OPTIONAL = ("reported_standard_error", "status", "limit", "note")
STATES = {"measured", "missing", "failed", "below-limit", "above-limit"}


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or abs(result) > 1e100:
        raise ValueError(f"{name} must be finite with magnitude at most 1e100")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"A JSON object repeats the field {key!r}.")
        result[key] = value
    return result


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("A table cell must be a string, a finite number or null.")
    if isinstance(value, (int, float)):
        finite(value, "Table cell")
    return str(value)


def mapped_csv(text, required, table_format="csv", columns=None, defaults=None):
    """Normalize syntax; columns, identifiers and missing-value tokens are as supplied."""
    if table_format not in {"csv", "json"}:
        raise ValueError(f"table_format must be csv or json; found {table_format!r}.")
    if columns is not None and (not isinstance(columns, dict) or not columns):
        raise ValueError(
            "columns must be a nonempty mapping from canonical field names to source "
            "column names."
        )
    if defaults is not None and not isinstance(defaults, dict):
        raise ValueError(
            "defaults must be a mapping from canonical field names to scalar values."
        )
    allowed = {*required, *OPTIONAL}
    if set(columns or {}) - allowed or set(defaults or {}) - allowed:
        raise ValueError(
            "columns or defaults names fields that are not table fields: "
            + ", ".join(sorted((set(columns or {}) | set(defaults or {})) - allowed))
            + ". The table fields are "
            + ", ".join(sorted(allowed))
            + "."
        )
    if any(not isinstance(v, str) or not v for v in (columns or {}).values()):
        raise ValueError("Each entry of columns must name a source column.")
    if len(set((columns or {}).values())) != len(columns or {}):
        raise ValueError("Each source column may occur only once in columns.")
    defaults = {k: _cell(v) for k, v in (defaults or {}).items()}
    if table_format == "json":
        try:
            records = json.loads(text, object_pairs_hook=_unique_object)
        except json.JSONDecodeError as exc:
            raise ValueError(f"The JSON table is malformed: {exc}") from exc
        if not isinstance(records, list) or not 1 <= len(records) <= 10000:
            raise ValueError(
                "The JSON table must be an array of 1 to 10,000 flat records."
            )
        if any(not isinstance(r, dict) for r in records):
            raise ValueError("Each record of the JSON table must be an object.")
        header = list(records[0])
        if any(set(r) != set(header) for r in records):
            raise ValueError(
                "All records of the JSON table must have the same fields. Use null for "
                "a missing cell."
            )
        records = [{k: _cell(v) for k, v in r.items()} for r in records]
    else:
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
            header = reader.fieldnames
            if not header or len(header) != len(set(header)):
                raise ValueError("The table header must not repeat a field name.")
            records = []
            for raw in reader:
                if len(records) >= 10000:
                    raise ValueError(
                        "The table has more than 10,000 rows; at most 10,000 "
                        "measurements are accepted."
                    )
                if None in raw or None in raw.values():
                    raise ValueError(
                        "A CSV row has a number of cells that differs from the number "
                        "of columns in the header."
                    )
                records.append(raw)
        except csv.Error as exc:
            raise ValueError(f"The CSV table is malformed: {exc}") from exc
    if not header or len(header) != len(set(header)):
        raise ValueError("The table header must not repeat a field name.")
    mapping = dict(columns) if columns is not None else {k: k for k in header}
    if set(mapping) - allowed:
        raise ValueError(
            "The table header contains the columns "
            + ", ".join(sorted(set(mapping) - allowed))
            + ", which are not table fields (required: "
            + ", ".join(required)
            + "; optional: "
            + ", ".join(OPTIONAL)
            + "). To use other column names, set columns, which maps each canonical "
            "field to a source column."
        )
    if set(mapping) & defaults.keys():
        raise ValueError(
            "A table field must not have both a source column and a default: "
            + ", ".join(sorted(set(mapping) & defaults.keys()))
            + "."
        )
    absent = [name for name in mapping.values() if name not in header]
    if absent:
        raise ValueError(
            "columns names source columns that the table does not have: "
            + ", ".join(absent)
            + ". The table has the columns "
            + ", ".join(header)
            + "."
        )
    unfilled = [
        name for name in required if name not in mapping and name not in defaults
    ]
    if unfilled:
        raise ValueError(
            "The required table fields "
            + ", ".join(unfilled)
            + " have neither a source column nor a default."
        )
    fields = [*required, *(k for k in OPTIONAL if k in mapping or k in defaults)]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        writer.writerow({**defaults, **{k: record[v] for k, v in mapping.items()}})
    return output.getvalue(), {
        "format": table_format,
        "columns": columns,
        "defaults": defaults,
        "unused_columns": [k for k in header if k not in mapping.values()],
    }


# Factors and offsets into each dimension's base unit. No molecular-weight,
# active-enzyme, extinction-coefficient or assay calibration assumptions.
UNITS = {
    "M": ("concentration", 1, 0),
    "mM": ("concentration", 1e-3, 0),
    "uM": ("concentration", 1e-6, 0),
    "nM": ("concentration", 1e-9, 0),
    "s": ("time", 1, 0),
    "min": ("time", 60, 0),
    "h": ("time", 3600, 0),
    "K": ("temperature", 1, 0),
    "degC": ("temperature", 1, 273.15),
    "mg/mL": ("mass-concentration", 1, 0),
    "g/L": ("mass-concentration", 1, 0),
    "ug/mL": ("mass-concentration", 1e-3, 0),
    "U/mg": ("specific-activity", 1, 0),
    "mU/mg": ("specific-activity", 1e-3, 0),
}
for concentration in ("M", "mM", "uM", "nM"):
    for time in ("s", "min", "h"):
        UNITS[f"{concentration}/{time}"] = (
            "concentration-rate",
            UNITS[concentration][1] / UNITS[time][1],
            0,
        )


def convert_row(row, target):
    source = row["unit"]
    if (
        source not in UNITS
        or target not in UNITS
        or UNITS[source][0] != UNITS[target][0]
    ):
        raise ValueError(
            f"The conversion from unit {source!r} to unit {target!r} is not supported: "
            "both units must be known and of the same dimension."
        )
    _, scale, offset = UNITS[source]
    _, target_scale, target_offset = UNITS[target]
    row["original"] = {
        k: row[k]
        for k in ("value", "unit", "limit", "reported_standard_error")
        if k in row
    }
    factor, shift = scale / target_scale, (offset - target_offset) / target_scale
    for key in ("value", "limit", "reported_standard_error"):
        if row.get(key) is not None:
            row[key] = finite(
                row[key] * factor + (0 if key == "reported_standard_error" else shift),
                key,
            )
    row["unit"] = target


def validate_condition_values(values, conditions):
    if not isinstance(values, dict) or set(values) - set(conditions):
        raise ValueError(
            "condition_values must be a mapping whose keys are condition identifiers "
            "of the request."
        )
    for fields in values.values():
        if not isinstance(fields, dict) or not 1 <= len(fields) <= 30:
            raise ValueError(
                "Each condition in condition_values must have 1 to 30 named quantities."
            )
        for name, quantity in fields.items():
            if not isinstance(name, str) or not name.strip() or len(name) > 200:
                raise ValueError(
                    "Each quantity name must be a nonempty string of at most 200 "
                    "characters."
                )
            if not isinstance(quantity, dict) or set(quantity) != {"value", "unit"}:
                raise ValueError(
                    f"The quantity {name} must have exactly the fields value and unit."
                )
            if isinstance(quantity["value"], str):
                raise ValueError(
                    f"The value of the quantity {name} must be a number, not a string."
                )
            finite(quantity["value"], name)
            if (
                not isinstance(quantity["unit"], str)
                or not quantity["unit"].strip()
                or len(quantity["unit"]) > 200
            ):
                raise ValueError(
                    f"The unit of the quantity {name} must be a nonempty string of at "
                    "most 200 characters."
                )
