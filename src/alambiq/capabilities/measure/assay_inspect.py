"""Read a table of measurements into a stored record.

The table is read as given. Each row is one measurement and stores its state,
so that a missing value and a measured zero are distinct. Four states are
distinguished: measured, missing, failed and censored. A censored measurement
has `status` `below-limit` or `above-limit` and stores its limit in `limit`; its
`value` is empty and is not set to the limit.

Replicate structure is preserved. An independent unit is one independent
replicate of a sample; the `independent_unit` request field describes it (for
example `preparation`). A technical reading is a repeated measurement of one
independent unit. Each row carries `independent_id` and `technical_id`, so that
an analysis can average technical readings within each independent unit.
Standard errors reported by a source are retained as reported.

Limits. At most 10,000 rows and 500 groups of rows that share sample,
condition, metric, unit and role. Conditions such as temperature and pH are
recorded as metadata; `assay.fit` uses one of them as the independent quantity
(`x_quantity`) and requires the others to be equal across the selected conditions
or pools the measurements across them (`pool_over`). Units are converted only for
the metrics named in the `units` request field, and only between units of one
dimension in a fixed table: concentration (M, mM, uM, nM), time (s, min, h), temperature (K, degC),
mass concentration (mg/mL, g/L, ug/mL), specific activity (U/mg, mU/mg) and
concentration rate (M, mM, uM or nM per s, min or h). A conversion that needs a
molecular weight, an extinction coefficient or an assay calibration is
refused. The original values are stored beside the converted ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from alambiq.capabilities.measure.inspect_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class InspectRequest:
    """The measurement table and the settings needed to read it.

    table:
        Path to the measurements, one row per measurement. Required columns:
        `measurement_id` (unique), `sample_id`, `condition_id`, `metric`, `value`
        (a number, or empty), `unit`, `independent_id` and `technical_id` (both
        empty when there is no replicate structure) and `role` (`sample`,
        `reference`, `blank` or `control`; it is stored with the row, the
        measurements of one sample, condition, metric and unit share one role,
        and values are stored as supplied). Optional
        columns: `status` (`measured`, `missing`, `failed`, `below-limit` or
        `above-limit`; when absent or empty it is `measured` if `value` is a
        number and `missing` if `value` is empty; `measured` requires a number
        and every other status requires an empty `value`), `limit` (a number,
        required for `below-limit` and `above-limit` and empty otherwise),
        `reported_standard_error` and `note`. At most 10,000 rows.
    conditions:
        A description for each condition id that the table refers to, 1 to 100
        conditions of 1 to 4,000 characters each. Every condition named by a row
        must appear here.
    independent_unit:
        The description of one independent replicate, for example
        `preparation`. Required when any row has an `independent_id`. A row
        without an `independent_id` is stored without replicate assignment;
        `assay.fit` with `supplied-values` fits each such row separately.
    table_format:
        `csv`, or `json` for a flat array of objects with the same fields in
        every object, one object per row. Default `csv`.
    condition_values:
        Numeric quantities per condition id, such as substrate concentration,
        each with a value and a unit (1 to 30 quantities per condition).
        `assay.fit` reads its independent quantity (`x_quantity`) from them.
    sample_descriptions:
        Maps a sample id to a nonblank description of at most 4,000 characters.
        Every key must be a sample id that occurs in the table.
    units:
        Requested unit conversions: each key is a `metric` of the table and each
        value is the target unit.
    columns:
        Maps each required or optional column name to the name of the column in
        the supplied table, for example `{"value": "reading", "unit": "units"}`.
        Applies when the headers differ from those names. When given, only the
        fields it names are read from the table; each source column may be used
        once.
    defaults:
        A constant used for a required or optional column in every row, such as
        `{"role": "sample"}`. A field takes its value from a source column or
        from a default, never both. Every required field must have one or the
        other, including `independent_id` and `technical_id`, which are given
        as the empty string when the table has no replicate structure.
    """

    table: str
    conditions: dict[str, str]
    independent_unit: str | None = None
    table_format: Literal["csv", "json"] = "csv"
    columns: dict[str, str] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    condition_values: dict[str, Any] = field(default_factory=dict)
    sample_descriptions: dict[str, str] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InspectResult:
    """The counts of rows in each state, the sample and condition ids and the path of the stored record."""

    measurements: int
    measured: int
    missing: int
    failed: int
    censored: int
    samples: list[str]
    conditions: list[str]
    independent_unit: str | None
    record: str
    """Path to the stored record. A later step binds its input to this path."""


def _count(rows: list[dict[str, Any]], state: str) -> int:
    return sum(1 for row in rows if row.get("status") == state)


@capability(
    "assay.inspect",
    version="1.0",
    family="measure",
    summary="Read a measurement table into a stored record",
    render=render,
)
def inspect(request: InspectRequest, ctx: StepContext) -> InspectResult:
    """Read the supplied table and report the state of each row."""
    from alambiq.capabilities.measure.science.inspection import inspect_assay

    record = inspect_assay(
        ctx.read_text(request.table),
        dict(request.conditions),
        request.independent_unit,
        table_format=request.table_format,
        columns=dict(request.columns) or None,
        defaults=dict(request.defaults) or None,
        condition_values=dict(request.condition_values) or None,
        sample_descriptions=dict(request.sample_descriptions) or None,
        units=dict(request.units) or None,
    )

    rows = record.get("rows") or []
    # A censored reading carries the limit it was censored at.
    censored = sum(1 for row in rows if row.get("limit") is not None)
    failed, missing = _count(rows, "failed"), _count(rows, "missing")
    measured = _count(rows, "measured")

    if censored:
        ctx.note(
            f"{censored} of {len(rows)} readings "
            f"{'is' if censored == 1 else 'are'} censored at a stated limit."
        )
    if failed or missing:
        failed_text = f"{failed} reading failed" if failed == 1 else f"{failed} readings failed"
        missing_text = f"{missing} is missing" if missing == 1 else f"{missing} are missing"
        ctx.note(
            f"{failed_text} and {missing_text}."
        )

    zeros = [
        row["measurement_id"]
        for row in rows
        if row.get("status") == "measured" and row.get("value") == 0
    ]
    if zeros:
        shown = ", ".join(zeros[:5]) + (", ..." if len(zeros) > 5 else "")
        ctx.note(
            f"{len(zeros)} reading{' is' if len(zeros) == 1 else 's are'} exactly zero "
            f"and marked measured ({shown}). "
            "Zero is a value and is fitted as one. Mark a reading that failed or was "
            "not taken as failed or missing, with an empty value."
        )

    return InspectResult(
        measurements=len(rows),
        measured=measured,
        missing=missing,
        failed=failed,
        censored=censored,
        samples=sorted({row.get("sample_id") for row in rows if row.get("sample_id")}),
        conditions=sorted(request.conditions),
        independent_unit=request.independent_unit,
        record=ctx.keep_json(record),
    )
