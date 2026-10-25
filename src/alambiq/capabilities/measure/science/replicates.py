"""Summarizing independent replicates of one measured quantity.

Each independent replicate contributes one mean, however many technical
readings it was averaged from, so the interval reflects variation between
independent units rather than repeated reads of the same one. The result
reports the mean, the sample standard deviation of the replicate means, and a
two-sided 95% interval for the mean using Student's t with n - 1 degrees of
freedom.

An estimate is marked converged only when there are enough independent
replicates and the interval is no wider than the stated maximum; otherwise the
reasons are given. Convergence here means the precision target was met.

Limits: the t quantiles come from a published table to three decimals for 1 to
30 degrees of freedom. Above 30 the normal value 1.96 is used, which is slightly
narrow for small samples in that range. Normality of replicate means is
assumed; with 3 replicates that assumption cannot be checked.

This module imports the standard library only.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True, slots=True)
class IndependentReplicate:
    """One independent unit's mean and how many readings formed it."""

    independent_replicate: int
    mean: float
    technical_measurement_count: int


@dataclass(frozen=True, slots=True)
class ReplicateEstimate:
    """The summary for one subject and one assay."""

    subject_id: str
    assay_id: str
    unit: str
    independent_replicate_count: int
    technical_measurement_count: int
    independent_replicates: tuple[IndependentReplicate, ...]
    mean: float | None
    standard_deviation: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    confidence_interval_width: float | None
    converged: bool
    reason_codes: tuple[str, ...]


def estimate_replicates(
    subject_id: str,
    assay_id: str,
    unit: str,
    independent_values: list[tuple[int, float, int]],
    minimum_replicates: int,
    maximum_ci_width: float,
) -> ReplicateEstimate:
    """Summarize `(replicate, mean, technical count)` triples for one quantity."""
    ordered = sorted(independent_values, key=lambda item: item[0])
    means = tuple(value for _, value, _ in ordered)
    technical = sum(count for _, _, count in ordered)
    replicates = tuple(
        IndependentReplicate(
            independent_replicate=replicate,
            mean=round(value, 6),
            technical_measurement_count=count,
        )
        for replicate, value, count in ordered
    )
    reasons: list[str] = []
    mean = standard_deviation = lower = upper = width = None
    if means:
        mean = statistics.mean(means)
    if len(means) >= 2:
        standard_deviation = statistics.stdev(means)
        critical = T_CRITICAL_975.get(len(means) - 1, 1.96)
        half_width = critical * standard_deviation / math.sqrt(len(means))
        lower, upper = mean - half_width, mean + half_width
        width = upper - lower
    if len(means) < minimum_replicates:
        reasons.append("insufficient-independent-replicates")
    if width is None:
        reasons.append("confidence-interval-unavailable")
    elif width > maximum_ci_width:
        reasons.append("confidence-interval-too-wide")

    def rounded(value: float | None) -> float | None:
        return round(value, 6) if value is not None else None

    return ReplicateEstimate(
        subject_id=subject_id,
        assay_id=assay_id,
        unit=unit,
        independent_replicate_count=len(means),
        technical_measurement_count=technical,
        independent_replicates=replicates,
        mean=rounded(mean),
        standard_deviation=rounded(standard_deviation),
        confidence_interval_lower=rounded(lower),
        confidence_interval_upper=rounded(upper),
        confidence_interval_width=rounded(width),
        converged=not reasons,
        reason_codes=tuple(reasons) if reasons else ("replicates-and-precision-pass",),
    )
