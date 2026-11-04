"""Combine the folded and reference legs of a substitution into the change in the free energy of folding.

Returns the change in the folding free energy of a substitution, ΔΔG(fold), in kcal/mol, with its
uncertainty and the diagnostics of each leg, and a pass, fail or unassessed verdict for each
requested check.

The thermodynamic cycle runs the substitution in the folded protein and in the reference tripeptide:

    ΔΔG(fold) = ΔG(folded leg) - ΔG(reference leg)

where each ΔG is the mean over the replicas of the leg, from the wild-type to the mutant
state. A positive value means that the substitution destabilizes the folded state against the
reference (the folding free energy rises); a negative value means that it stabilizes it.

Every value comes with:

- the sample standard deviation of the replica estimates of each leg;
- the error that MBAR reports within each replica;
- the smallest overlap between neighboring lambda windows over the replicas;
- the smallest effective sample size per window over the replicas;
- the difference between the forward and reverse estimates at half of the data.

The uncertainty of ΔΔG(fold) combines the standard errors of the two means:
`replica_standard_error` from the spread of the replicas (when both legs have 2 or more
replicas) and `mbar_error` from the MBAR errors. The checks in the request turn each diagnostic
into a verdict.

Limits.

- The spread between replicas measures the reproducibility of the sampling from independent
  runs. The force field, the template conformation of the new side chain, the choice of the
  reference state and the protonation are not part of it.
- The MBAR error assumes uncorrelated samples and understates the error when they are
  correlated.
- Overlap, effective sample size and the forward-reverse difference indicate whether the
  lambda path is converged.
- The result describes the folded structure that was supplied; it holds for a substitution that
  leaves that fold in place.
- The capability refuses legs of different substitutions or temperatures, two legs of the same
  kind, and a leg with no replicas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from alambiq.capabilities.investigate.simulation_analyze_stability_change_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class StabilityChecks:
    """Thresholds that turn diagnostics into verdicts; each threshold is optional.

    minimum_adjacent_overlap:
        The smallest permitted overlap between neighboring lambda windows over the replicas
        of each leg.
    maximum_replica_sd_kcal_mol:
        The largest permitted sample standard deviation of the replica estimates of each leg.
    maximum_forward_reverse_gap_kcal_mol:
        The largest permitted difference between the forward and reverse estimates at half of
        the data, over the replicas of each leg.
    maximum_replica_standard_error_kcal_mol:
        The largest permitted standard error of ΔΔG(fold) from the replica spread.
    """

    minimum_adjacent_overlap: float | None = None
    maximum_replica_sd_kcal_mol: float | None = None
    maximum_forward_reverse_gap_kcal_mol: float | None = None
    maximum_replica_standard_error_kcal_mol: float | None = None


@dataclass(frozen=True, slots=True)
class StabilityRequest:
    """The two legs and the checks.

    folded, reference:
        The paths of the leg records (`leg_record`) of `simulation.run_mutation_free_energy` for
        the folded leg and the reference leg of one substitution.
    checks:
        The thresholds of the verdicts; without it, no verdict is made.
    """

    folded: str
    reference: str
    checks: StabilityChecks | None = None


@dataclass(frozen=True, slots=True)
class LegSummary:
    """One leg of the cycle."""

    leg: str
    replicas: int
    mean_delta_g_kcal_mol: float
    replica_sd_kcal_mol: float | None
    mbar_error_rms_kcal_mol: float
    minimum_adjacent_overlap: float
    minimum_effective_samples: float
    forward_reverse_gap_kcal_mol: float | None


@dataclass(frozen=True, slots=True)
class Verdict:
    """One diagnostic against its limit."""

    diagnostic: str
    subject: str
    value: float | None
    limit: float
    status: str
    """`pass`, `fail`, or `unassessed` when the diagnostic is not available."""


@dataclass(frozen=True, slots=True)
class StabilityResult:
    """The change in the folding free energy and its diagnostics."""

    mutation: str
    temperature_kelvin: float
    ddg_fold_kcal_mol: float
    replica_standard_error_kcal_mol: float | None
    """None when either leg has a single replica."""
    mbar_error_kcal_mol: float
    convention: str
    legs: list[LegSummary]
    verdicts: list[Verdict]
    status: str
    """`fail` when any verdict fails; otherwise `unassessed` when there is no verdict or one is unavailable; otherwise `pass`."""
    record: str


def _summary(leg: str, s: dict) -> LegSummary:
    return LegSummary(
        leg=leg,
        replicas=s["replicas"],
        mean_delta_g_kcal_mol=s["mean"],
        replica_sd_kcal_mol=s["sd"],
        mbar_error_rms_kcal_mol=s["mbar_rms"],
        minimum_adjacent_overlap=s["minimum_overlap"],
        minimum_effective_samples=s["minimum_effective_samples"],
        forward_reverse_gap_kcal_mol=s["forward_reverse_gap"],
    )


def _verdict(name: str, subject: str, value, limit, *, minimum: bool) -> Verdict:
    if value is None:
        status = "unassessed"
    else:
        status = "pass" if (value >= limit if minimum else value <= limit) else "fail"
    return Verdict(diagnostic=name, subject=subject, value=value, limit=limit, status=status)


@capability(
    "simulation.analyze_stability_change",
    version="1.0",
    family="investigate",
    summary="Combine two legs of a substitution into the change in folding free energy",
    render=render,
)
def analyze_stability_change(request: StabilityRequest, ctx: StepContext) -> StabilityResult:
    """Check that the legs belong together, combine them, and apply the checks."""
    from alambiq.capabilities.investigate.science.mutation import leg_statistics

    records = {"folded": ctx.read_json(request.folded), "reference": ctx.read_json(request.reference)}
    for expected, record in records.items():
        if record.get("leg") != expected:
            raise ValueError(
                f"The {expected} record names the leg {record.get('leg')!r}; the {expected} leg is required there"
            )
        if not record.get("replicas"):
            raise ValueError(f"The {expected} leg has no replicas; at least 1 is required")
    f, r = records["folded"], records["reference"]
    if (f["wild_type_residue"], f["mutant_residue"]) != (r["wild_type_residue"], r["mutant_residue"]):
        raise ValueError(
            f"The legs describe different substitutions ({f['wild_type_residue']}>{f['mutant_residue']} and "
            f"{r['wild_type_residue']}>{r['mutant_residue']}); two legs of one substitution are required"
        )
    if f["temperature_kelvin"] != r["temperature_kelvin"]:
        raise ValueError(
            f"The legs differ in temperature ({f['temperature_kelvin']} K and {r['temperature_kelvin']} K); "
            "one temperature is required"
        )
    sf, sr = leg_statistics(f), leg_statistics(r)
    ddg = sf["mean"] - sr["mean"]
    mbar = math.sqrt(sf["mbar_rms"] ** 2 / sf["replicas"] + sr["mbar_rms"] ** 2 / sr["replicas"])
    standard_error = None
    if sf["sd"] is not None and sr["sd"] is not None:
        standard_error = math.sqrt(sf["sd"] ** 2 / sf["replicas"] + sr["sd"] ** 2 / sr["replicas"])
    verdicts: list[Verdict] = []
    checks = request.checks
    if checks is not None:
        for name, s in (("folded", sf), ("reference", sr)):
            if checks.minimum_adjacent_overlap is not None:
                verdicts.append(
                    _verdict("minimum_adjacent_overlap", name, s["minimum_overlap"], checks.minimum_adjacent_overlap, minimum=True)
                )
            if checks.maximum_replica_sd_kcal_mol is not None:
                verdicts.append(
                    _verdict("replica_sd_kcal_mol", name, s["sd"], checks.maximum_replica_sd_kcal_mol, minimum=False)
                )
            if checks.maximum_forward_reverse_gap_kcal_mol is not None:
                verdicts.append(
                    _verdict("forward_reverse_gap_kcal_mol", name, s["forward_reverse_gap"], checks.maximum_forward_reverse_gap_kcal_mol, minimum=False)
                )
        if checks.maximum_replica_standard_error_kcal_mol is not None:
            verdicts.append(
                _verdict(
                    "replica_standard_error_kcal_mol", "cycle", standard_error,
                    checks.maximum_replica_standard_error_kcal_mol, minimum=False,
                )
            )
    if any(v.status == "fail" for v in verdicts):
        status = "fail"
    elif not verdicts or any(v.status == "unassessed" for v in verdicts):
        status = "unassessed"
    else:
        status = "pass"
    if sf["replicas"] < 3 or sr["replicas"] < 3:
        ctx.note(
            f"The legs have {sf['replicas']} and {sr['replicas']} replicas; 3 or more give a replica spread that is a usable estimate."
        )
    mutation = f"{f['wild_type_residue']}>{f['mutant_residue']}"
    data = {
        "mutation": mutation,
        "temperature_kelvin": f["temperature_kelvin"],
        "ddg_fold_kcal_mol": ddg,
        "folded": sf,
        "reference": sr,
        "methods": {"folded": f.get("method"), "reference": r.get("method")},
    }
    return StabilityResult(
        mutation=mutation,
        temperature_kelvin=f["temperature_kelvin"],
        ddg_fold_kcal_mol=ddg,
        replica_standard_error_kcal_mol=standard_error,
        mbar_error_kcal_mol=mbar,
        convention="ΔΔG(fold) = ΔG(folded leg) - ΔG(reference leg); positive destabilizes the folded state",
        legs=[_summary("folded", sf), _summary("reference", sr)],
        verdicts=verdicts,
        status=status,
        record=ctx.keep_json(data),
    )
