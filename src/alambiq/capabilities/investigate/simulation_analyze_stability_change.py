"""Combine the folded and reference legs of a substitution into the change in the free energy of folding.

Returns the change in the folding free energy of a substitution, ΔΔG(fold), in kcal/mol, with its
uncertainty and the diagnostics of each leg, and a pass, fail or unassessed verdict for each
requested check.

The thermodynamic cycle runs the substitution in the folded protein and in the reference tripeptide:

    ΔΔG(fold) = ΔG(folded leg) - ΔG(reference leg)

where each ΔG is the mean over the preparations of the leg, from the wild-type to the mutant
state. A positive value means that the substitution destabilizes the folded state against the
reference (the folding free energy rises); a negative value means that it stabilizes it.

Each leg is given as a list of records, one per independent preparation (a run of
`structure.prepare_mutation_inputs` with its own seed, followed by `simulation.run_mutation_free_energy`
of its leg). A preparation can hold several replicas. The mean of a preparation weights its replicas
equally, and the mean of a leg weights its preparations equally.

Every value comes with:

- the sample standard deviation of the preparation means of each leg, and its combination into the
  standard error of ΔΔG(fold) (`preparation_standard_error`);
- the largest sample standard deviation of the replicas within one preparation;
- the error that MBAR reports within each replica, combined over the replicas;
- the smallest overlap between neighboring lambda windows over the replicas;
- the smallest effective sample size per window over the replicas;
- the difference between the forward and reverse estimates at half of the data.

The checks in the request turn each diagnostic into a verdict.

Limits.

- The replicas of one preparation start from the same coordinates, so their spread measures the
  sampling from that start. Independent preparations differ in the placement of water and ions and in
  the starting conformation of the side chain, and their spread is larger: the convergence record of
  this method area shows a spread across preparations that is about 10 times the MBAR error.
  `preparation_standard_error` needs 2 or more preparations in each leg.
- The MBAR error assumes uncorrelated samples and understates the error when they are
  correlated.
- Overlap, effective sample size and the forward-reverse difference indicate whether the
  lambda path is converged.
- The force field, the template conformation of the new side chain, the choice of the reference
  state and the protonation are not part of any reported spread.
- The result describes the folded structure that was supplied; it holds for a substitution that
  leaves that fold in place.
- The capability refuses legs of different substitutions or temperatures, records of the wrong leg,
  and a record with no replicas.
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
        The largest permitted sample standard deviation of the replica estimates within one
        preparation, for each leg.
    maximum_forward_reverse_gap_kcal_mol:
        The largest permitted difference between the forward and reverse estimates at half of
        the data, over the replicas of each leg.
    maximum_preparation_standard_error_kcal_mol:
        The largest permitted standard error of ΔΔG(fold) from the spread across preparations.
    """

    minimum_adjacent_overlap: float | None = None
    maximum_replica_sd_kcal_mol: float | None = None
    maximum_forward_reverse_gap_kcal_mol: float | None = None
    maximum_preparation_standard_error_kcal_mol: float | None = None


@dataclass(frozen=True, slots=True)
class StabilityRequest:
    """The two legs and the checks.

    folded, reference:
        The paths of the leg records (`leg_record`) of `simulation.run_mutation_free_energy` for
        the folded legs and the reference legs of one substitution, one record per independent
        preparation, 1 to 50 of each.
    checks:
        The thresholds of the verdicts; without it, no verdict is made.
    """

    folded: list[str]
    reference: list[str]
    checks: StabilityChecks | None = None


@dataclass(frozen=True, slots=True)
class LegSummary:
    """One leg of the cycle."""

    leg: str
    preparations: int
    replicas: int
    """The replicas of all preparations."""
    mean_delta_g_kcal_mol: float
    preparation_sd_kcal_mol: float | None
    """None with a single preparation."""
    replica_sd_kcal_mol: float | None
    """The largest sample standard deviation of the replicas within one preparation; None when no preparation has 2 replicas."""
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
    preparation_standard_error_kcal_mol: float | None
    """From the spread across preparations; None when either leg has a single preparation."""
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
        preparations=s["preparations"],
        replicas=s["replicas"],
        mean_delta_g_kcal_mol=s["mean"],
        preparation_sd_kcal_mol=s["preparation_sd"],
        replica_sd_kcal_mol=s["replica_sd"],
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
    return Verdict(
        diagnostic=name, subject=subject, value=value, limit=limit, status=status
    )


@capability(
    "simulation.analyze_stability_change",
    version="1.0",
    family="investigate",
    summary="Combine two legs of a substitution into the change in folding free energy",
    render=render,
)
def analyze_stability_change(
    request: StabilityRequest, ctx: StepContext
) -> StabilityResult:
    """Check that the legs belong together, combine them, and apply the checks."""
    from alambiq.capabilities.investigate.science.mutation import combine_preparations

    records = {
        "folded": [ctx.read_json(p) for p in request.folded],
        "reference": [ctx.read_json(p) for p in request.reference],
    }
    for expected, group in records.items():
        if not 1 <= len(group) <= 50:
            raise ValueError(
                f"The {expected} list holds {len(group)} records; 1 to 50 are required"
            )
        for record in group:
            if record.get("leg") != expected:
                raise ValueError(
                    f"A {expected} record names the leg {record.get('leg')!r}; the {expected} leg is required there"
                )
            if not record.get("replicas"):
                raise ValueError(
                    f"A {expected} record has no replicas; at least 1 is required"
                )
    everything = [*records["folded"], *records["reference"]]
    first = everything[0]
    for record in everything:
        if (record["wild_type_residue"], record["mutant_residue"]) != (
            first["wild_type_residue"],
            first["mutant_residue"],
        ):
            raise ValueError(
                f"The legs describe different substitutions ({first['wild_type_residue']}>{first['mutant_residue']} and "
                f"{record['wild_type_residue']}>{record['mutant_residue']}); records of one substitution are required"
            )
        if record["temperature_kelvin"] != first["temperature_kelvin"]:
            raise ValueError(
                f"The legs differ in temperature ({first['temperature_kelvin']} K and {record['temperature_kelvin']} K); "
                "one temperature is required"
            )
    sf, sr = (
        combine_preparations(records["folded"]),
        combine_preparations(records["reference"]),
    )
    ddg = sf["mean"] - sr["mean"]
    mbar = math.sqrt(sf["mbar_error"] ** 2 + sr["mbar_error"] ** 2)
    standard_error = None
    if sf["preparation_sd"] is not None and sr["preparation_sd"] is not None:
        standard_error = math.sqrt(
            sf["preparation_sd"] ** 2 / sf["preparations"]
            + sr["preparation_sd"] ** 2 / sr["preparations"]
        )
    verdicts: list[Verdict] = []
    checks = request.checks
    if checks is not None:
        for name, s in (("folded", sf), ("reference", sr)):
            if checks.minimum_adjacent_overlap is not None:
                verdicts.append(
                    _verdict(
                        "minimum_adjacent_overlap",
                        name,
                        s["minimum_overlap"],
                        checks.minimum_adjacent_overlap,
                        minimum=True,
                    )
                )
            if checks.maximum_replica_sd_kcal_mol is not None:
                verdicts.append(
                    _verdict(
                        "replica_sd_kcal_mol",
                        name,
                        s["replica_sd"],
                        checks.maximum_replica_sd_kcal_mol,
                        minimum=False,
                    )
                )
            if checks.maximum_forward_reverse_gap_kcal_mol is not None:
                verdicts.append(
                    _verdict(
                        "forward_reverse_gap_kcal_mol",
                        name,
                        s["forward_reverse_gap"],
                        checks.maximum_forward_reverse_gap_kcal_mol,
                        minimum=False,
                    )
                )
        if checks.maximum_preparation_standard_error_kcal_mol is not None:
            verdicts.append(
                _verdict(
                    "preparation_standard_error_kcal_mol",
                    "cycle",
                    standard_error,
                    checks.maximum_preparation_standard_error_kcal_mol,
                    minimum=False,
                )
            )
    if any(v.status == "fail" for v in verdicts):
        status = "fail"
    elif not verdicts or any(v.status == "unassessed" for v in verdicts):
        status = "unassessed"
    else:
        status = "pass"
    if sf["preparations"] < 3 or sr["preparations"] < 3:
        ctx.note(
            f"The legs have {sf['preparations']} and {sr['preparations']} independent preparation(s). The replicas of one "
            "preparation share their starting coordinates, and the spread across independent preparations is larger; "
            "3 or more preparations give a usable estimate of it."
        )
    mutation = f"{first['wild_type_residue']}>{first['mutant_residue']}"
    data = {
        "mutation": mutation,
        "temperature_kelvin": first["temperature_kelvin"],
        "ddg_fold_kcal_mol": ddg,
        "folded": sf,
        "reference": sr,
        "methods": {
            "folded": [r.get("method") for r in records["folded"]],
            "reference": [r.get("method") for r in records["reference"]],
        },
    }
    return StabilityResult(
        mutation=mutation,
        temperature_kelvin=first["temperature_kelvin"],
        ddg_fold_kcal_mol=ddg,
        preparation_standard_error_kcal_mol=standard_error,
        mbar_error_kcal_mol=mbar,
        convention="ΔΔG(fold) = ΔG(folded leg) - ΔG(reference leg); positive destabilizes the folded state",
        legs=[_summary("folded", sf), _summary("reference", sr)],
        verdicts=verdicts,
        status=status,
        record=ctx.keep_json(data),
    )
