"""How a replica comparison appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def _flag(value) -> str:
    return "" if value is None else ("yes" if value else "no")


def render(result) -> str:
    rows = [
        [
            o.observable,
            o.unit,
            o.mean,
            o.replica_standard_error,
            f"{o.ci95_low:.4g} to {o.ci95_high:.4g}",
            "" if o.split_rhat is None else o.split_rhat,
            "" if o.effective_samples is None else o.effective_samples,
            o.precision_target,
            _flag(o.precision_passed),
            "" if o.reference_value is None else o.reference_value,
            "" if o.tolerance is None else o.tolerance,
            _flag(o.agrees),
        ]
        for o in result.observables
    ]
    first = result.first_sustained_checkpoint
    sustained = (
        "The precision was not sustained to the end of the run."
        if first is None
        else f"The precision was met from {first.frames_per_replica} frames "
        f"({first.simulated_ps_per_replica:g} ps) per replica to the end."
    )
    verdict = "met" if result.precision_passed else "was not met by at least one observable"
    agreement = (
        ""
        if result.agrees_with_reference is None
        else (
            " Every observable with a tolerance agreed with the reference value."
            if result.agrees_with_reference
            else " At least one observable with a tolerance did not agree with the reference value."
        )
    )
    return "".join(
        [
            f"<p>{len(result.replicas)} replicas ({esc(', '.join(result.replicas))}), "
            f"{result.frames_per_replica} frames each, sampled every {result.sampling_interval_ps:g} ps: "
            f"{result.simulated_ps_per_replica:g} ps per replica, "
            f"{result.total_simulated_ps:g} ps in all. The requested precision {verdict}."
            f"{esc(agreement)} {esc(sustained)}</p>",
            table(
                [
                    "Observable",
                    "Unit",
                    "Mean",
                    "SE",
                    "95 % interval",
                    "Split R-hat",
                    "Effective n",
                    "Precision target",
                    "Precise",
                    "Reference",
                    "Tolerance",
                    "Agrees",
                ],
                rows,
                numeric={2, 3, 5, 6, 7, 9, 10},
            ),
        ]
    )
