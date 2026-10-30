"""How a Boltz-2 prediction appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table
from alambiq.report.viewers import structure_file


def _value(number) -> str | float:
    return "" if number is None else number


def render(result) -> str:
    chains = table(
        ["Chain", "Residues", "Alignment"],
        [[c.id, c.residues, c.alignment] for c in result.chains],
        numeric={1},
    )
    samples = table(
        [
            "Sample",
            "Confidence",
            "pTM",
            "ipTM",
            "Ligand ipTM",
            "pLDDT",
            "ipLDDT",
            "PDE",
            "iPDE",
        ],
        [
            [
                s.index,
                s.confidence_score,
                _value(s.ptm),
                _value(s.iptm),
                _value(s.ligand_iptm),
                _value(s.complex_plddt),
                _value(s.complex_iplddt),
                _value(s.complex_pde),
                _value(s.complex_ipde),
            ]
            for s in result.samples
        ],
        numeric={0, 1, 2, 3, 4, 5, 6, 7, 8},
    )
    ligand = (
        ""
        if result.ligand is None
        else f"<p>Ligand {esc(result.ligand.id)}: {esc(result.ligand.canonical_smiles)} "
        f"(charge {result.ligand.formal_charge:+d}).</p>"
    )
    affinity = ""
    if result.affinity is not None:
        estimates = [("Ensemble", result.affinity)] + [
            (f"Model {i}", m) for i, m in enumerate(result.affinity.models, start=1)
        ]
        affinity = table(
            ["Affinity", "log10 IC50 (µM)", "IC50 (µM)", "Binder probability"],
            [
                [
                    name,
                    round(e.log10_ic50_um, 3),
                    float(f"{10 ** e.log10_ic50_um:.3g}"),
                    round(e.binder_probability, 3),
                ]
                for name, e in estimates
            ],
            numeric={1, 2, 3},
        )
    best = max(result.samples, key=lambda s: s.confidence_score, default=None)
    view = (
        structure_file(
            best.structure,
            "cif",
            caption=f"Sample {best.index}, the highest confidence score of "
            f"{len(result.samples)} sample(s).",
        )
        if best is not None
        else ""
    )
    return "".join(
        [
            f"<p>{esc(result.model)} on the {esc(result.accelerator)}, seed {result.seed}, "
            f"{result.sampling_steps} diffusion steps, {len(result.samples)} sample(s); "
            f"{result.seconds:.0f} s.</p>",
            chains,
            ligand,
            affinity,
            samples,
            view,
        ]
    )
