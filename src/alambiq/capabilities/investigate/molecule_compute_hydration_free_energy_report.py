"""How a hydration free energy appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table


def render(result) -> str:
    spread = (
        f" ± {number(result.uncertainty_kcal_mol, 3)}"
        if result.uncertainty_kcal_mol is not None
        else ""
    )
    parts = [
        f"<p>{esc(result.canonical_smiles)}: hydration free energy "
        f"{number(result.hydration_free_energy_kcal_mol, 3)}{spread} kcal/mol over "
        f"{len(result.repeats)} repeats, {number(result.production_ns)} ns of solvent "
        f"production on {esc(result.platform)} at {number(result.temperature_kelvin)} K. "
        f"{esc(result.force_field)}; {esc(result.charge_model)}.</p>"
    ]
    rows = []
    for index, r in enumerate(result.repeats, start=1):
        rows.append(
            [
                index,
                r.hydration_free_energy_kcal_mol,
                r.mbar_error_kcal_mol,
                r.solvent.decoupling_kcal_mol,
                r.vacuum.decoupling_kcal_mol,
                min(
                    r.solvent.minimum_adjacent_overlap,
                    r.vacuum.minimum_adjacent_overlap,
                ),
                min(
                    r.solvent.minimum_effective_samples,
                    r.vacuum.minimum_effective_samples,
                ),
            ]
        )
    parts.append(
        table(
            [
                "Repeat",
                "Hydration (kcal/mol)",
                "MBAR error",
                "Solvent leg",
                "Vacuum leg",
                "Minimum overlap",
                "Minimum effective samples",
            ],
            rows,
            numeric=set(range(7)),
        )
    )
    return "".join(parts)
