"""How a dynamics run appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.4g}"


def render(result) -> str:
    rows = []
    for phase in result.phases:
        detail = phase.detail
        if phase.name == "readiness":
            note = f"{len(detail['windows'])} window(s), ready: {detail['ready']}"
        elif phase.name == "minimization":
            note = (
                f"potential {detail['potential_before_kj_mol']:.1f} → "
                f"{detail['potential_after_kj_mol']:.1f} kJ/mol"
            )
        else:
            stats = detail.get("statistics", {})
            temperature = stats.get("temperature_kelvin", {}).get("mean")
            note = (
                "" if temperature is None else f"mean temperature {temperature:.1f} K"
            )
        rows.append([phase.name, "" if phase.steps is None else phase.steps, note])
    energy = (
        ""
        if result.total_energy_min_kj_mol is None
        else f" Total energy over production: {result.total_energy_min_kj_mol:.5g} to "
        f"{result.total_energy_max_kj_mol:.5g} kJ/mol."
    )
    production = (
        f"Production wrote {result.frames} frames; mean temperature "
        f"{_fmt(result.mean_temperature_kelvin)} K, mean volume {_fmt(result.mean_volume_nm3)} nm³, "
        f"mean density {_fmt(result.mean_density_g_ml)} g/mL.{energy}"
        if result.production_run
        else "No production was run."
    )
    return "".join(
        [
            f"<p>{esc(result.ensemble)} ensemble, {result.particles} particles, on "
            f"{esc(result.platform)} (OpenMM {esc(result.versions.get('openmm', ''))}). "
            f"Ready for production: {result.ready_for_production}. {esc(production)}</p>",
            table(["Phase", "Steps", "Summary"], rows, numeric={1}),
        ]
    )
