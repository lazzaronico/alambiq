"""How QM/MM dynamics appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    rows = [
        ["Production frames", result.production_frames],
        [
            "Equilibration / production steps",
            f"{result.equilibration_steps} / {result.production_steps}",
        ],
        ["Frame interval (fs)", result.frame_interval_fs],
        [
            "Electronic solutions / cache hits",
            f"{result.electronic_calls} / {result.cache_hits}",
        ],
        ["Mean temperature (K)", result.mean_temperature_kelvin],
        ["Mean potential (kJ/mol)", result.mean_potential_kj_mol],
        ["Mean unbiased potential (kJ/mol)", result.mean_unbiased_potential_kj_mol],
        ["Mean total energy (kJ/mol)", result.mean_total_energy_kj_mol],
        ["Total energy, last minus first (kJ/mol)", result.total_energy_drift_kj_mol],
        ["Largest constraint error (Å)", result.maximum_constraint_error_angstrom],
    ]
    if result.mean_coordinate_angstrom is not None:
        rows.append(["Mean umbrella coordinate (Å)", result.mean_coordinate_angstrom])
        if result.coordinate_sd_angstrom is not None:
            rows.append(["Coordinate SD (Å)", result.coordinate_sd_angstrom])
    return "".join(
        [
            f"<p>{result.seconds:.0f} s. Hamiltonian {esc(result.hamiltonian_sha256[:12])}.</p>",
            table(["Quantity", "Value"], rows, numeric={1}),
        ]
    )
