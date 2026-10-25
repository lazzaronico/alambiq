"""How a defined operation appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table


def render(result) -> str:
    rows = [
        [
            c.channel,
            c.catalyst,
            c.species,
            c.capacity_M_s,
            c.scale_M,
            c.vmax_substrate_equivalent_M_s,
            f"{number(c.fit_domain_M[0])} to {number(c.fit_domain_M[1])}",
            "extrapolation" if c.extrapolates else "within the fit",
        ]
        for c in result.channels
    ]
    kept = ", ".join(esc(c) for c in result.unchanged_channels) or "none"
    extra = []
    dependent = [
        [
            c.channel,
            "—"
            if c.activation_energy_J_mol is None
            else c.activation_energy_J_mol / 1000,
            "—"
            if c.ph_pKa is None
            else f"{number(c.ph_pKa[0])} and {number(c.ph_pKa[1])}",
        ]
        for c in result.channels
        if c.activation_energy_J_mol is not None or c.ph_pKa is not None
    ]
    if dependent:
        extra.append(
            table(
                ["Channel", "Activation energy (kJ/mol)", "pH profile pKa values"],
                dependent,
                numeric={1},
            )
        )
    if result.equilibria:
        extra.append(
            table(
                [
                    "Forward",
                    "Reverse",
                    "Equilibrium constant",
                    "Reverse capacity (M/s)",
                    "Equilibrium constant of the rate law",
                ],
                [
                    [
                        e.forward,
                        e.reverse,
                        e.equilibrium_constant,
                        e.reverse_capacity_M_s,
                        e.implied_equilibrium_constant,
                    ]
                    for e in result.equilibria
                ],
                numeric={2, 3, 4},
            )
        )
    if result.inactivation:
        extra.append(
            table(
                [
                    "Catalyst",
                    "Deactivation (1/s)",
                    "Half-life (s)",
                    "At (K)",
                    "Activation energy (kJ/mol)",
                ],
                [
                    [
                        i.catalyst,
                        i.deactivation_per_s,
                        i.half_life_s,
                        i.decay_temperature_K,
                        "—"
                        if i.activation_energy_J_mol is None
                        else i.activation_energy_J_mol / 1000,
                    ]
                    for i in result.inactivation
                ],
                numeric={1, 2, 3},
            )
        )
    return "".join(
        [
            table(
                [
                    "Channel",
                    "Catalyst",
                    "Substrate",
                    "Capacity (M/s)",
                    "Saturation scale (M)",
                    "Assay Vmax, substrate basis (M/s)",
                    "Fit range (M)",
                    "Highest substrate concentration",
                ],
                rows,
                numeric={3, 4, 5},
            ),
            *extra,
            f'<p class="sub">Operation {esc(result.operation_id)}. Channels unchanged '
            f"from the template: {kept}.</p>",
        ]
    )
