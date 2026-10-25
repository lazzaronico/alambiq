"""How an operating summary appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table


def render(result) -> str:
    parts = [
        table(
            [
                "Product",
                "Net formed (mol)",
                "Selected output (mol)",
                "Useful output (mol)",
                "Reaction-time productivity (mol/s)",
                "Process-time productivity (mol/s)",
            ],
            [
                [
                    result.product,
                    result.net_formed_mol,
                    result.selected_output_mol,
                    result.useful_output_mol,
                    result.reaction_time_productivity_mol_s,
                    result.process_time_productivity_mol_s,
                ]
            ],
            numeric={1, 2, 3, 4, 5},
        )
    ]
    if result.catalysts:
        parts.append(
            table(
                [
                    "Catalyst",
                    "Supplied active sites (mol)",
                    "Product per active site",
                    "Useful output per active site",
                ],
                [
                    [
                        c.id,
                        c.supplied_active_site_mol,
                        c.product_per_active_site,
                        c.useful_output_per_active_site,
                    ]
                    for c in result.catalysts
                ],
                numeric={1, 2, 3},
            )
        )
    for t in result.lifetimes:
        when = (
            f"first below between {number(t.first_below_interval_s[0])} s and "
            f"{number(t.first_below_interval_s[1])} s"
            if t.status == "observed_below_threshold"
            else f"{esc(t.status.replace('_', ' '))} through {number(t.horizon_s)} s"
        )
        parts.append(
            f'<p class="sub">Threshold {esc(t.id)} on {esc(t.catalyst)}: {when}</p>'
        )
    return "".join(parts)
