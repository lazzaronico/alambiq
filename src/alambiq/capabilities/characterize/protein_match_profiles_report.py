"""How profile matches appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    def rows(matches):
        return [
            [
                m.accession,
                m.name,
                m.description,
                m.clan_accession or "",
                f"{m.query_start}–{m.query_end}",
                f"{m.profile_start}–{m.profile_end} of {m.profile_length}",
                m.profile_coverage,
                m.evalue,
                m.bits,
            ]
            for m in matches
        ]

    headers = [
        "Profile",
        "Name",
        "Description",
        "Clan",
        "Protein positions",
        "Profile positions",
        "Profile covered",
        "E-value",
        "Bits",
    ]
    selected = [m for m in result.matches if m.status == "selected"]
    filtered = [m for m in result.matches if m.status == "filtered"]
    parts = [
        f"<p>{result.selected} selected and {result.filtered} filtered alignments "
        f"from {result.families_returned} profiles, for {esc(result.query_id)} "
        f"({result.query_length} residues), searched against "
        f"{result.profiles_searched:,} profiles in {esc(result.library)} with "
        f"{esc(result.engine_versions.get('mmseqs', 'MMseqs2'))}.</p>",
        table(headers, rows(selected), numeric={6, 7, 8}) if selected else "",
    ]
    if filtered:
        reasons = {
            m.accession + f" {m.query_start}–{m.query_end}": ", ".join(m.reasons)
            for m in filtered
        }
        parts.append(
            "<details><summary>Filtered alignments</summary>"
            + table(headers, rows(filtered), numeric={6, 7, 8})
            + '<p class="sub">'
            + esc("; ".join(f"{k}: {v}" for k, v in reasons.items()))
            + "</p></details>"
        )
    return "".join(parts)
