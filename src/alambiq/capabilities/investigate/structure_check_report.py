"""How a structure check appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table

SHOWN = 20


def _residue(chain: str, number: int, name: str, insertion: str = "") -> str:
    return f"{chain}:{name}{number}{insertion}"


def _section(title: str, headers: list[str], rows: list[list], numeric: set[int] | None = None) -> str:
    if not rows:
        return f"<h3>{esc(title)}</h3><p>None found.</p>"
    more = (
        f'<p class="sub">The first {SHOWN} of {len(rows)} are shown; the result lists all.</p>'
        if len(rows) > SHOWN
        else ""
    )
    return f"<h3>{esc(title)} ({len(rows)})</h3>" + table(headers, rows[:SHOWN], numeric=numeric) + more


def render(result) -> str:
    chains = ", ".join(f"{c.id} ({c.residues} residues, {c.first} to {c.last})" for c in result.chains)
    parts = [
        f"<p>Model {result.model_index} of {result.model_count}: {result.atoms} atoms, "
        f"{len(result.chains)} chain(s) of amino acids: {esc(chains) or 'none'}. "
        f"{len(result.ligands)} ligand(s), {len(result.metals)} metal ion(s), {result.waters} "
        f"water(s), {result.hydrogens} hydrogen atom(s).</p>",
        _section(
            "Chain breaks",
            ["Chain", "From", "To", "C–N distance (Å)", "Residue numbers skipped"],
            [
                [b.chain, f"{b.from_name}{b.from_number}", f"{b.to_name}{b.to_number}", b.c_n_distance_angstrom, b.numbering_gap]
                for b in result.breaks
            ],
            numeric={3, 4},
        ),
        _section(
            "Missing heavy atoms",
            ["Residue", "Atoms"],
            [[_residue(r.chain, r.number, r.name, r.insertion), r.detail] for r in result.missing_atoms],
        ),
        _section(
            "Alternate locations",
            ["Residue", "Locations"],
            [[_residue(r.chain, r.number, r.name, r.insertion), r.detail] for r in result.alternate_locations],
        ),
        _section(
            "Atoms of zero occupancy",
            ["Residue", "Atoms"],
            [[_residue(r.chain, r.number, r.name, r.insertion), r.detail] for r in result.zero_occupancy],
        ),
        _section(
            "Nonstandard amino acids",
            ["Residue", "Note"],
            [[_residue(r.chain, r.number, r.name, r.insertion), r.detail] for r in result.nonstandard_residues],
        ),
        _section(
            "Cis or twisted peptide bonds",
            ["From", "To", "Omega (°)", "Deviation (°)", "Cis", "Twisted"],
            [
                [
                    f"{p.chain}:{p.from_name}{p.from_number}",
                    f"{p.to_name}{p.to_number}",
                    p.omega_degrees,
                    p.deviation_degrees,
                    "yes" if p.cis else "",
                    "yes" if p.twisted else "",
                ]
                for p in result.peptide_bonds
            ],
            numeric={2, 3},
        ),
        _section(
            "Alpha carbons off the L arrangement",
            ["Residue", "Signed volume (Å³)", "Orientation"],
            [
                [_residue(a.chain, a.number, a.name, a.insertion), a.signed_volume_angstrom3, a.orientation]
                for a in result.alpha_centers
            ],
            numeric={1},
        ),
        _section(
            "Close contacts",
            ["First", "Second", "Distance (Å)"],
            [
                [
                    f"{_residue(c.first.chain, c.first.number, c.first.name, c.first.insertion)} {c.first.atom}",
                    f"{_residue(c.second.chain, c.second.number, c.second.name, c.second.insertion)} {c.second.atom}",
                    c.distance_angstrom,
                ]
                for c in result.close_contacts
            ],
            numeric={2},
        ),
        _section(
            "Ligands and metal ions",
            ["Residue", "Atoms", "Element"],
            [
                [_residue(h.chain, h.number, h.name, h.insertion), h.atoms, h.element or ""]
                for h in [*result.ligands, *result.metals]
            ],
            numeric={1},
        ),
        f'<p class="sub">{esc(result.gemmi_version)}; structure SHA-256 {esc(result.structure_sha256)}</p>',
    ]
    return "".join(parts)
