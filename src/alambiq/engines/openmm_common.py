"""Descriptive diagnostics shared by the OpenMM engines.

Runs inside the openmm environment. Imports the standard library at the top;
numpy is imported inside the functions that need it.
"""

import csv


def cell_diagnostics(state, unit, cutoff_nm=None):
    import numpy as np

    box = np.asarray(state.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    if box.shape != (3, 3) or not np.isfinite(box).all() or np.linalg.det(box) <= 0:
        raise ValueError("The periodic cell must have finite box vectors and a positive volume")
    # OpenMM's reduced triclinic representation uses these widths for its cutoff check.
    width = float(np.diag(box).min())
    return {
        "vectors_nm": box.tolist(),
        "volume_nm3": float(np.linalg.det(box)),
        "minimum_reduced_cell_width_nm": width,
        "cutoff_nm": cutoff_nm,
        "twice_cutoff_margin_nm": None if cutoff_nm is None else width - 2 * cutoff_nm,
        "scope": "Reduced cell width and its margin over twice the cutoff (`twice_cutoff_margin_nm`)",
    }


def force_diagnostics(state, unit, topology=None):
    import numpy as np

    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    if not np.isfinite(forces).all():
        raise ValueError("The prepared state contains non-finite forces")
    result = {
        "rms_component_kj_mol_nm": float(np.sqrt(np.mean(forces**2))),
        "maximum_atom_kj_mol_nm": float(np.linalg.norm(forces, axis=1).max()),
        "scope": "Raw force-field forces; constraint forces are excluded",
    }
    if topology is not None:
        atoms = list(topology.atoms())
        if len(atoms) != len(forces):
            raise ValueError(
                f"The topology has {len(atoms)} atoms and the state has {len(forces)} force rows; "
                "the counts must be equal"
            )
        index = int(np.argmax(np.linalg.norm(forces, axis=1)))
        atom = atoms[index]
        result["maximum_force_atom"] = {
            "atom_one_based": index + 1,
            "atom_name": atom.name,
            "residue_one_based": atom.residue.index + 1,
            "residue_name": atom.residue.name,
            "chain_id": atom.residue.chain.id,
            "residue_id": atom.residue.id,
        }
    return result


def protein_geometry_diagnostics(topology, positions_nm, system):
    """Inspect explicit bonded peptides and standard L-alpha-amino-acid geometry.

    The physical System supplies bonds: a reloaded, flattened PDB can invent
    peptide links across original chain ends. The observations are recorded
    in the result and leave coordinates and execution unchanged. Cis peptides are reported
    separately from twists.
    """
    import numpy as np

    atoms = list(topology.atoms())
    xyz = np.asarray(positions_nm, dtype=float)
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        raise ValueError(
            f"The coordinates have shape {tuple(xyz.shape)}, or contain non-finite values; "
            f"finite coordinates for {len(atoms)} atoms are required"
        )
    if system.getNumParticles() != len(atoms):
        raise ValueError(
            f"The system has {system.getNumParticles()} particles and the topology has "
            f"{len(atoms)} atoms; the counts must be equal"
        )
    bonds = set()
    for force in system.getForces():
        if type(force).__name__ == "HarmonicBondForce":
            for i in range(force.getNumBonds()):
                a, b = force.getBondParameters(i)[:2]
                bonds.add(tuple(sorted((a, b))))
    for i in range(system.getNumConstraints()):
        a, b = system.getConstraintParameters(i)[:2]
        bonds.add(tuple(sorted((a, b))))
    amino = set(
        "ALA ARG ASN ASP ASH CYS CYM CYX GLN GLU GLH HIS HID HIE HIP ILE LEU LYS LYN MET PHE PRO SER THR TRP TYR VAL".split()
    )
    named = {r.index: {a.name: a for a in r.atoms()} for r in topology.residues()}
    peptides = []
    for ai, bi in sorted(bonds):
        a, b = atoms[ai], atoms[bi]
        if a.name == "N" and b.name == "C":
            a, b = b, a
        if a.name != "C" or b.name != "N" or a.residue == b.residue:
            continue
        if a.residue.name not in amino | {"GLY"} or b.residue.name not in amino | {
            "GLY"
        }:
            continue
        if any("CA" not in named[r.index] for r in (a.residue, b.residue)):
            continue
        indices = [
            named[a.residue.index]["CA"].index,
            a.index,
            b.index,
            named[b.residue.index]["CA"].index,
        ]
        p, q, r, s = xyz[indices]
        axis = r - q
        length = np.linalg.norm(axis)
        angle = deviation = None
        if length > 0:
            axis = axis / length
            u, v = p - q, s - r
            u, v = u - np.dot(u, axis) * axis, v - np.dot(v, axis) * axis
            if np.linalg.norm(u) > 0 and np.linalg.norm(v) > 0:
                angle = float(
                    np.degrees(np.arctan2(np.dot(np.cross(axis, u), v), np.dot(u, v)))
                )
                deviation = min(abs(angle), 180 - abs(angle))
        peptides.append(
            {
                "residues_one_based": [a.residue.index + 1, b.residue.index + 1],
                "atoms_one_based": [i + 1 for i in indices],
                "omega_degrees": angle,
                "deviation_from_cis_or_trans_degrees": deviation,
            }
        )
    centers, unassessed = [], []
    for residue in topology.residues():
        if residue.name not in amino:
            continue
        row = named[residue.index]
        if not {"N", "CA", "C", "CB"} <= row.keys():
            unassessed.append(residue.index + 1)
            continue
        ca = xyz[row["CA"].index]
        n, c, cb = (xyz[row[name].index] - ca for name in ("N", "C", "CB"))
        triple = float(np.dot(n, np.cross(c, cb)))
        centers.append(
            {
                "residue_one_based": residue.index + 1,
                "residue_name": residue.name,
                "signed_N_C_CB_volume_nm3": triple,
                "orientation": "L"
                if triple > 0
                else "inverted"
                if triple < 0
                else "planar",
            }
        )
    deviations = [
        p["deviation_from_cis_or_trans_degrees"]
        for p in peptides
        if p["deviation_from_cis_or_trans_degrees"] is not None
    ]
    return {
        "scope": "Descriptive bonded peptide planarity and standard L-amino-acid alpha chirality of the supplied coordinates",
        "bond_source": "Physical HarmonicBondForce and constraints; inferred PDB bonds are excluded",
        "chirality_convention": "dot(N-CA, cross(C-CA, CB-CA)); positive for standard L alpha geometry, including cysteine; glycine has no alpha stereocenter",
        "peptides": peptides,
        "alpha_centers": centers,
        "maximum_peptide_planarity_deviation_degrees": max(deviations, default=None),
        "inverted_alpha_residues_one_based": [
            c["residue_one_based"] for c in centers if c["orientation"] == "inverted"
        ],
        "planar_alpha_residues_one_based": [
            c["residue_one_based"] for c in centers if c["orientation"] == "planar"
        ],
        "unassessed_alpha_residues_one_based": unassessed,
    }


def thermodynamic_diagnostics(path, mass_dalton):
    """Summarize the recorded rows."""
    import numpy as np

    with path.open() as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 2:
        return {"sample_count": 0, "assessment": "No samples are recorded"}
    values = np.asarray([[float(v) for v in row] for row in rows[1:]])
    if (
        values.shape[1] != (7 if mass_dalton is not None else 6)
        or not np.isfinite(values).all()
        or (mass_dalton is not None and np.any(values[:, 6] <= 0))
    ):
        raise ValueError(
            "The state data must be finite, have 6 columns (7 when a mass is given) and, "
            "with a mass, a positive mass"
        )
    result = {
        "sample_count": len(values),
        "assessment": "Descriptive statistics of the samples",
    }
    for name, series in (
        ("temperature_kelvin", values[:, 5]),
        *([("volume_nm3", values[:, 6])] if mass_dalton is not None else []),
        ("potential_energy_kj_mol", values[:, 2]),
        *(
            [("density_g_ml", mass_dalton * 0.00166053906660 / values[:, 6])]
            if mass_dalton is not None
            else []
        ),
    ):
        half = len(series) // 2
        result[name] = {
            "mean": float(series.mean()),
            "standard_deviation": float(series.std()),
            "first_half_mean": float(series[:half].mean()) if half else None,
            "second_half_mean": float(series[half:].mean()) if half else None,
        }
    return result


def validate_hydrogen_timestep(system, topology, timestep_fs, unit, app):
    """Require constrained bonds to ordinary-mass hydrogen above the 1 fs profile."""
    if timestep_fs <= 1:
        return
    constrained = {
        frozenset(system.getConstraintParameters(i)[:2])
        for i in range(system.getNumConstraints())
    }
    for a, b in topology.bonds():
        light_h = any(
            atom.element == app.element.hydrogen
            and 0 < system.getParticleMass(atom.index).value_in_unit(unit.dalton) < 3
            for atom in (a, b)
        )
        if light_h and frozenset((a.index, b.index)) not in constrained:
            raise ValueError(
                f"timestep_fs is {timestep_fs} fs; timesteps above 1 fs require constrained bonds to "
                "hydrogens of ordinary mass. Prepare the system with HBonds constraints or "
                "set timestep_fs to 1 fs or less"
            )
