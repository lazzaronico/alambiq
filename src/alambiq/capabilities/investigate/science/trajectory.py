"""Streaming inspection of an OpenMM trajectory with MDTraj.

The trajectory is read once, in chunks. Every frame contributes to the scalar
series and statistics; only a bounded number of frames is kept as coordinates for
a preview, and a bounded number as points of the plotted series. Which atoms are
fitted, how periodic images are treated and which distances, angles and contacts
are measured are all explicit choices of the caller.
"""

from __future__ import annotations

import csv
import io
import math
from itertools import islice, product
from typing import Any

from alambiq.capabilities.investigate.science.observables import (
    atom_identity,
    molecular_observables,
    resolve_atoms,
    validate_observables,
)
from alambiq.capabilities.investigate.science.periodic_images import (
    inspect_frames,
    protein_atoms,
)

MAX_FRAMES = 100_000
MAX_ATOMS = 250_000
MAX_VALUES = 50_000_000
STATE_COLUMNS = [
    "step",
    "time_ps",
    "potential_kj_mol",
    "kinetic_kj_mol",
    "total_kj_mol",
    "temperature_kelvin",
]


def check_inspection(
    *,
    max_preview_frames: int,
    max_plot_frames: int,
    chunk_frames: int,
    imaging: str,
    observables: list[dict[str, Any]],
    display_atom_indices: list[int] | None,
    ligand_contacts: dict[str, Any] | None,
    protein_image_check: dict[str, Any] | None,
) -> None:
    """Refuse settings the inspection cannot honour, before any file is read."""
    if not 2 <= max_preview_frames <= 100:
        raise ValueError(
            f"max_preview_frames is {max_preview_frames}; max_preview_frames must be an "
            "integer from 2 to 100"
        )
    if not 100 <= max_plot_frames <= 5000:
        raise ValueError(
            f"max_plot_frames is {max_plot_frames}; max_plot_frames must be an integer "
            "from 100 to 5000"
        )
    if not 1 <= chunk_frames <= 64:
        raise ValueError(
            f"chunk_frames is {chunk_frames}; chunk_frames must be an integer from 1 to 64"
        )
    if imaging not in {"as-stored", "whole-molecules"}:
        raise ValueError(
            f"imaging is {imaging!r}; imaging must be as-stored or whole-molecules"
        )
    validate_observables(observables)
    if protein_image_check is not None:
        cutoff, frames = protein_image_check["cutoff_nm"], protein_image_check["frames"]
        if not (math.isfinite(cutoff) and 0 < cutoff <= 3):
            raise ValueError(
                f"protein_image_check cutoff_nm is {cutoff}; cutoff_nm must be in (0, 3]"
            )
        if not 1 <= frames <= MAX_FRAMES:
            raise ValueError(
                f"protein_image_check frames is {frames}; frames must be 1–{MAX_FRAMES}"
            )
    if display_atom_indices is not None and (
        not 1 <= len(display_atom_indices) <= 5000
        or any(i < 0 for i in display_atom_indices)
        or len(set(display_atom_indices)) != len(display_atom_indices)
    ):
        raise ValueError(
            f"display_atom_indices holds {len(display_atom_indices)} indices; "
            "display_atom_indices must select 1–5000 distinct nonnegative topology indices"
        )
    if ligand_contacts is not None:
        cutoff = ligand_contacts["cutoff_angstrom"]
        if ligand_contacts["ligand_residue_index"] < 0 or not (
            math.isfinite(cutoff) and 0 < cutoff <= 20
        ):
            raise ValueError(
                f"ligand_contacts has ligand_residue_index "
                f"{ligand_contacts['ligand_residue_index']} and cutoff_angstrom {cutoff}; "
                "ligand_contacts needs a zero-based ligand_residue_index and a "
                "cutoff_angstrom in (0, 20]"
            )


def parse_state_data(text: str) -> list[dict[str, Any]]:
    """Read the state samples `simulation.run_molecular_dynamics` writes."""
    reader = csv.reader(io.StringIO(text), strict=True)
    header = next(reader, None)
    if header not in (STATE_COLUMNS, [*STATE_COLUMNS, "volume_nm3"]):
        raise ValueError(
            f"State data has the columns {', '.join(header or [])}. State data needs the "
            "columns step, time_ps, potential_kj_mol, kinetic_kj_mol, total_kj_mol, "
            "temperature_kelvin and optionally volume_nm3"
        )
    rows: list[dict[str, Any]] = []
    for raw in reader:
        if len(rows) >= MAX_FRAMES + 1 or len(raw) != len(header):
            raise ValueError(
                f"State data has more than {MAX_FRAMES + 1} records or a record with "
                f"{len(raw)} columns against {len(header)} in the header. State data needs "
                f"at most {MAX_FRAMES + 1} records with matching columns"
            )
        if not raw[0].isdigit():
            raise ValueError(f"State step is not a nonnegative integer: {raw[0]!r}")
        values = [float(v) for v in raw[1:]]
        row = dict(zip(header, [int(raw[0]), *values], strict=True))
        if not all(math.isfinite(v) and abs(v) <= 1e100 for v in values):
            raise ValueError(
                f"State data holds a non-finite value or a magnitude above 1e100 at step "
                f"{raw[0]}"
            )
        if row["temperature_kelvin"] < 0 or row.get("volume_nm3", 1) <= 0:
            raise ValueError(
                f"State step {row['step']} has temperature {row['temperature_kelvin']:g} K "
                f"and volume {row.get('volume_nm3', 'not recorded')}; the temperature must "
                "not be negative and the volume must be positive"
            )
        if rows and (
            row["step"] <= rows[-1]["step"] or row["time_ps"] <= rows[-1]["time_ps"]
        ):
            raise ValueError(
                f"State step {row['step']} at {row['time_ps']:g} ps does not follow step "
                f"{rows[-1]['step']} at {rows[-1]['time_ps']:g} ps; steps and times must "
                "increase"
            )
        rows.append(row)
    if not rows:
        raise ValueError("State data is empty")
    steps = [r["step"] for r in rows]
    if len({b - a for a, b in zip(steps, steps[1:], strict=False)}) > 1:
        raise ValueError(
            "State samples are not evenly spaced in steps; the spacing in steps must be "
            "constant"
        )
    return rows


def ligand_contacts(traj, choices: dict[str, Any]) -> dict[str, Any]:
    """Minimum heavy-atom distances on original coordinates, before CA fitting."""
    import mdtraj as md
    import numpy as np

    index = choices["ligand_residue_index"]
    if index >= traj.topology.n_residues:
        raise ValueError(
            f"ligand_residue_index {index} is outside the supplied topology, which has "
            f"{traj.topology.n_residues} residues"
        )
    ligand = traj.topology.residue(index)
    if (
        ligand.is_protein
        or ligand.is_water
        or not any(
            atom.element is not None and atom.element.symbol == "C"
            for atom in ligand.atoms
        )
    ):
        raise ValueError(
            f"Residue {ligand.name}{ligand.resSeq} at index {index} is a protein or water "
            "residue or contains no carbon; select a non-protein, non-water ligand residue "
            "containing carbon"
        )
    if choices["periodic"]:
        boxes = traj.unitcell_vectors
        if (
            boxes is None
            or not np.isfinite(boxes).all()
            or np.any(np.linalg.det(boxes) <= 0)
        ):
            raise ValueError(
                "Periodic contacts require a finite, positive-volume unit cell in every frame"
            )

    def identity(residue):
        atoms = list(residue.atoms)
        if any(atom.element is None for atom in atoms):
            raise ValueError(
                f"Residue {residue.name}{residue.resSeq} has atoms without elements; "
                "contact residues require explicit atom elements"
            )
        heavy = [atom.index for atom in atoms if atom.element.symbol != "H"]
        if not heavy:
            raise ValueError(
                f"Residue {residue.name}{residue.resSeq} has no heavy atoms; contact "
                "residues require heavy atoms"
            )
        return {
            "residue_index": residue.index,
            "residue_number": residue.resSeq,
            "residue_name": residue.name,
            "chain_index": residue.chain.index,
            "chain_id": residue.chain.chain_id or str(residue.chain.index),
            "heavy_atom_indices": heavy,
        }

    selected = identity(ligand)
    residues = []
    for residue in traj.topology.residues:
        if not residue.is_protein:
            continue
        record = identity(residue)
        pairs = product(selected["heavy_atom_indices"], record["heavy_atom_indices"])
        minimum = np.full(traj.n_frames, np.inf)
        while batch := list(islice(pairs, 4096)):
            distances = md.compute_distances(traj, batch, periodic=choices["periodic"])
            minimum = np.minimum(minimum, distances.min(axis=1).astype(float) * 10)
        if not np.isfinite(minimum).all():
            raise ValueError(
                f"The ligand contact distance to residue {residue.name}{residue.resSeq} "
                "is non-finite"
            )
        count = int(np.count_nonzero(minimum <= choices["cutoff_angstrom"]))
        residues.append(
            {
                **record,
                "minimum_distances_angstrom": minimum.tolist(),
                "contact_frame_count": count,
                "contact_fraction": count / traj.n_frames,
            }
        )
    return {
        "ligand": selected,
        "cutoff_angstrom": choices["cutoff_angstrom"],
        "periodic": choices["periodic"],
        "frame_count": traj.n_frames,
        "residues": residues,
        "method": "Minimum ligand-to-protein-residue heavy-atom distance in every saved frame; contact when distance <= cutoff; fraction counts saved frames, unweighted by time",
        "coordinates": "Original unfitted coordinates; minimum-image distances with recorded unit cells"
        if choices["periodic"]
        else "Original unfitted coordinates; nonperiodic Euclidean distances",
    }


def align_state(rows: list[dict[str, Any]], frames: int) -> list[dict[str, Any]]:
    """Pair state samples with trajectory frames.

    The dynamics run writes a state sample at the start of production and one at
    each frame, so there is one more sample than frames; the first belongs to no frame.
    """
    if len(rows) == frames + 1:
        return rows[1:]
    if len(rows) == frames:
        return rows
    raise ValueError(
        f"The trajectory has {frames} frames but the state data has {len(rows)} samples; "
        "the number of samples must equal the number of frames or exceed it by one "
        "sample at the start"
    )


def inspect_trajectory(
    topology,
    trajectory,
    state_rows: list[dict[str, Any]],
    params: dict[str, Any],
    *,
    reference=None,
    array_output=None,
) -> dict[str, Any]:
    """Stream coordinates once; keep scalar arrays and explicitly indexed previews."""
    import mdtraj as md
    import numpy as np

    top = md.load_pdb(str(topology)).topology
    with md.formats.DCDTrajectoryFile(str(trajectory)) as handle:
        count = len(handle)
    if not 1 <= count <= MAX_FRAMES or not 3 <= top.n_atoms <= MAX_ATOMS:
        raise ValueError(
            f"The trajectory has {count} frames and the topology {top.n_atoms} atoms; "
            f"inspection supports 1–{MAX_FRAMES} frames and 3–{MAX_ATOMS} atoms"
        )
    series = align_state(state_rows, count)
    fit = top.select("protein and name CA")
    display = np.asarray(
        params["display_atom_indices"]
        if params["display_atom_indices"] is not None
        else top.select("not water and not element H").tolist(),
        dtype=int,
    )
    if len(fit) < 3 or not 1 <= len(display) <= 5000 or np.any(display >= top.n_atoms):
        raise ValueError(
            f"The topology has {len(fit)} protein CA atoms and {len(display)} display atoms. "
            f"Inspection needs at least 3 protein CA atoms and 1–5000 display atoms with "
            f"indices below {top.n_atoms}; set display_atom_indices for larger solutes"
        )
    selected_rows = params["observables"]
    selected_indices = [resolve_atoms(top, row["atoms"]) for row in selected_rows]
    metric_ids = [
        "ca_rmsd_angstrom",
        "ca_radius_gyration_angstrom",
        *[r["id"] for r in selected_rows],
    ]
    if len(set(metric_ids)) != len(metric_ids):
        duplicates = sorted({m for m in metric_ids if metric_ids.count(m) > 1})
        raise ValueError(
            f"Observable ids {duplicates} collide with built-in structural metrics"
        )
    contact_residue_count = (
        sum(r.is_protein for r in top.residues)
        if params["ligand_contacts"] is not None
        else 0
    )
    if count * (len(metric_ids) + contact_residue_count) > MAX_VALUES:
        raise ValueError(
            f"The request needs {count * (len(metric_ids) + contact_residue_count)} scalar "
            f"observable values; inspection supports at most {MAX_VALUES}"
        )
    preview = np.unique(
        np.linspace(0, count - 1, min(params["max_preview_frames"], count), dtype=int)
    )
    plot = np.union1d(
        preview,
        np.linspace(0, count - 1, min(params["max_plot_frames"], count), dtype=int),
    )
    coords: list = []
    values = np.empty((count, len(metric_ids)), dtype=np.float64)
    contact_values = (
        np.empty((count, contact_residue_count), dtype=np.float32)
        if contact_residue_count
        else None
    )
    reference_frame = md.load_pdb(str(reference)) if reference else None
    supplied_reference = reference_frame
    reference_fit = (
        resolve_atoms(
            reference_frame.topology, [atom_identity(top.atom(int(i))) for i in fit]
        )
        if reference
        else fit
    )
    moments = np.zeros((len(fit), 3), dtype=np.float64)
    squares = np.zeros_like(moments)
    contacts = None
    image_check = params["protein_image_check"]
    image_records: list = []
    image_atoms = protein_atoms(top) if image_check else None
    image_frames = (
        np.unique(
            np.linspace(0, count - 1, min(count, image_check["frames"]), dtype=int)
        )
        if image_check
        else []
    )
    offset = 0
    # Bound the coordinate batch even at the maximum atom count.
    chunk_frames = min(params["chunk_frames"], max(1, 4_000_000 // top.n_atoms))

    def chunks():
        # Stored files have no extension, so the format is named explicitly.
        with md.formats.DCDTrajectoryFile(str(trajectory)) as handle:
            while True:
                try:
                    chunk = handle.read_as_traj(top, n_frames=chunk_frames)
                except ValueError as exc:
                    raise ValueError(
                        f"The trajectory and the topology disagree about the atoms: {exc}"
                    ) from exc
                if not len(chunk):
                    break
                yield chunk

    for chunk in chunks():
        if not np.isfinite(chunk.xyz).all() or chunk.n_atoms != top.n_atoms:
            raise ValueError(
                "A trajectory chunk has non-finite coordinates or an atom count that "
                "differs from the topology"
            )
        end = offset + chunk.n_frames
        if end > count:
            raise ValueError(
                f"The trajectory frame count differs from the {count} frames counted at "
                "the start of the inspection"
            )
        periodic = params["periodic_observables"]
        if periodic or image_check or params["imaging"] == "whole-molecules":
            if (
                chunk.unitcell_vectors is None
                or not np.isfinite(chunk.unitcell_vectors).all()
                or np.any(np.linalg.det(chunk.unitcell_vectors) <= 0)
            ):
                raise ValueError(
                    "Periodic analysis requires a positive-volume cell in every frame"
                )
        if image_check:
            image_records.extend(
                inspect_frames(
                    chunk,
                    image_atoms,
                    image_frames,
                    offset,
                    series,
                    image_check["cutoff_nm"],
                )
            )
        if contact_values is not None:
            observed = ligand_contacts(chunk, params["ligand_contacts"])
            if contacts is None:
                contacts = observed
            contact_values[offset:end] = np.asarray(
                [r["minimum_distances_angstrom"] for r in observed["residues"]]
            ).T
        if params["imaging"] == "whole-molecules":
            chunk.make_molecules_whole(inplace=True)
        extra = molecular_observables(chunk, selected_rows, selected_indices, periodic)
        if reference_frame is None:
            reference_frame = chunk[0]
        ref_xyz = reference_frame.xyz[0, reference_fit]
        if np.linalg.matrix_rank(ref_xyz - ref_xyz.mean(axis=0), tol=1e-6) < 2:
            raise ValueError("Reference CA coordinates cannot all be collinear")
        # Fit before computing RMSD to avoid platform-specific MDTraj RMSD centering side effects.
        chunk.superpose(
            reference_frame, atom_indices=fit, ref_atom_indices=reference_fit
        )
        xyz = chunk.xyz[:, fit].astype(np.float64)
        values[offset:end, 0] = (
            np.sqrt(np.mean(np.sum((xyz - ref_xyz) ** 2, axis=2), axis=1)) * 10
        )
        centered = xyz - xyz.mean(axis=1, keepdims=True)
        values[offset:end, 1] = (
            np.sqrt(np.mean(np.sum(centered**2, axis=2), axis=1)) * 10
        )
        for i, value in enumerate(extra, 2):
            values[offset:end, i] = value
        moments += xyz.sum(axis=0)
        squares += (xyz**2).sum(axis=0)
        coords.extend(
            (
                chunk.xyz[[int(i - offset) for i in preview if offset <= i < end]][
                    :, display
                ]
                * 10
            )
            .astype(float)
            .tolist()
        )
        offset = end
    if offset != count or not np.isfinite(values).all():
        raise ValueError(
            f"The inspection read {offset} of {count} frames, or a structural observable "
            "is non-finite"
        )
    if contacts is not None:
        contacts["frame_count"] = count
        for i, residue in enumerate(contacts["residues"]):
            column = contact_values[:, i]
            residue.update(
                minimum_distances_angstrom=column[plot].astype(float).tolist(),
                contact_frame_count=int(
                    np.count_nonzero(column <= contacts["cutoff_angstrom"])
                ),
                minimum_distance_angstrom=float(column.min()),
            )
            residue["contact_fraction"] = residue["contact_frame_count"] / count
    if supplied_reference is not None:
        ref_ca = supplied_reference.xyz[0, reference_fit].astype(np.float64)
        ref_values = [
            0.0,
            float(
                np.sqrt(np.mean(np.sum((ref_ca - ref_ca.mean(axis=0)) ** 2, axis=1)))
                * 10
            ),
        ]
        ref_periodic = (
            params["periodic_observables"]
            and supplied_reference.unitcell_vectors is not None
        )
        ref_values += [
            float(v[0])
            for v in molecular_observables(
                supplied_reference,
                selected_rows,
                [
                    resolve_atoms(supplied_reference.topology, row["atoms"])
                    for row in selected_rows
                ],
                ref_periodic,
            )
        ]
    else:
        ref_values = values[0].tolist()
    plot_series = [
        {**series[i], **dict(zip(metric_ids, values[i].tolist(), strict=True))}
        for i in plot
    ]
    lookup = {int(original): i for i, original in enumerate(display)}
    atoms = []
    for index in display:
        atom = top.atom(int(index))
        residue = atom.residue
        atoms.append(
            {
                "source_atom_index": int(index),
                "name": atom.name,
                "element": atom.element.symbol if atom.element else "X",
                "residue_name": residue.name,
                "residue_index": residue.index,
                "residue_number": residue.resSeq,
                "chain_index": residue.chain.index,
                "chain_id": residue.chain.chain_id or str(residue.chain.index),
            }
        )
    units = [
        "angstrom",
        "angstrom",
        *[
            "degree"
            if r["kind"] == "angle"
            else "fraction"
            if r["kind"] == "contact"
            else "angstrom"
            for r in selected_rows
        ],
    ]
    arrays = {
        "values": values,
        "observable_ids": np.asarray(metric_ids),
        "units": np.asarray(units),
        "steps": np.asarray([r["step"] for r in series], dtype=np.int64),
        "times_ps": np.asarray([r["time_ps"] for r in series]),
    }
    if contact_values is not None:
        arrays["ligand_distances_angstrom"] = contact_values
        arrays["contact_residue_indices"] = np.asarray(
            [r["residue_index"] for r in contacts["residues"]], dtype=np.int64
        )
    if array_output:
        np.savez_compressed(array_output, **arrays)
    rmsf = (
        np.sqrt(np.maximum(0, (squares / count - (moments / count) ** 2).sum(axis=1)))
        * 10
    )
    return {
        **({"ligand_contacts": contacts} if contacts is not None else {}),
        **(
            {
                "protein_image_check": {
                    "cutoff_nm": image_check["cutoff_nm"],
                    "checked_frame_count": len(image_records),
                    "total_frame_count": count,
                    "all_saved_frames_checked": len(image_records) == count,
                    "within_cutoff_frame_count": sum(
                        r["within_cutoff"] for r in image_records
                    ),
                    "minimum_distance_nm": min(
                        r["minimum_distance_nm"] for r in image_records
                    ),
                    "passed": not any(r["within_cutoff"] for r in image_records),
                    "frames": image_records,
                    "scope": "Exact closest protein atom to translated protein copy in selected saved frames, before fitting, after whole-molecule reconstruction. Includes hydrogens. Requires one connected protein. Applies to the selected saved frames and to the protein-to-image distance of the protein.",
                }
            }
            if image_check
            else {}
        ),
        "atom_count": top.n_atoms,
        "frame_count": count,
        "fit_atom_indices": fit.tolist(),
        "atoms": atoms,
        "bonds": [
            [lookup[a.index], lookup[b.index]]
            for a, b in top.bonds
            if a.index in lookup and b.index in lookup
        ],
        "preview_frame_indices": preview.tolist(),
        "series_frame_indices": plot.tolist(),
        "coordinates_angstrom": coords,
        "series": plot_series,
        "parameters": params,
        "reference_values": {
            "source": "supplied structure"
            if supplied_reference is not None
            else "first saved frame",
            "values": dict(zip(metric_ids, ref_values, strict=True)),
        },
        "observables": {
            "ids": metric_ids,
            "units": units,
            "selections": selected_rows,
            "mean": values.mean(axis=0).tolist(),
            "standard_deviation": values.std(axis=0).tolist(),
            "minimum": values.min(axis=0).tolist(),
            "maximum": values.max(axis=0).tolist(),
            "first": values[0].tolist(),
            "last": values[-1].tolist(),
        },
        "ca_fluctuations": {
            "atoms": [atom_identity(top.atom(int(i))) for i in fit],
            "rmsf_angstrom": rmsf.tolist(),
            "reference": "CA fit to the common reference; fluctuations about this trajectory's fitted mean",
        },
        "state": {
            "samples": len(series),
            "first_step": series[0]["step"],
            "last_step": series[-1]["step"],
            "first_time_ps": series[0]["time_ps"],
            "last_time_ps": series[-1]["time_ps"],
            "mean_temperature_kelvin": float(
                np.mean([r["temperature_kelvin"] for r in series])
            ),
            "mean_potential_kj_mol": float(
                np.mean([r["potential_kj_mol"] for r in series])
            ),
            "mean_volume_nm3": (
                float(np.mean([r["volume_nm3"] for r in series]))
                if "volume_nm3" in series[0]
                else None
            ),
        },
        "streaming": {
            "chunk_frames": chunk_frames,
            "plotted_frame_count": len(plot),
            "full_frame_count": count,
        },
        "method": {
            "reference": "supplied reference_structure, exact CA identities"
            if reference
            else "first saved frame",
            "fit": "least-squares protein CA fit; CA RMSD in angstrom",
            "display": "explicit topology indices or non-water heavy atoms",
            "periodic_boundary": params["imaging"],
            "time": "State samples matched to saved frames by order; times as recorded by the run",
            "bonds": "from the supplied PDB topology as parsed by MDTraj; single bonds for display",
            "series": "All frames contribute to statistics and the full observable arrays; plots and coordinate previews use the recorded frame-index selections",
            "scope": "Structural descriptors and fluctuations of the saved frames",
        },
    }
