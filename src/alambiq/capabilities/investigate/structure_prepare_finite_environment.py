"""Cut a finite droplet around a solute from a periodic solvated system.

The capability builds a finite cluster of a solute and solvent for use as the
surroundings of a QM/MM calculation. The inputs are the Amber topology of a periodic,
solvated system and an OpenMM state file of it. Bonds are unwrapped so that every
molecule is whole, whole molecules are imaged around an anchor molecule, and every water
or ion with any atom inside a sphere of the radius given in the request, centered on the
heavy atom of the anchor molecule that is nearest to its center of mass, is retained
whole, together with the declared solute residues. The retained atoms receive their own
Amber topology. The capability compares every force-field parameter of every retained
particle with the source and refuses a cluster in which any parameter differs. The
boundary is a separate set of explicit potentials: harmonic position restraints tie
the heavy atoms of the outer solute residues to their source positions, and a wall acts
on the center of mass of each retained water or ion. Protected atoms are the atoms listed
in `protected_atoms`, for example the quantum region; the capability refuses a request
in which a protected atom belongs to a residue that would be restrained.

The result is a finite cluster, neutral or charged, whose retained atoms and parameters
are those of the periodic system, with an explicit boundary, for use as the surroundings
of a QM/MM calculation.

Limits. A droplet has a surface. The wall and the springs approximate the bulk beyond the
surface, and the dielectric screening, the density at the edge and the net charge of the
cluster differ from those of the periodic system. The quantum region has to lie inside
the sphere and away from its surface; the capability refuses a protected atom only when
its residue would be restrained, a residue whose atoms all lie farther than
`inner_solute_radius_angstrom` from the droplet center. The cluster starts from the
positions of one frame of the source.
The periodic cell must be orthorhombic. Of the `box_shape` values of the
preparation capabilities, only `cube` gives an orthorhombic cell. The capability refuses
a residue that is not a complete solute residue, a whole water or a single ion, and a
cluster of more than 100,000 atoms. The cluster has no bond constraints, and its charge
and Lennard-Jones interactions are evaluated for all atom pairs. The cluster is a
finite system and is valid only with the boundary potentials in `external_system_xml`,
for QM/MM calculations and dynamics.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from alambiq.capabilities.investigate.structure_prepare_finite_environment_report import (
    render,
)
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class FiniteRequest:
    """The periodic system, the solute and the droplet.

    amber_topology:
        The Amber topology of the periodic solvated system, for example the one
        `structure.prepare_predicted_complex_for_dynamics` returns.
    state:
        An OpenMM state file of that system, with positions and the periodic box.
    solute_residues:
        The one-based residue numbers of the solute, ascending and distinct. Every other
        residue must be a whole water or a single ion.
    anchor_residue:
        The one-based number of the residue on which the droplet is centered, one of the
        solute residues. The center is the heavy atom of this residue that is nearest to
        its center of mass.
    protected_atoms:
        The one-based indices of the atoms that are never restrained, all in the solute
        (for example the quantum region).
    solvent_radius_angstrom:
        The radius of the sphere around the center of the droplet within which waters and
        ions are retained, 10 to 50.
    inner_solute_radius_angstrom:
        Heavy atoms of each solute residue whose atoms all lie farther than this radius from
        the droplet center are restrained. Range 5 to 40; less than
        `solvent_radius_angstrom`.
    position_coefficient_kj_mol_nm2:
        The spring constant of those restraints.
    wall_coefficient_kj_mol_nm2:
        The force constant of the wall on the center of mass of each water and ion.
    threads:
        The number of CPU threads for the force evaluation that gives the force
        diagnostics.
    """

    amber_topology: str
    state: str
    solute_residues: list[int]
    anchor_residue: int
    protected_atoms: list[int]
    solvent_radius_angstrom: float = 25.0
    inner_solute_radius_angstrom: float = 20.0
    position_coefficient_kj_mol_nm2: float = 2092.0
    wall_coefficient_kj_mol_nm2: float = 4184.0
    threads: int = 2


@dataclass(frozen=True, slots=True)
class FiniteResult:
    """The cluster's size and charge, its boundary and its files."""

    source_atoms: int
    atoms: int
    solute_atoms: int
    solute_residues: int
    solvent_and_ion_molecules: int
    net_charge_e: float
    restrained_heavy_atoms: int
    rms_force_kj_mol_nm: float
    """On the source positions, without the boundary potentials."""
    max_force_kj_mol_nm: float
    inverted_alpha_residues: list[int]
    source_frame_time_ps: float
    system_xml: str
    external_system_xml: str
    """The boundary potentials, a separate system over the same particles."""
    state_xml: str
    topology_pdb: str
    amber_topology: str
    source_atom_map: str
    """The source atom of each retained atom, and the positions of the protected atoms and the center in the cluster."""
    manifest: str
    versions: dict[str, str]


@capability(
    "structure.prepare_finite_environment",
    version="1.0",
    family="investigate",
    summary="Cut a finite solvent droplet with explicit boundary potentials around a solute",
    render=render,
    environment="openff",
)
def prepare_finite_environment(
    request: FiniteRequest, ctx: StepContext
) -> FiniteResult:
    """Check the declaration, cut the droplet, and describe it."""
    for name, items in (
        ("solute_residues", request.solute_residues),
        ("protected_atoms", request.protected_atoms),
    ):
        if not items or any(i < 1 for i in items) or items != sorted(set(items)):
            raise ValueError(
                f"{name} is empty, contains an index below 1, or is not ascending and "
                "distinct; a nonempty ascending list of distinct one-based indices is required"
            )
    if request.anchor_residue not in request.solute_residues:
        raise ValueError(
            f"anchor_residue is {request.anchor_residue}, which is not in solute_residues; "
            "a member of solute_residues is required"
        )
    for name, value, low, high in (
        ("solvent_radius_angstrom", request.solvent_radius_angstrom, 10, 50),
        ("inner_solute_radius_angstrom", request.inner_solute_radius_angstrom, 5, 40),
        (
            "position_coefficient_kj_mol_nm2",
            request.position_coefficient_kj_mol_nm2,
            1e-6,
            1e5,
        ),
        ("wall_coefficient_kj_mol_nm2", request.wall_coefficient_kj_mol_nm2, 1e-6, 1e5),
    ):
        if not (math.isfinite(value) and low <= value <= high):
            raise ValueError(
                f"{name} is {value}; a finite value from {low} to {high} is required"
            )
    if request.inner_solute_radius_angstrom >= request.solvent_radius_angstrom:
        raise ValueError(
            f"inner_solute_radius_angstrom ({request.inner_solute_radius_angstrom}) is not "
            f"smaller than solvent_radius_angstrom ({request.solvent_radius_angstrom}); "
            "a smaller value is required"
        )
    if not 1 <= request.threads <= 64:
        raise ValueError(
            f"threads is {request.threads}; an integer from 1 to 64 is required"
        )
    output = ctx.workdir / "finite"
    found = ctx.run_in_environment(
        "openff",
        "openff_finite_environment:run",
        {
            "amber_topology": str(ctx.resolve(request.amber_topology)),
            "state": str(ctx.resolve(request.state)),
            "solute_residues": request.solute_residues,
            "anchor_residue": request.anchor_residue,
            "protected_atoms": request.protected_atoms,
            "solvent_radius_angstrom": request.solvent_radius_angstrom,
            "inner_solute_radius_angstrom": request.inner_solute_radius_angstrom,
            "position_coefficient_kj_mol_nm2": request.position_coefficient_kj_mol_nm2,
            "wall_coefficient_kj_mol_nm2": request.wall_coefficient_kj_mol_nm2,
            "threads": request.threads,
            "output": str(output),
        },
        timeout=3600,
    )
    geometry: dict[str, Any] = found["geometry_diagnostics"]
    if abs(found["net_charge_e"]) > 1e-3:
        ctx.note(f"The cluster has a net charge of {found['net_charge_e']:+.3f} e.")
    return FiniteResult(
        source_atoms=found["source_atom_count"],
        atoms=found["atom_count"],
        solute_atoms=found["solute_atom_count"],
        solute_residues=found["solute_residue_count"],
        solvent_and_ion_molecules=len(found["solvent_ion_groups_one_based"]),
        net_charge_e=found["net_charge_e"],
        restrained_heavy_atoms=len(found["restrained_heavy_atoms_one_based"]),
        rms_force_kj_mol_nm=found["physical_forces"]["rms_component_kj_mol_nm"],
        max_force_kj_mol_nm=found["physical_forces"]["maximum_atom_kj_mol_nm"],
        inverted_alpha_residues=geometry["inverted_alpha_residues_one_based"],
        source_frame_time_ps=found["source_frame_time_ps"],
        system_xml=str(ctx.keep(output / "system.xml", "application/xml").path),
        external_system_xml=str(
            ctx.keep(output / "external.xml", "application/xml").path
        ),
        state_xml=str(ctx.keep(output / "state.xml", "application/xml").path),
        topology_pdb=str(ctx.keep(output / "topology.pdb", "chemical/x-pdb").path),
        amber_topology=str(
            ctx.keep(output / "complex.prmtop", "application/octet-stream").path
        ),
        source_atom_map=str(
            ctx.keep(output / "source-atom-map.json", "application/json").path
        ),
        manifest=ctx.keep_json(json.loads(json.dumps(found))),
        versions=found["versions"],
    )
