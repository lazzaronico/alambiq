# Binding free energy and QM/MM

Relative binding free energies, and the building, sampling and analysis of a
quantum-mechanical/molecular-mechanical (QM/MM) reaction. The capabilities are
`structure.prepare_relative_binding_inputs`, `simulation.run_relative_free_energy` and
`simulation.analyze_binding_free_energy` for binding; `structure.prepare_reactive_coordinates`,
`structure.prepare_protein_ligand_system`, `structure.prepare_finite_environment`,
`simulation.evaluate_qmmm_state`, `simulation.run_qmmm_dynamics` and
`simulation.analyze_reaction_free_energy` for QM/MM. A number from either set of
capabilities is the result of a calculation under a force field and a protocol. The
diagnostics reported with it (sampling, overlap, replica spread, cycle closure) describe
sampling and internal consistency.

## Relative binding free energy

A relative binding free energy compares two similar ligands: the binding free energy of
ligand B minus that of ligand A, for the directed edge A to B. A negative value means that
B binds better than A. It is computed by transforming A into B alchemically, once in the
binding site of the protein (the bound leg) and once in water (the solvent leg). The
bound-leg value minus the solvent-leg value is the relative binding free energy, because
the thermodynamic cycle closes.

`structure.prepare_relative_binding_inputs` starts from a deposited complex with a
six-carbon aromatic ring in the site (for example benzene in a lysozyme cavity) and
prepares the protein chain and a series of neutral congeners. The ring of each
congener is laid on the deposited ring in each of its 12 symmetry-related ways, the
substituents are relaxed with the ring held, and the placement with the lowest clash is
retained. The pose of a substituent is a geometric placement. The clash score of a
placement is the summed squared shortfall below 2.2 Å between substituent atoms and
protein heavy atoms. It is recorded in the manifest; no clash score causes refusal. The
atom mapping for each edge is proposed by the LOMAP mapper of OpenFE, without element
changes. `simulation.run_relative_free_energy` checks that mapped atoms retain element,
isotope, aromaticity and charge, that stereocenters and double bonds retain their
orientation, that every ring touched is mapped whole and that the core is connected.

`simulation.run_relative_free_energy` runs one leg with OpenFE. The ligands use Sage 2.3.0
with AshGC charges; the protein and the water use ff14SB and TIP3P. The leg uses hydrogen
masses of 1.0 amu (no hydrogen mass repartitioning), a 2 fs step, lambda windows
connecting the two ligands (5 to 41; default 11) and replica exchange between the lambda
windows. The request field `replicas` sets 1 to 10 independent replicas per leg (default
3). Each replica reports its free-energy difference, the error given by MBAR, the overlap
between neighboring windows, the effective sample size per window and the forward and
reverse estimates for increasing fractions of the production samples. A leg with a
production length below 1,000 ps or fewer than 3 replicas receives a note that the
settings are far below those a converged free energy needs. Convergence is assessed from
the overlap, the effective sample sizes and the forward and reverse estimates.

The test suite runs the transformation of a ligand into an identical copy of itself, for
which the free-energy difference is 0, when the engine environments are configured
(`ALAMBIQ_ENV_OPENFE`, `ALAMBIQ_ENV_OPENMM`). It asserts, for each replica, an absolute
free-energy difference below 1e-4 kcal/mol and an MBAR error below 1e-4 kcal/mol.

`simulation.analyze_binding_free_energy` pairs the legs of each edge, subtracts the
solvent mean from the bound mean, and reports with every number:

- the spread across replicas, which measures reproducibility from independent starts;
- the error that MBAR reports within a replica; it assumes uncorrelated samples and is
  reported separately from the spread, never combined with it;
- the overlap and the effective samples;
- the difference between the forward and the reverse estimate at the stored fraction 0.5
  of the data (each leg stores forward and reverse estimates for increasing fractions of
  the production samples; the difference is absent when 0.5 is not stored);
- closed cycles of edges summed (closure measures the internal consistency of the
  simulated edges);
- measured reference values placed beside the calculated differences when they are
  supplied.

Measured values are given per ligand, and the reference difference of an edge is computed
from the values of its two ligands. Edges that share a ligand therefore share its error,
and the analysis propagates that covariance. An experimental cycle built from per-ligand
values closes algebraically; the independently simulated cycle is the consistency
diagnostic. When a cycle check is set, the closure must not exceed the larger of an
absolute floor and a multiple of the replica standard error of the cycle (the request
fields `cycle_absolute_floor_kcal_mol` and `cycle_replica_sem_multiplier`). The replica
standard error of a cycle assumes that the edges of the cycle were sampled independently.

Checks set in the request convert each diagnostic into a verdict. Drift of the ligand
center of mass in the bound leg is detected only when structural diagnostics are supplied
and `maximum_ligand_com_drift_angstrom` is set. The verdicts apply when the protein
conformation, the protonation of the binding site and the ligand charge states are as
prepared.

## Building a QM/MM system

`structure.prepare_reactive_coordinates` builds the coordinates of an enzyme with its
substrate from deposited parts: a target structure whose termini the crystal did not
resolve, a donor structure that contains those termini, a structure in which the
substrate or a close analogue is deposited, and the substrate as an explicit molecule.
Missing termini are copied from the donor chain whose anchor residues (1 to 10 observed
residues next to the gap) fit the anchor residues of the target with the lowest RMSD; a
terminus is accepted only if the RMSD does not exceed the declared maximum, the new
peptide bond length lies in the declared range and no clash occurs. The substrate takes
the coordinates of the deposited ligand through an atom mapping, is transferred to the
target by aligning an analogue located in the active site, and receives hydrogens from a
force field with every heavy atom held. Deposited atoms are at their deposited positions
and every modeled terminal atom is copied from a donor chain. The result is a geometric model.

`structure.prepare_protein_ligand_system` parameterizes the modeled protein from
`structure.prepare_reactive_coordinates` with its ligands (Sage 2.3.0 and AshGC for the
ligands, ff14SB for the protein). It then relaxes the unsolvated model in vacuum with
every observed heavy atom and every ligand held exactly, and the residues declared as
modeled tied by springs to their placed positions. With `solvent: tip3p` it adds a water
box with neutralizing ions and minimizes the solvent and the hydrogens with all solute
heavy atoms still held. A held atom that moved is an error, checked for exact equality of
coordinates. The result reports how far the modeled residues moved. The relaxation is an
energy minimization that reduces steric clashes; the strain from the held
atoms remains. A larger spring constant holds
the modeled residues nearer their placed positions and a smaller one permits larger
displacement; the spring constant is set in the request and recorded. The result also
gives the positions of the ligands in the topology, which `structure.prepare_finite_environment`
and the QM/MM capabilities take.

`structure.prepare_finite_environment` extracts a finite droplet from a periodic system.
Whole waters and ions with any atom inside a sphere of the requested radius around an
anchor are retained together with the solute residues named in the request, and every
force-field parameter is checked to equal that of the source. The boundary consists of two
explicit potentials: springs on the outer heavy atoms of the solute and a wall acting on
the center of mass of each water and ion. Atoms listed in `protected_atoms`, for example
the quantum region, are never restrained. The wall and the springs approximate the bulk
solvent outside the droplet; dielectric screening, the density at the edge and the net
charge of the droplet differ from those of the periodic system. The quantum region must
lie inside the sphere and away from its surface. The researcher sets
`solvent_radius_angstrom` and `inner_solute_radius_angstrom` so that it does; the
capability refuses a protected atom that belongs to a restrained residue. A cluster of
more than 100,000 atoms is refused. The cell must be a cube or other orthorhombic box.

## Evaluating and sampling QM/MM

`simulation.evaluate_qmmm_state` solves a quantum region (atoms listed in the request) with
closed-shell PBE/6-31G** in PySCF in the field of the Gaussian-smeared charges of the
classical atoms, capping a single cut covalent bond with a hydrogen, and returns the
electronic, classical and external energies and the force on every atom. The convergence of
the self-consistent field, the electron count and the stationarity of the orbitals are
checked: the energy tolerance is 1e-11 hartree and the orbital-gradient tolerance is 1e-9
with at most 200 cycles. The electron count must equal the expected count within 1e-7.
Before forces are stored, the norm of the final orbital gradient, recomputed from the
converged orbitals, must be at most 1e-7. The energy is that of PBE in a double-zeta basis,
without dispersion or exact exchange, with the classical charges fixed during the
self-consistent-field solution. The quantum/classical partition and the hydrogen cap are set
in the request and change the energy. The capability applies to closed-shell systems on the
CPU, and the system must be a molecule with all-pair nonbonded terms (no cutoff and no
cell).

`simulation.run_qmmm_dynamics` runs Langevin dynamics with an electronic solution at each
step, optionally with a harmonic umbrella on the difference of two interatomic distances.
The length of a trajectory is `production_steps` times `timestep_fs` (defaults 10 steps and
0.5 fs); each step requires one self-consistent-field solution. An error estimate requires
independently seeded replicas, and the first steps of a trajectory reflect the starting
geometry; equilibration steps run before production are not recorded (`equilibration_steps`,
default 0). The kinetic energy is the half-step value of the OpenMM LangevinMiddle
integrator, so the total energy of exact dynamics is not constant. The result reports
`total_energy_drift_kj_mol` (last production total energy minus the first), which includes
this effect. In the test with friction 1e-4 ps^-1, the spread of the total energy was lower
at a timestep of 0.125 fs than at 0.25 fs.

## Reaction free energy

`simulation.analyze_reaction_free_energy` takes umbrella windows (each a QM/MM run with a
bias on the same coordinate, centered at its own value) in 2 to 20 independently seeded
replicas. For each replica it decorrelates the windows with a stride derived from the
statistical inefficiency, combines them with MBAR against the unbiased state and computes
the potential of mean force from the density of the coordinate in the bins of the request,
relative to a reference bin in the reactant basin. It gives the reactant-to-product
free-energy difference, the height of the transition bins above the reference bin, and, when
requested, a one-way flux through a dividing surface as a transition-state rate with
transmission factor 1. Time blocks of each replica are analyzed separately, and a block for
which the profile cannot be reconstructed is stored in the record as a failure.

The capability refuses windows when the graph of window pairs with an overlap above 1e-12
is not connected. The result reports the effective sample size of each state, the
agreement between replicas and the agreement between time blocks, which are the
diagnostics of convergence. The asymptotic uncertainty assumes decorrelated samples and
applies when the relevant motions are faster than the runs. The profile is a function of
one chosen coordinate and applies when motions orthogonal to it relax within the runs. The
reported uncertainty is statistical; a barrier computed with PBE/6-31G** includes the
error of the functional. The capability refuses windows that do not share a Hamiltonian, temperature
and coordinate, and windows that reuse a random seed.

The bias in every window is `K (q - center)^2` on the difference q of two distances, with K
in kJ/mol per square ångström. A bias written as (1/2) k (q - center)^2 in a source corresponds to
K = k/2. The sign convention of q (distance 1 minus distance 2) and the factor 1/2
determine the bias applied; entering k instead of K doubles the bias. The
profile is the density of q, as -RT times the logarithm of the probability per ångström,
relative to the reference bin, and the basin difference is -RT times the logarithm of the
ratio of the integrated product and reactant probabilities.

The optional rate assumes a classical canonical distribution, a constant inverse mass Z of
the coordinate and a transmission factor of 1. The request supplies Z in
`inverse_mass_dalton` (1/dalton). For a difference of distances between 4 distinct atoms
that are free to move under the constraints, Z equals the sum of the inverse masses of
those 4 atoms. A coordinate in which two distances share an atom has a metric that
depends on geometry, which is outside this estimate. With the density at the dividing
surface rho and the integrated reactant probability P, the one-way velocity is
sqrt(R T Z / 2 pi) and the rate is (rho / P) times that velocity. The density rho is in
Å^-1, the velocity in Å/ps and the rate in 1/s. The activation free
energy is referenced to the Eyring frequency kB T / h. The rate depends on the width of
the surface bin and on the extent of the reactant basin, both set in the request;
refining the bins tests the sensitivity. The rate is a transition-state estimate that assumes no
recrossing and neglects nuclear quantum effects.

## Limits

A calculated difference or barrier is a property of the force field, charge model or
functional, and protocol used, until it is compared with a measurement;
`simulation.analyze_binding_free_energy` takes the measurement as a reference set.

The test suite checks that a ligand transformed into an identical copy of itself gives a
free-energy difference of zero (`simulation.run_relative_free_energy`; runs only when the
engine environments are configured), that an analytic harmonic profile is recovered
(`simulation.analyze_reaction_free_energy`), that finite-difference forces match analytic
forces (`simulation.evaluate_qmmm_state`) and that held atoms do not move
(`structure.prepare_protein_ligand_system`).
