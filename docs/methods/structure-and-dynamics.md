# Structure and dynamics

Comparison, prediction and preparation of structures, molecular dynamics, and the
inspection of a trajectory. The capabilities are `structure.compare`,
`structure.detect_binding_pockets`, `structure.predict_biomolecular_complex`,
`structure.prepare_protein_system`, `structure.prepare_predicted_complex_for_dynamics`,
`simulation.run_molecular_dynamics`, `simulation.run_metadynamics`,
`simulation.inspect_trajectory`, `simulation.compare_scalar_couplings` and
`simulation.compare_replica_structures`. The preparation capabilities write a system that
`simulation.run_molecular_dynamics` or `simulation.run_metadynamics` reads.
`simulation.inspect_trajectory` reads a trajectory. `simulation.compare_scalar_couplings`
and `simulation.compare_replica_structures` read the results for several replicas.
`structure.compare` (which reads stored `protein.inspect` records),
`structure.detect_binding_pockets` and `structure.predict_biomolecular_complex` are
independent of these.

## Comparing and finding sites

`structure.compare` superposes one chain on another over the residues paired in the
request (a list, an alignment, or the same query) and reports the RMSD before and after,
the distance of every pair, and the fraction of each protein that was paired. The RMSD
depends on the paired residues and is reported with the coverage.

`structure.detect_binding_pockets` runs P2Rank on a structure and ranks surface pockets
by the P2Rank score, the output of a model trained on known ligand-binding sites. A
pocket is a surface feature, and the score orders the pockets of one structure.

## Predicting a complex

`structure.predict_biomolecular_complex` runs Boltz-2 on 1 to 8 protein chains and at most 1
ligand, and returns sampled structures with the confidence values of the model. These values
rank the samples against each other. A chain without an alignment runs in single-sequence
mode, in which the model is less accurate. A run with fewer than 50 diffusion steps per
sample or fewer than 2 recycling passes (defaults 200 and 3) is a pipeline test; the
capability states this in its notes. The capability reads the ligand chemistry from the
molecule that Boltz processed and refuses the prediction if its canonical SMILES differs
from that of the supplied SMILES with hydrogens removed. The ligand of each sample is
written as an SDF file with bonds and charges from the SMILES and positions from the
prediction, matched by atom name. The stereochemistry of the ligand pose in each sample is
computed from its coordinates and reported. A sample whose ligand pose differs in
stereochemistry from the supplied SMILES is flagged in the notes and kept in the result. A
co-folded ligand pose is a hypothesis about the ligand position.

The model files are pinned and verified by the capability before a run; a missing file
fails the step. On Apple silicon, GPU execution requires a patched Boltz in a separate
engine environment, selected by `accelerator: mps`; the capability runs on the device named
by `accelerator`.

## Preparing a system

`structure.prepare_protein_system` takes one chain of a deposited structure and adds
hydrogens, terminal oxygens, the missing heavy atoms that the request allows and the
substitutions that the request names. It then solvates the chain in TIP3P with
neutralizing ions or leaves it in implicit solvent, and minimizes it. The geometry diagnostics
(peptide planarity, alpha-carbon chirality) do not change a coordinate. A twisted
peptide bond or an inverted alpha carbon in the input remains in the prepared system, and
dynamics starts from it.

The periodic cell is sized by the convention of OpenMM: the diameter of the bounding sphere
of the solute plus the padding, which places every solute atom at least the padding away
from its own periodic copy. The padding is added once, to the diameter. The default cell is
a rhombic dodecahedron with 2 nm of padding. Its volume is about 70.7 % of that of a cube
with the same padding. When the protein comes within the cutoff of its own periodic image,
it interacts with its image; the distance to the periodic image determines whether the cell
is large enough. In a recorded 50 ns run of GB3 with 3 replicas, 1 nm of padding and a 0.9
nm cutoff, the protein came within 0.9 nm of its periodic image in 231, 392 and 555 of 5,000
saved frames (minimum distances 0.57, 0.55 and 0.32 nm), while the mean temperature was
298.2 to 298.3 K (target 298 K) and the mean alpha-carbon RMSD was 0.98 to 1.29 Å
(`environments/openmm/verification.gb3-50ns-padding1-failure-20261003.json`). The separation
during the run is measured with the periodic image check of `simulation.inspect_trajectory`;
the padding of 2 nm applies to the starting structure.

Implicit solvent (GBn2 with ff14SB) replaces the water with a continuum without a cutoff.
The ionic strength I, the temperature T and the solvent dielectric determine a screening
constant, 367.434915 sqrt(I / (dielectric T)) nm^-1 with I in mol/L and T in K, with
dielectrics of 1 for the solute and 78.5 for the solvent unless set in the request. This
models screening. Simulated time under GBn2 is reported as simulated, without conversion to
explicit-water time. Both solvent models use ff14SB, so agreement between them compares two
computations with the same force field.

`structure.prepare_predicted_complex_for_dynamics` prepares a predicted protein and its
ligand as `structure.prepare_protein_system` prepares a chain, with Sage 2.3.0 parameters
and AshGC charges for the ligand and ff14SB for the protein. It refuses a residue that it
does not handle, a missing atom, an ambiguous disulfide distance, a ligand pose
whose recorded source digest differs from the structure file, a ligand whose formal
charge differs from the declared charge and a ligand whose stereochemistry differs from the
supplied SMILES. A finite, lower energy after minimization shows that the model can be built.

## Running dynamics

`simulation.run_molecular_dynamics` runs the phases set in the request: minimization,
velocities, restrained thermalization, release under the barostat, readiness windows and
production. The request sets the length of each phase in picoseconds. A readiness window
checks 4 blocks of unrestrained dynamics for drift in temperature, density and potential
energy. Windows repeat until one passes, up to `max_readiness_windows` (default 3). When the
last window fails and `require_ready` is true (default), production is not run. Density is
assessed only for periodic systems. A passed window applies to the solvent and the
thermostat. A window has at least 20 state samples in 4 blocks and passes when:

- the mean temperature is within 5 % of the target;
- the block means of temperature differ by at most 3 % of the target;
- the block means of density differ by at most 2 % of their mean;
- the block means of potential energy differ by at most 0.1 kJ/mol per atom.

These thresholds detect drift larger than the stated tolerances and are recorded in the
result.

A single trajectory is one sample; an error estimate requires replicas. A trajectory samples
motions with time scales shorter than its length. On one CPU thread (the default), the same
seed gave the same trajectory in the test of an implicit-solvent run. With more threads or
on another platform, runs agree only statistically.

`simulation.run_metadynamics` biases 1 or 2 torsions and then fixes the bias for production.
The fixed bias permits reweighting of the production samples to the unbiased distribution;
the result reports the unbiased free-energy surface and an effective sample size. The frames
are samples of the biased ensemble; equilibrium properties are computed from the reweighted
samples. The bias accelerates transitions, so the number of transitions per unit of
simulated time is a property of the biased run; the result provides the free-energy surface.
The effective sample size measures how concentrated the weights are on few frames;
correlation in time, states that were never visited and variation between independent
replicas require separate analysis.

## Trajectory inspection

`simulation.inspect_trajectory` reads the trajectory once and reports the
alpha-carbon RMSD and radius of gyration of every frame after fitting, the per-residue
fluctuation, the distances, angles and contacts named in the request by exact atom, and
optionally ligand contacts and the distance from the protein to its own periodic image. A
molecule that crosses the periodic box edge has its atoms on opposite sides of the box
unless whole-molecule imaging is requested. Distances and contacts use the stored
coordinates unless `periodic_observables` is set. The periodic image check covers
`protein_image_check.frames` evenly spaced saved frames. The fit uses alpha carbons only,
so the RMSD reports backbone motion; motion of side chains or of a ligand appears in the
named distances, angles and contacts. A rise in RMSD shows that the structure moved away from
the reference.

## Comparison with experiment and with other replicas

`simulation.compare_scalar_couplings` computes three-bond couplings from a Karplus relation
on the dihedral of the unique bonded path between two atoms, averages equally over frames
and then over replicas, and reports the result beside the measurement and its reported
error. The residual (calculated minus measured value) combines errors in the force field,
the sampling, the Karplus coefficients and the measurement. The coefficients are empirical
and their own errors are not propagated. They are applied to the dihedral of the four atoms
as computed. Coefficients published for a dihedral offset by a constant angle (for example
some phi relations) require conversion before entry. β-methylene hydrogens named HB2 and HB3
can be assigned by geometry (the IUPAC convention) instead of by the labels of the topology,
because force fields label them differently.

`simulation.compare_replica_structures` judges whether independent replicas determine the
mean of each named observable to the precision set in the request. For each observable it
gives the mean, the standard error from the spread of the replica means, the split R-hat,
the effective sample size and the drift between halves. It compares the standard error with
the precision set in the request, and judges whether the 95 % interval of the mean (t
quantile with n - 1 degrees of freedom for n replicas) lies within the tolerance set in the
request around the value computed on a reference structure. The standard error is estimated
from the replica means, so its precision depends on the number of replicas. Agreement among
replicas applies to the sampled time; replicas can all remain in the same region. A
checkpoint is an initial portion of every replica (a number of frames per replica) on which
the precision is judged again; the checkpoints show how the precision changes within the
available frames.
