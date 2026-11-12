# Structure and dynamics

These capabilities compare and predict structures, prepare a system, run molecular dynamics
and inspect trajectories.

`structure.compare` (which reads stored `protein.inspect` records),
`structure.measure_ligand_site`, `structure.fetch`, `structure.check`, `structure.assign_protonation`,
`structure.detect_binding_pockets` and `structure.predict_biomolecular_complex` are
independent of the others. `structure.prepare_protein_system` and
`structure.prepare_predicted_complex_for_dynamics` write a system that
`simulation.run_molecular_dynamics` or `simulation.run_metadynamics` reads.
`simulation.inspect_trajectory`, `simulation.measure_ligand_pose` and
`simulation.cluster_trajectory` read a trajectory.
`simulation.compare_scalar_couplings`
and `simulation.compare_replica_structures` read the results for several replicas.

## Fetching a structure

`structure.fetch` downloads one structure file: a deposited entry from the RCSB Protein Data Bank
(by PDB entry identifier) or a predicted model from the AlphaFold Protein Structure Database (by
UniProt accession), in mmCIF or PDB format. Its result is the stored file with the URL, the time
of retrieval, the size and the SHA-256, so a later run can identify the exact file used. The step
connects to the network only when the request sets `allow_network: true`, which keeps a campaign
without that setting offline. The AlphaFold route reads the file URL from the prediction listing,
and the result records that URL, which carries the model version. `base_url` names a mirror that
serves the same paths. The archives revise entries and models over time; the SHA-256 identifies
the file that a result used. Terms of use and citation requirements belong to the archive.

## Checking a structure

`structure.check` reads one model of a PDB or mmCIF file and reports what is missing, doubled,
broken or crowded: heavy atoms absent from a standard residue, alternate locations, atoms of
zero occupancy, nonstandard amino acids, chain breaks, cis and twisted peptide bonds, alpha
carbons off the L arrangement, close contacts between heavy atoms, and the ligands, metal ions,
waters and hydrogens. The capability changes nothing. The thresholds (the C–N distance of a
break, the deviation of omega, the contact distance) are request fields, and each finding
carries its measured value, so a reader can apply a different threshold to the same file.

The report is a first look before `structure.prepare_protein_system`, which refuses ligands,
metals, other chains and nonstandard residues or drops them, and rebuilds missing heavy atoms
only when asked. A flagged residue can be real: a cis proline, a strained loop or a region of
lower resolution. Contacts with symmetry mates are not searched.

## Protonating a structure

`structure.assign_protonation` sets the protonation states of a protein at a pH and places its
hydrogen atoms. PROPKA 3.5 predicts a pKa for each aspartate, glutamate, arginine, lysine,
histidine, cysteine and tyrosine and for the chain termini. PDB2PQR 3.7 protonates each group
whose pKa is above the pH, names the residue as Amber does (`ASH`, `GLH`, `LYN`, `HIP`, `CYX`),
adds the hydrogens and removes bumps. It then optimizes the hydrogen-bond network. The
optimization picks the tautomer of each neutral histidine and may turn the amide of an
asparagine or glutamine, or the ring of a histidine, by 180 degrees. The two positions of such
a group cannot be told apart in an X-ray map, since oxygen and nitrogen scatter alike, so the
file's orientation is one of two equally supported by the density. The result lists the
residues that turned, the heavy atoms that were rebuilt, and the pKa of every group with the pKa
that PROPKA takes for the group free in water.

The pKa values are empirical estimates from one structure. A group whose state matters, such as
the histidine of an active site, is worth reading in the list. The state assigned at one pH
holds throughout a simulation that follows; a simulation with fixed protonation does not
titrate. `structure.prepare_protein_system` chooses variants with OpenMM Modeller from the pH and the
standard pKa of each residue type, without regard to the environment of the residue in the
structure; this capability gives a structure-based estimate for the same decision. Both `structure.prepare_protein_system` and
`structure.prepare_mutation_inputs` take the protonated structure as `protonated_structure`: its
residue names become the protonation of the titratable residues of the chain, so that a campaign
makes the decision once. A `residue_variants` entry of `structure.prepare_protein_system` still
overrides it. The numbers of the residues in the protonated structure must be those of the
structure it was made from.

## Comparing and finding sites

`structure.compare` superposes one chain on another over the residues paired in the request
(a list, an alignment, or the same query). It reports the RMSD before and after, the
distance of every pair and the fraction of each protein that was paired. The RMSD depends on
the paired residues and is reported with the coverage.

`structure.measure_ligand_site` superposes a structure that holds a ligand (a predicted
complex) on a reference (a deposited structure) over all protein chains, and measures the
distance from the ligand to chosen atoms of the reference, for example from the carbonyl
oxygen of a substrate to the catalytic zinc or to the hydride-accepting carbon of the
cofactor. Residues correspond by author number and name. The chains are paired by the
assignment with the smallest alpha-carbon RMSD, which matters for an oligomer, whose
predicted chains carry names unrelated to those of the crystal. When several assignments fit
within a tolerance (a nearly symmetric oligomer), the ligand lies in a different subunit
site in each, and the result gives the range of every nearest distance over them. The
distances place the ligand relative to atoms that the prediction does not contain; they do
not show that the pose is the productive binding mode. The result also gives, for each
ligand atom, the nearest heavy atom of the whole reference (waters excluded), so that a ligand
placed on a cofactor or inside the protein shows as a distance far below the contact distance.
A prediction that holds the cofactor and the metal can also be measured against itself:
`own_targets` name atoms of the mobile structure, and `own_contacts` gives the nearest atom of
the mobile structure outside the ligand to each ligand atom, so the pose is checked against the
cofactor that the prediction placed as well as against the one in the deposited structure.

`structure.detect_binding_pockets` runs P2Rank on a structure and ranks surface pockets by
the P2Rank score, the output of a model trained on known ligand-binding sites. A pocket is a
surface feature, and the score orders the pockets of one structure.

## Predicting a complex

`structure.predict_biomolecular_complex` runs Boltz-2 on 1 to 8 protein chains, at most 1
ligand and optional cofactors. It returns sampled structures with the confidence values of the model, which rank
the samples against each other. A co-folded ligand pose is a hypothesis about the ligand
position.

A cofactor or ion is named by its code in the Chemical Component Dictionary (`NAP`, `ZN`)
with a number of copies, for example one per subunit of a tetramer. Each copy is a chain of
its own, and the ligand is placed with the cofactors present. Without them the model places a
ligand where the cofactor would be, which `structure.measure_ligand_site` shows as an overlap
with the cofactor of a deposited structure.

A chain without an alignment runs in single-sequence mode, in which the model is less
accurate. A run with fewer than 50 diffusion steps per sample or fewer than 2 recycling
passes (defaults 200 and 3) is a pipeline test; the capability states this in its notes.

The capability reads the ligand chemistry from the molecule that Boltz processed. It refuses
the prediction if the canonical SMILES of that molecule differs from that of the supplied
SMILES with hydrogens removed. The ligand of each sample is written as an SDF file with
bonds and charges from the SMILES and positions from the prediction, matched by atom name.
The stereochemistry of each ligand pose is computed from its coordinates and reported. A
sample whose ligand pose differs in stereochemistry from the supplied SMILES is flagged in
the notes and kept in the result.

Each chain and the ligand may carry a name; one without a name takes the next unused capital
letter (the chains first, then the ligand). The confidence scores are the model's estimates of
the accuracy of a structure and are not binding affinities. An affinity exists only for a
request with a ligand: two protein chains, such as an antibody and its antigen, return a structure
and its confidence and no affinity.

With `affinity`, the capability also returns Boltz-2's estimate of how strongly the ligand
binds: the log10 of an IC50 in µM (lower is stronger; 0 is 1 µM), the same estimate as pIC50
(6 minus that value; higher is stronger) and the probability that the ligand binds, for the
ensemble of two models and for each model. Boltz standardizes the ligand
first with the ChEMBL structure pipeline (largest fragment, neutralization); the structures
and the estimate then refer to the standardized molecule, and a changed molecule is named in
the notes. Following the Boltz documentation, the probability separates binders from decoys
and the IC50 value compares active molecules. The ligand has at most 128 atoms (heavy atoms
and the hydrogens that RDKit keeps); above 56 atoms the result carries a note.

The capability verifies the pinned model files before a run; a missing file fails the step.
It runs on the device named by `accelerator`. On Apple silicon, GPU execution
(`accelerator: mps`) requires a patched Boltz in a separate engine environment.

## Preparing a system

`structure.prepare_protein_system` takes one chain of a deposited structure and adds
hydrogens, terminal oxygens, the missing heavy atoms that the request allows and the
substitutions that the request names. It then solvates the chain in TIP3P with neutralizing
ions or leaves it in implicit solvent, and minimizes it. The geometry diagnostics (peptide
planarity, alpha-carbon chirality) do not change a coordinate. A twisted peptide bond or an
inverted alpha carbon in the input remains in the prepared system, and dynamics starts from
it.

The periodic cell is sized by the convention of OpenMM: the diameter of the bounding sphere
of the solute plus the padding. This places every solute atom at least the padding away
from its own periodic copy. The padding is added once, to the diameter. The default cell is
a rhombic dodecahedron with 2 nm of padding. Its volume is about 70.7 % of that of a cube
with the same padding.

The distance to the periodic image determines whether the cell is large enough. When the
protein comes within the cutoff of its own periodic image, it interacts with its image. In
a recorded 50 ns run of GB3 with 3 replicas, 1 nm of padding and a 0.9 nm cutoff, the
protein came within 0.9 nm of its periodic image in 231, 392 and 555 of 5,000 saved frames
(minimum distances 0.57, 0.55 and 0.32 nm). The mean temperature was 298.2 to 298.3 K
(target 298 K) and the mean alpha-carbon RMSD was 0.98 to 1.29 Å
(`environments/openmm/verification.gb3-50ns-padding1-failure-20261003.json`). The periodic
image check of `simulation.inspect_trajectory` measures the separation during the run; the
padding of 2 nm applies to the starting structure.

Implicit solvent (GBn2 with ff14SB) replaces the water with a continuum without a cutoff.
The ionic strength I, the temperature T and the solvent dielectric determine a screening
constant, 367.434915 sqrt(I / (dielectric T)) nm^-1, with I in mol/L and T in K. The
dielectrics are 1 for the solute and 78.5 for the solvent unless set in the request. The
constant models screening. Simulated time under GBn2 is reported as simulated, without
conversion to explicit-water time. Both solvent models use ff14SB, so agreement between
them compares two computations with the same force field.

`structure.prepare_predicted_complex_for_dynamics` prepares a predicted protein and its
ligand as `structure.prepare_protein_system` prepares a chain. It uses Sage 2.3.0 parameters
and AshGC charges for the ligand and ff14SB for the protein. It refuses:

- a residue that it does not handle;
- a missing atom;
- an ambiguous disulfide distance;
- a ligand pose whose recorded source digest differs from the structure file;
- a ligand whose formal charge differs from the declared charge;
- a ligand whose stereochemistry differs from the supplied SMILES.

A finite, lower energy after minimization shows that the model can be built.

## Running dynamics

`simulation.run_molecular_dynamics` runs the phases set in the request: minimization,
velocities, restrained thermalization, release under the barostat, readiness windows and
production. The request sets the length of each phase in picoseconds.

A readiness window checks 4 blocks of unrestrained dynamics for drift in temperature,
density and potential energy. Windows repeat until one passes, up to `max_readiness_windows`
(default 3). When the last window fails and `require_ready` is true (default), production is
not run. Density is assessed only for periodic systems. A passed window applies to the
solvent and the thermostat. A window has at least 20 state samples in 4 blocks and passes
when:

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
The fixed bias permits reweighting of the production samples to the unbiased distribution.
The result reports the unbiased free-energy surface and an effective sample size. The frames
are samples of the biased ensemble, so equilibrium properties come from the reweighted
samples. The bias accelerates transitions. The number of transitions per unit of simulated
time is a property of the biased run, and the free-energy surface is the result.
The effective sample size measures how concentrated the weights are on few frames.
Correlation in time, states that were never visited and variation between independent
replicas require separate analysis.

## Trajectory inspection

`simulation.inspect_trajectory` reads the trajectory once and reports:

- the alpha-carbon RMSD and radius of gyration of every frame after fitting;
- the per-residue fluctuation;
- the distances, angles and contacts named in the request by exact atom;
- optionally, ligand contacts and the distance from the protein to its own periodic image.

A molecule that crosses the periodic box edge has its atoms on opposite sides of the box
unless whole-molecule imaging is requested. Distances and contacts use the stored
coordinates unless `periodic_observables` is set. The periodic image check covers
`protein_image_check.frames` evenly spaced saved frames.

The fit uses alpha carbons only, so the RMSD reports backbone motion. Motion of side chains
or of a ligand appears in the named distances, angles and contacts. A rise in RMSD shows
that the structure moved away from the reference.

## Measuring a ligand pose

`simulation.measure_ligand_pose` reads the trajectory once and reports, for every frame, how
far the ligand lies from its pose in a reference structure.

- The protein is fitted to the reference by the alpha carbons of the residues that have an atom
  within `fit_cutoff_angstrom` of the reference ligand.
- The ligand RMSD is in Å, over heavy atoms, and the ligand is not fitted again after the
  protein fit. The atoms of the two files are paired by the molecular graph that `smiles`
  describes, and the smallest RMSD over the pairings that the symmetry of the molecule allows
  is reported. A benzene turned by 60 degrees about its ring normal has an RMSD of 0.
- The centroid shift is the distance between the centroids of the ligand and of the reference
  ligand. A ligand that turns in place has a small centroid shift and a nonzero RMSD.
- The summary gives the fraction of frames within `threshold_angstrom` of the reference pose and
  the first frame beyond it. `state_data` gives the time of each frame.

The reference is a PDB with the ligand in the pose of interest. With a deposited complex as the
reference, the first frame measures the prepared prediction against the experimental pose and
the later frames measure the dynamics. `imaging: whole-molecules` is set when a molecule crossed
the box edge, as for `simulation.inspect_trajectory`.

## Trajectory clustering

`simulation.cluster_trajectory` groups the frames of a trajectory by structural similarity. The
RMSD between two frames is computed over the selected atoms (alpha carbons, backbone or protein
heavy atoms) after the best rigid superposition of the pair. Clustering follows Daura et al.
(1999): the frame with the most neighbors within `rmsd_cutoff_angstrom` is the center of the
first cluster, its neighbors are its members, and the procedure repeats on the remaining frames.
The result gives the size and share of each cluster, the center frame, and the mean and maximum
RMSD of the members to the center. The center structures of the largest clusters are written as
PDB files, for example as starting structures of a docking or QM/MM calculation, and a table
assigns every frame read to a cluster.

The count and the sizes depend on the atom selection and on the cutoff; a cutoff that leaves
most frames in clusters of 1 is reported in a note. The populations count saved frames of one
trajectory and describe the time sampled. At most 5,000 frames are clustered, and `first_frame`
and `stride` thin a longer trajectory. A molecule that crosses the periodic box edge is broken
across it in the stored coordinates unless `imaging: whole-molecules` is set.

## Comparison with experiment and with other replicas

`simulation.compare_scalar_couplings` computes three-bond couplings from a Karplus relation
on the dihedral of the unique bonded path between two atoms. It averages equally over frames
and then over replicas, and reports the result beside the measurement and its reported
error. The residual (calculated minus measured value) combines errors in the force field,
the sampling, the Karplus coefficients and the measurement.

The Karplus coefficients are empirical and their own errors are not propagated. They are
applied to the dihedral of the four atoms as computed. Coefficients published for a
dihedral offset by a constant angle (for example some phi relations) require conversion
before entry. β-methylene hydrogens named HB2 and HB3 can be assigned by geometry (the
IUPAC convention) instead of by the labels of the topology, because force fields label them
differently.

`simulation.compare_replica_structures` judges whether independent replicas determine the
mean of each named observable to the precision set in the request. For each observable it
gives the mean, the standard error from the spread of the replica means, the split R-hat,
the effective sample size and the drift between halves. It compares the standard error with
the precision set in the request. It also judges whether the 95 % interval of the mean (t
quantile with n - 1 degrees of freedom for n replicas) lies within the tolerance set in the
request around the value computed on a reference structure. The standard error is estimated
from the replica means, so its precision depends on the number of replicas.

Agreement among replicas applies to the sampled time; replicas can all remain in the same
region. A checkpoint is an initial portion of every replica (a number of frames per replica)
on which the precision is judged again. The checkpoints show how the precision changes
within the available frames.
