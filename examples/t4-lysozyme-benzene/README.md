# T4 lysozyme and benzene: from a protein to dynamics

**Question.** Does benzene stay in the engineered cavity of T4 lysozyme L99A, in the pose of
the deposited structure 181L, when the complex predicted from the sequence is simulated in
water?

The campaign has 7 steps in two branches. The sequence read by `dossier` is bound into the
prediction of `complex`, which leads through `system` and `dynamics` to the two measurements.
`pockets` is a second branch that reads the deposited structure.

| Step | Capability | Role |
| --- | --- | --- |
| `dossier` | `protein.inspect` | reads the sequence and the deposited structure 181L; its `sequence` is the input of `complex` |
| `pockets` | `structure.detect_binding_pockets` | predicts the pockets of the deposited structure (P2Rank) |
| `complex` | `structure.predict_biomolecular_complex` | predicts the complex of the sequence with benzene (Boltz-2, one sample, seed 1) |
| `system` | `structure.prepare_predicted_complex_for_dynamics` | solvates the predicted complex in a rhombic dodecahedron of water, assigns the force fields (OpenFF) and minimizes |
| `dynamics` | `simulation.run_molecular_dynamics` | equilibrates and runs 10 ns of NPT dynamics (OpenMM) |
| `pose` | `simulation.measure_ligand_pose` | measures the heavy-atom RMSD of benzene to its pose in 181L after fitting the protein by the pocket |
| `contacts` | `simulation.inspect_trajectory` | the fraction of frames in which each residue lies within 4 Å of benzene |

```bash
alambiq run examples/t4-lysozyme-benzene/campaign.yaml --report report.html
```

**Needs.** The `dossier`, `trajectory` and `chemistry` extras, the P2Rank, Boltz-2, OpenFF
and OpenMM environments (`ALAMBIQ_ENV_P2RANK`, `ALAMBIQ_ENV_BOLTZ`, `ALAMBIQ_ENV_OPENFF`,
`ALAMBIQ_ENV_OPENMM`; see [install](../../docs/install.md)) and their resources. The
campaign is written for a machine with a CUDA GPU. On a Mac, set `accelerator: mps` in
`complex` and `platform: CPU` in `dynamics`, and shorten `production_ps`; 10 ns on a CPU takes
days. The test suite does not run it.

**Run on one NVIDIA GB10 (the DGX Spark).** Wall time per step: `dossier` 0.02 s,
`pockets` 3 s, `complex` 57 s, `system` 211 s, `dynamics` 5,167 s (86 min), `pose` 15 s,
`contacts` 14 s; 5,452 s for the campaign. Two identical campaigns shared the GPU in this run;
alone on the GPU (the seed 2 run below), `dynamics` took 2,067 s (34 min). `system` minimizes on one CPU thread. The
run directory holds about 570 MB.

**What to look at.**

- **`complex`**: the confidence of the predicted complex (complex pLDDT 0.978, ipTM 0.973). The
  prediction is a model, and `pose` compares what dynamics makes of it with the experiment.
- **`pose`**: the 1,000 saved frames span about 10 ns. The heavy-atom RMSD of benzene to the
  pose in 181L has a mean of 0.80 Å (median 0.79 Å, maximum 1.57 Å), and the RMSD of all 1,000
  frames lies below the 2 Å threshold. The 6 heavy atoms have 12 symmetry-equivalent
  pairings, and the RMSD is the smallest over them. The mean RMSD of the 54 fit atoms is 0.53 Å.
- **`contacts`**: the residues with the highest contact fraction are Ala99 (0.996), Val111
  (0.93), Leu118 (0.90), Val87 (0.79) and Leu84 (0.74). Ala99 is the residue that replaces
  leucine in the L99A cavity.
- **`pockets`**: P2Rank ranks the cavity second (score 4.18, probability 0.17), not first. Its
  residues include Leu84, Val87, Leu91, Ala99, Val111, Leu118, Leu121 and Phe153, and each
  has a contact fraction above 0.3 in `contacts`.

Change a downstream parameter, for example `threshold_angstrom` of `pose`, and rerun with the
same `--run-dir`. Only `pose` is computed again; the other steps are reused. With a threshold
of 1 Å, `pose` ran again in 3.7 s, 873 of the 1,000 frames lay within the threshold, and the
first frame beyond it was frame 43; `dossier`, `pockets`, `complex`, `system`, `dynamics` and
`contacts` were reused.

Change `seed` of `complex` and rerun with the same `--run-dir`. `dossier` and `pockets` are
reused, and `complex` and every step after it are computed again. With seed 2 the confidence
was ipTM 0.972, the RMSD of benzene had a mean of 0.78 Å and a maximum of 1.45 Å, all 1,000
frames lay below 2 Å, and the order of the five residues with the highest contact fraction was
the same (Ala99 0.999, Val111 0.94, Leu118 0.87, Val87 0.76, Leu84 0.755).

**Limits.**

- The RMSD states how far the atoms lie from the pose in 181L over the trajectory. It does not
  state a binding free energy or a residence time.
- The numbers above are one prediction sample (seed 1) and one 10 ns trajectory of one
  starting structure. Another seed gives other values.
- A rerun of the same request gives the same result. Two campaigns with seed 1 in separate run
  directories wrote the same predicted complex, the same minimized system (`state.xml` and
  `topology.pdb` byte for byte) and the same 1,000 frames (the largest coordinate difference
  is 0.0 Å at the precision of the trajectory file); the energies in the state data agree to a
  relative 7e-15. This holds under the defaults: `system` minimizes on one CPU thread with
  deterministic forces (with more threads the order of the force sums varies and the minimized
  structure differs), and `dynamics` on CUDA uses mixed precision with deterministic forces.
  It was checked on one GB10; another device or driver may differ in the last digits.
- 181L, or a close homologue, may be among the training data of Boltz-2, so a prediction that
  agrees with 181L does not show that the model predicts new complexes.
- The 10 ns of dynamics start from the predicted pose. They show that the pose is stable in
  this force field, and the stability is not evidence that the pose is the true one.
- The simulation covers 10 ns. A ligand that leaves the cavity on a longer time scale is not
  represented.
- The pose in 181L is one experimental structure of the complex, at the resolution of the
  deposited model.

**Data.** `181L.cif` and `181L.pdb` are the deposited structure of T4 lysozyme L99A with
benzene (RCSB PDB 181L, CC0; Morton and Matthews, Biochemistry 1995, doi:10.1021/bi00027a007). `t4l.fasta` is its
sequence. `181L.pdb` is the same structure in PDB format, which `pose` reads as the
reference.
