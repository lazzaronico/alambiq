# T4 lysozyme and benzene: from a protein to dynamics

**Question.** Does benzene stay in the engineered cavity of T4 lysozyme L99A, in the pose of
the deposited structure 181L, when the complex predicted from the sequence is simulated in
water?

The campaign has 7 steps:

| Step | Capability | Role |
| --- | --- | --- |
| `dossier` | `protein.inspect` | reads the sequence and the deposited structure 181L |
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

**Run on one NVIDIA GB10 (the DGX Spark).** Wall time per step: `dossier` 0.04 s,
`pockets` 3 s, `complex` 75 s, `system` 155 s, `dynamics` 4,846 s (81 min), `pose` 14 s,
`contacts` 14 s. The run directory holds 572 MB.

**What to look at.**

- **`complex`**: the confidence of the predicted complex (complex pLDDT 0.98, ipTM 0.97). The
  prediction is a model, and `pose` compares what dynamics makes of it with the experiment.
- **`pose`**: the 1,000 saved frames span about 10 ns. The heavy-atom RMSD of benzene to the
  pose in 181L has a mean of 0.79 Å (median 0.79 Å, maximum 1.60 Å), and the RMSD of all 1,000
  frames lies below the 2 Å threshold. The 6 heavy atoms have 12 symmetry-equivalent
  pairings, and the RMSD is the smallest over them. The mean RMSD of the 54 fit atoms is 0.52 Å.
- **`contacts`**: the residues with the highest contact fraction are Ala99 (0.999), Val111
  (0.94), Leu118 (0.87), Val87 (0.81) and Leu84 (0.70). Ala99 is the residue that replaces
  leucine in the L99A cavity.
- **`pockets`**: P2Rank ranks the cavity second (score 4.18, probability 0.17), not first. Its
  residues include Leu84, Val87, Leu91, Ala99, Val111, Leu118, Leu121 and Phe153, which
  are the residues of the `contacts` table with a contact fraction above 0.4.

Change a downstream parameter, for example `threshold_angstrom` of `pose`, and rerun with the
same `--run-dir`. Only `pose` is computed again; the other steps are reused.

**Limits.**

- The RMSD states how far the atoms lie from the pose in 181L over the trajectory. It does not
  state a binding free energy or a residence time.
- The numbers above are one prediction sample (seed 1) and one 10 ns trajectory of one
  starting structure. Another seed or trajectory gives other values.
- The simulation covers 10 ns. A ligand that leaves the cavity on a longer time scale is not
  represented.
- The pose in 181L is one experimental structure of the complex, at the resolution of the
  deposited model.

**Data.** `181L.cif` and `181L.pdb` are the deposited structure of T4 lysozyme L99A with
benzene (RCSB PDB 181L, CC0; Morton and Matthews, Biochemistry 1995, doi:10.1021/bi00027a007). `t4l.fasta` is its
sequence. `181L.pdb` is the same structure in PDB format, which `pose` reads as the
reference.
