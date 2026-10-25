# OpenMM engine environment

This engine environment contains OpenMM and PDBFixer. `ALAMBIQ_ENV_OPENMM` names its
Python. Use the [Mamba recipe](environment.yaml) for protein preparation and dynamics. In
`structure.prepare_protein_system`, PDBFixer adds the missing heavy atoms that the request
declares. Campaigns with a protein require OpenMM and PDBFixer. PDBFixer also applies the
substitutions of standard residues that the request declares. The capability checks the
source residue identities, records the atoms and bonds that are added and removed, and
exports the mutant as FASTA and mmCIF. See
[protein preparation](../../docs/methods/structure-and-dynamics.md#preparing-a-system). The
[substitution file](verification.protein-substitutions-20261004.json) records checks of the
chemical graph and of the residue lineage, and 3 preparations of BglB (445 residues; wild
type, N220Y, N404A). In all 3 preparations, the resolved protonation states of the
20 histidines are identical, the cell is neutral and the minimum distance between periodic
images of the protein is above 3.22 nm.

Create the environment with:

```bash
mamba env create --prefix /path/to/openmm \
  --file environments/openmm/environment.yaml
```

On a compatible Linux NVIDIA host, add `cuda-version=13.0` to the dependencies in a copy of
this recipe before creation. The section "CUDA on the Spark host" gives two more routes for
CUDA: the explicit conda lock and a pip installation. Use one OpenMM installation route
per prefix.

After either installation route, set `ALAMBIQ_ENV_OPENMM` to the Python of the environment
and check it:

```bash
export ALAMBIQ_ENV_OPENMM=/absolute/path/to/openmm/bin/python
alambiq env check openmm
```

`alambiq env check openmm` reports whether `openmm`, `numpy` and `pdbfixer` import in the
environment. `ALAMBIQ_ENV_OPENMM` is the engine environment of
`structure.prepare_protein_system`, `simulation.run_molecular_dynamics` and
`simulation.run_metadynamics`. The example `examples/ensemble-vs-nmr` uses it: it prepares
GB3 (PDB 1P7E), runs 3 CPU replicas and compares the computed scalar couplings with experiment.
It takes 10 to 25 minutes on a laptop CPU, with 3 single-thread replicas in parallel, and it
needs the `trajectory` and `dossier` extras:

```bash
alambiq run examples/ensemble-vs-nmr/campaign.yaml --report report.html
```

`simulation.inspect_trajectory` runs in the core with the `trajectory` extra.

A prepared system is passed as self-contained OpenMM `System` and `State` XML documents.
Alambiq validates the whole document before it loads it: the root must be `System` or
`State`, a DOCTYPE is rejected, and every nested type must be on a fixed list of OpenMM data
types (force classes, barostats, the Andersen thermostat, the center-of-mass motion remover,
tabulated functions and virtual sites). `PythonForce`, which executes code, is
rejected.

## Metadynamics verification

The torsion metadynamics of `simulation.run_metadynamics` (see
[running dynamics](../../docs/methods/structure-and-dynamics.md#running-dynamics)) uses
OpenMM's metadynamics implementation in this engine environment. Four verification files
record it:

- The [macOS file](verification.metadynamics.osx-arm64.json) records numerical controls of
  the bias, the forces and the weights, and a canonical GB3 run; the controls
  check the implementation.
- The [macOS recovery file](verification.metadynamics-recovery.osx-arm64.json) tests recovery
  of a run after an interruption, from a native state checkpoint paired with the bias
  checkpoint, on the Reference and CPU platforms.
- The [GB10 recovery file](verification.metadynamics-recovery.linux-aarch64.json) records
  recovery from a production checkpoint on CUDA with mixed precision.
- The [PME force and peptide file](verification.metadynamics-cuda-pme-20261004.json) records
  that, with PME and a nonzero torsion bias on CUDA, repeated force queries at identical
  coordinates differ by up to 141.8 kJ/mol/nm with the default setting. With
  `DisablePmeStream: "true"`, the difference is 0. The system is a periodic ACE-ALA-NME
  peptide with 878 TIP3P waters (2,656 particles). CUDA torsion metadynamics requires
  `DisablePmeStream: "true"` in `platform_properties`. The file also records a staged pilot
  run of this peptide with preparation, force, geometry and cost checks.

## Apple Silicon OpenCL environment

The [Apple recipe](environment.apple-ensemble.yaml) selects the official
`py311h0e1cade_0_apple` build of OpenMM 8.6.0 and PDBFixer 1.12.0. Create the environment
with the recipe, or with the explicit lock
[locks/osx-arm64.apple-ensemble.explicit.txt](locks/osx-arm64.apple-ensemble.explicit.txt):

```bash
mamba env create --prefix /path/to/openmm-apple \
  --file environments/openmm/environment.apple-ensemble.yaml
mamba create --prefix /path/to/openmm-apple \
  --file environments/openmm/locks/osx-arm64.apple-ensemble.explicit.txt
```

Use one of the two commands. On the Apple M2 Max, the
OpenCL platform runs with `platform_properties: {Precision: single}`. The GB3 ensemble and a
force comparison between CPU and OpenCL at fixed coordinates were run with this
configuration. The [BglB file](verification.bglb-apple-pocket-20261004.json) checks, with
`UseCpuPme: "false"` and `DisablePmeStream: "true"` on the same build, 4 chemical models of
BglB with forces without and with a nonzero bias, and 5 trajectories of 20 saved frames.

On macOS 26.6.2, the Apple wrapper of the alternative Khronos build failed to load
`_clGetGLContextInfoKHR`. The build for Apple links Apple's OpenCL framework directly. Mixed
precision is not available on this GPU. The probe and the trajectories ran with
`Precision: single`. The verification file records the OpenMM build and the precision; the
Khronos failure and the unavailability of mixed precision were observed during setup outside
the verification files.

## Reanalysis of stored GB3 segments

The [reanalysis file](verification.gb3-retained-stage-reanalysis-20261008.json) records the
reanalysis of 9 completed segments of earlier GB3 runs on macOS ARM64: 3 replicas of 50 ns at
298 K, 3 replicas of 50 ns at 310 K, and the 3 continuations of the 298 K replicas from 50 to
100 ns. The segments contain 45,000 saved frames in total (5,000 per segment).

- The reanalysis reproduces the 6 original coupling arrays (6,750,000 values, computed with
  the atom names of the original topology) bit for bit.
- The geometric IUPAC assignment of the beta-methylene hydrogens swaps 9 of 12 requested
  residues and changes 18 of 225 observations. The RMSE of the first 50 ns at 298 K changes
  from 1.853 Hz to 0.673 Hz.
- A separate implementation reconstructs all 10,125,000 reanalyzed coupling values to within
  2.72e-6 Hz (tolerance 5e-5 Hz), verifies the handedness in every frame, and reconstructs
  the split R-hat and the effective sample size.
- In a second run, execution of both analysis providers was disabled, and the restored report
  is byte-identical to the original report.
- In all 9 segments, some saved frames have a distance between the protein and a periodic
  image that is below the 0.9 nm cutoff: 945 of 45,000 frames, with a minimum of 0.563 nm.
  Brute-force distances computed by separate code agree on 27 predeclared frames and on the
  9 frames with the minimum distance of each segment.
- The reanalysis file fixes these sampling criteria: a split R-hat of at most 1.05 and a
  combined effective sample size of at least 200, and 214 of 225 observations must pass
  both. At 50 ns, 186 of 225 observations (298 K) and 184 of 225 (310 K) pass. At 100 ns,
  198 of 225 (298 K, 3 replicas) pass.

The original coupling arrays are unchanged. The segments cover 100 ns at 298 K and 50 ns at
310 K and fail both sampling criteria. Because the periodic-image distances are below the
cutoff, the results include contacts between periodic images of the protein. The 2 nm
preparations below have their own verification files.

## CUDA on the Spark host

The Spark host is a Linux ARM64 machine with an NVIDIA GB10 GPU and driver 580.159.03.

The [minimal protein runtime file](verification.gb3-pilot-20261003.json) records a new
environment of 47 OpenMM and PDBFixer packages and 3 replicas of 1 ns of GB3 on the GB10. Its
check of the distance to periodic images, made at 2 snapshots, passed. A later
[run of 3 replicas of 50 ns](verification.gb3-50ns-padding1-failure-20261003.json) with
1 nm padding contained saved frames in which the distance between the protein and a periodic
image is below the 0.9 nm cutoff: 231, 392 and 555 of 5,000 frames in the 3 replicas, with a
minimum of 0.317 nm. The runs had passed the thermal readiness checks, and the stored bonds
were intact. The distance between periodic images requires a separate check.

The [2 nm pilot](verification.gb3-padding2-pilot-20261003.json) uses the same environment
and 2 nm padding, which is the default `padding_nm` of `structure.prepare_protein_system`.
All 3 replicas of 0.4 ns pass thermal entry and all 120 saved-frame separation checks, with
a minimum separation of 2.042 nm against a cutoff of 0.9 nm. The charge and ion counts, the
CPU and CUDA energies and forces, and the restoration of a native checkpoint also pass.

The [2 nm benchmark](verification.gb3-padding2-50ns-20261004.json) records 3 replicas of
50 ns and fixes its own sampling criteria, which differ from those of the reanalysis above.
All 15,000 saved frames pass the image-contact check at 0.9 nm; the minimum separation is
1.308 nm. Thermal entry, finite energies and the production clocks and counts pass.
Production runs at 494.6 to 499.5 ns/day on the GB10. The comparison of 225 couplings has an
RMSE of 0.624 Hz and an MAE of 0.445 Hz. 172 of 225 observations meet both sampling criteria
(split R-hat at most 1.05, combined effective sample size at least 400); 53 do not. The
numerical and geometry checks pass, and the ensemble fails the sampling criterion because
225 of 225 observations must pass. The results apply to GB3 with 2 nm padding.

The CUDA runs in this section were made with earlier versions of the capabilities (source
commit `3c9d55b` for the 2 nm benchmark). With the present capabilities, the Docker image ran
the GB3 ensemble example on the GPU (`platform: CUDA`, mixed precision), and 165 tests passed
(2026-10-23; see the [Docker notes](../docker/README.md)).

The lock `locks/linux-aarch64.cuda13.explicit.txt` reproduces the recorded conda
environment:

```bash
mamba create --prefix /path/to/openmm \
  --file environments/openmm/locks/linux-aarch64.cuda13.explicit.txt
```

Alternatively, install the CUDA platform with pip into a new Mamba Python 3.11 prefix, and
use the Python of that prefix:

```bash
python -m pip install -r environments/openmm/requirements.cuda13.txt
python -m pip check
python -m openmm.testInstallation
export ALAMBIQ_ENV_OPENMM=/absolute/path/to/the/prefix/bin/python
```

`requirements.cuda13.txt` installs OpenMM 8.6.0 with the CUDA platform (`openmm[cuda13]`),
PDBFixer 1.12.0 and the NVIDIA CUDA 13.0.2 components `cuda-toolkit[crt,cudart,cufft,cupti,nvcc,nvrtc,nvjitlink,nvvm]`.
Alambiq is installed in the core environment, which needs the `dossier` and `trajectory`
extras for structure parsing and trajectory inspection.

On the Spark host, the unconstrained `openmm[cuda13]` installation selected CUDA
13.3 components, which failed to load generated PTX on this driver (observed during setup,
outside the verification files). The Docker image, which installs the pinned CUDA
13.0.2 components from `requirements.cuda13.txt`, passed 165 tests on this driver.

`python -m openmm.testInstallation` exits with status 0 when CUDA fails and other platforms
pass (observed during setup, outside the verification files); read the result for each
platform. Alambiq uses the platform that is named in the request. See
[OpenMM installation guidance](https://docs.openmm.org/latest/userguide/application/01_getting_started.html)
and
[NVIDIA's PTX compatibility limitation](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

To run a campaign on CUDA, copy its YAML and set both `structure.prepare_protein_system` and
`simulation.run_molecular_dynamics` to:

```yaml
platform: CUDA
platform_properties: {Precision: mixed}
```

Stochastic trajectories differ between
platforms, and the CUDA results are compared with the experiment separately from the CPU
results.

The [staged-readiness and matched-throughput file](verification.readiness-and-throughput.json)
records new systems of Trp-cage and of an enzyme, throughput on 8 CPU threads on Apple
Silicon and on the Spark host, throughput on the GB10 with CUDA mixed precision, agreement of
CPU and CUDA forces at fixed states, and the GB3 example at 2 temperatures. See
[structure and dynamics](../../docs/methods/structure-and-dynamics.md) for the preparation
defaults and the scientific scope.
