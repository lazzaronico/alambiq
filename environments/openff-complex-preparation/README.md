# OpenFF protein-ligand preparation

The three capabilities of this environment use OpenFF Sage 2.3.0 with AshGC charges from
`openff-gnn-am1bcc-1.0.0.pt`. OpenMM supplies the ff14SB protein parameters and the
TIP3P water parameters. The ligand parameters differ from those of the earlier
GAFF2/AM1-BCC preparations; results of those preparations apply to their own parameter set.

From the repository root, install as follows:

```sh
mamba env create --prefix /path/to/openff-env \
  --file environments/openff-complex-preparation/environment.yaml
/path/to/openff-env/bin/python environments/install_openff_sources.py
export ALAMBIQ_ENV_OPENFF=/path/to/openff-env/bin/python
alambiq env check openff
```

`ALAMBIQ_ENV_OPENFF` is the engine environment of
`structure.prepare_protein_ligand_system`,
`structure.prepare_predicted_complex_for_dynamics` and
`structure.prepare_finite_environment` (`alambiq capability show` names the environment of
each capability). Dynamics on the prepared system is run by
`simulation.run_molecular_dynamics` in the `openmm` engine environment
(`ALAMBIQ_ENV_OPENMM`). `ALAMBIQ_ENV_OPENMM` can name the Python of the
[OpenMM recipe](../openmm/README.md) or of this environment, which contains OpenMM.

On Linux with a compatible NVIDIA driver, add `cuda-version=13.0` to the dependencies in a
copy of the recipe before creating the environment. The recipe installs CPU-only PyTorch,
which infers the charges; GPU dynamics use OpenMM's CUDA platform. Set the OpenMM platform in
the request.

The source installer verifies the hash of each upstream archive against the value in the
[source manifest](../openff-sources.json) and installs OpenMMForceFields 0.15.1 without the
AmberTools dependency of its Conda recipe. The installer installs OpenFE 1.12.0 from the
same manifest when it runs with `--binding`. The [binding environment](../binding-free-energy/README.md)
installs OpenMM 8.4 with OpenFE 1.12.0. A run reads the installed packages and resources and uses no network.

`alambiq env check openff` reports whether the declared packages of the environment import.
The key of a step includes the Python version and the versions of the declared packages of
its environment. The packaged Sage file is the unconstrained variant, which contains the bonded terms of
constrained bonds. OpenMM applies the requested production constraints when it creates the
System. ParmEd writes Amber-format topology and restart files without an AmberTools
executable.

The predicted-complex preparation uses 20 Å padding and 0.15 M added NaCl, with a
5,000-iteration cap on minimization. A verification file is a JSON file that records one
run or one test session: the operating system and hardware, the package versions and the
outcome of each check. Two verification files record this preparation:

- The [parameter regression file](verification.predicted-settings-20261008.json) records
  7 test cases. They check that the configured padding, salt concentration and minimization
  cap are passed to the solvation step and written to the preparation manifest, with salt
  concentrations of 0, 0.15 and 0.3 M and with the chemistry calls replaced by stubs.
- The [Apple Silicon file for the predicted complex](verification.predicted-native-osx-arm64-20261008.json)
  records the preparation of a saved lysozyme-benzene prediction (162 protein residues, a
  ligand with 6 heavy atoms), 2 ps of CPU dynamics, a check of the distance between the
  protein and its periodic images in all 10 saved frames, and a second run that reused all
  stored step results. The system has 34,706 atoms, including 29 added NaCl pairs and
  8 neutralizing chloride ions. The potential energy decreases from 5.81 × 10^13 kJ/mol to
  -5.92 × 10^5 kJ/mol during minimization. The regenerated report is byte-identical to the
  original report.

These runs check the integration of the steps.

## Verification files

These verification files cover the other preparations:

- The [Apple Silicon file](verification.osx-arm64.json) records the preparation of a
  protein with a positioned ligand, followed by CPU dynamics and inspection; the assembly of
  a trimer, its solvation and finite projection; and binding-input preparation.
- The [Linux file](verification.linux-aarch64.json) records 4 ligand chemistry controls, the
  export of portable parameters, and 100-step dynamics on the Reference, CPU and CUDA
  platforms.
- The [Linux file for the trimer campaign (outside Docker)](verification.native-joins-linux-aarch64-20261008.json) records
  a trimer assembly, orthorhombic solvation that preserves the parameters, a finite projection
  that keeps all 6,249 solute atoms, and binding-input preparation. A separate implementation
  that reads the stored results regenerated the reports of the assembly and of the
  binding-input preparation. The SHA-256 of each regenerated report equals that of the
  original report. An earlier attempt with an octahedral cell completed assembly and
  solvation, and the finite projection rejected the cell. Finite projection requires an
  orthorhombic cell. The orthorhombic campaign reused the stored coordinates and assembly
  results of that attempt.
- The [electronic-state evaluation](../qmmm/verification.openff-preparation-linux-aarch64-20261008.json)
  computes a single point on the finite preparation of 8,677 atoms with the 24 atoms of
  chorismate in the quantum region. A separate implementation reproduces the forces, the
  energy components and the electronic population, and regenerates the report.
- The [Docker file of 2026-10-09](verification.docker-linux-aarch64-20261009.json) records
  Linux ARM64 CPU runs in the Docker image. All 7 preparation, dynamics and inspection steps
  completed with the network disabled and a read-only root file system. The finite projection
  of the trimer contains 8,655 atoms in this run and 8,677 atoms in the run outside Docker
  above (the solvated systems contain 110,927 and 110,570 atoms); the file records no cause
  of the difference. The predicted-complex campaign has
  34,706 atoms, 29 added salt pairs and 8 neutralizing chloride ions, and its 2 ps trajectory
  passes the periodic-image check in all 10 saved frames. In a second run with engine
  execution disabled, every step reused its stored result, and the regenerated reports equal
  the original reports byte for byte. The image contains 231 package files. 230 of them are
  identical to the source tree at the time of the check, including all scientific and report
  modules. One file, `plugins/runtime_locations.py` of the earlier layout, differs. A program
  that reads the stored results, installed from a wheel that was built from source commit
  `244bbaf`, restores both original reports byte for byte.
- The [regression file](verification.migration.json) records, for the earlier repository
  layout, a test suite of 1,370 tests (78 skipped) run on the source tree and on the
  installed wheel (in each case all selected tests passed), 11 command-line checks, 9
  chemistry-contract tests and 12 numerical tests of the installed wheel.
- The [binding file](../binding-free-energy/verification.openff-20261003.json) records the
  binding calculation that uses these parameters.

These files record integration checks and the preservation of earlier results.

## Sources and licenses

- [OpenFF Toolkit](https://github.com/openforcefield/openff-toolkit) and
  [NAGL](https://github.com/openforcefield/openff-nagl): MIT.
- [Sage force-field data](https://github.com/openforcefield/openff-forcefields)
  and [NAGL model assets](https://github.com/openforcefield/openff-nagl-models):
  CC BY 4.0. The attribution is separate from the toolkit license.
- [OpenMMForceFields](https://github.com/openmm/openmmforcefields/tree/0.15.1): MIT.
- RDKit: BSD-3-Clause.
- PDBFixer: MIT.
- Gemmi: MPL-2.0.
- [ParmEd](https://github.com/ParmEd/ParmEd): LGPL-2.1-or-later.
- OpenMM: MIT for the core, the API and the CPU platform; LGPL for the GPU platforms; NVIDIA
  terms for the CUDA components.

The Python runtime from Conda includes GPL readline, GCC runtime libraries carry the GCC
runtime exception, and the package inventories in `locks/` list the license of each
package. AmberTools and FFTW are not installed in the recorded preparation environment. The
verification files of installations that included AmberTools describe the earlier
preparations.
