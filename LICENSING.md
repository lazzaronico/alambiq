# Licensing

The code, documentation and configuration written for Alambiq are licensed under
[Apache-2.0](LICENSE). The licenses of the third-party data in `examples/` are stated in the
[example data-source index](docs/data-sources.md).

Alambiq selects scientific dependencies and model weights whose licenses permit commercial
use without license fees, individually negotiated agreements, or academic-only or
noncommercial-only restrictions. MIT, BSD and Apache-2.0 are preferred;
the pattern `ALLOWED` in `tools/scan_licenses.py` defines the other accepted
licenses. The code, the model weights and the datasets of a component are checked
separately, each at the pinned version. Components are selected to meet these requirements;
the data terms of the `thermodynamics` extra are stated in the list below.

text-unidecode, which is installed through python-slugify, a dependency of Prefect, is
offered under either the GPL (version 2 or later) or the Artistic License. Alambiq uses
it under the Artistic License, which the Open Source Initiative has approved.
`tools/scan_licenses.py` lists each installed dependency whose license is not classed as
allowed; `--all` lists every dependency.

Ligand preparation uses the OpenFF Sage 2.3.0 force field with AshGC charges. The scientific
software of each engine environment is the software named in its package inventories
(`locks/*.packages.json`).

OpenMM's GPU platforms, MDTraj and ParmEd are licensed under the LGPL; redistributing them
is subject to the terms of the LGPL. The package inventories of the OpenFE, OpenFF,
OpenMM and reaction-free-energy environments (`locks/*.packages.json` in their directories
under `environments/`) list readline 8.3, a runtime dependency of Python, under
GPL-3.0-only (the conda-forge label; upstream is GPL version 3 or later). The inventories
`environments/openmm/locks/osx-arm64.apple-ensemble.packages.json` and
`environments/reaction-free-energy/locks/osx-arm64.packages.json` record no license for any
package. Distributing an image of an environment requires the GPL text and the source of
readline. The OpenFE environment omits the graph renderer and the
hicolor icon theme (GPL); the
[verification receipt](environments/binding-free-energy/verification.runtime-simplification-20261003.json)
lists the packages removed from it.
[OpenMM's licensing documentation](https://docs.openmm.org/latest/userguide/library/01_introduction.html#license)
identifies its Python, API and CPU components as MIT and its GPU platforms as LGPL.
The pinned OpenMM 8.6.0 [CUDA source notice](https://github.com/openmm/openmm/blob/8.6.0/platforms/cuda/src/CudaPlatform.cpp)
and [OpenCL source notice](https://github.com/openmm/openmm/blob/8.6.0/platforms/opencl/src/OpenCLPlatform.cpp)
specify LGPL-3.0-or-later.

Each JSON specification in `src/alambiq/resources/` states the license of one resource. Where an
environment directory under `environments/` has a licensing section or a package
inventory, it states the licenses of the software in that environment.

Third-party material remains under its own license: viewer code bundled in the report, example
sequences, structures and measurements, optional native software, model weights,
databases and force fields. The conditions of attribution, redistribution and use of each
item apply. The wheel includes [third-party notices](src/alambiq/resources/THIRD_PARTY_NOTICES.md)
and the [offline viewer notices](src/alambiq/report/assets/README.md). The notices of an
optional item are in the directory of its environment under `environments/` or among the
files of its resource. The [example data-source index](docs/data-sources.md) lists the
source data of each example and the license of each source.

Optional resources are installed by the user from the upstream source.
Before redistributing one, check its license, its trademark terms and the ownership and
publication permissions that apply to it.

## Terms of optional software, models and data

- The pinned Biohub ESMC-300M and ESM3-small resources are licensed under MIT terms with
  third-party notices; see their [environment recipe](environments/protein-representations/README.md).
  These terms apply to the pinned Biohub resources; older EvolutionaryScale distributions and
  hosted services have their own terms. A different model version needs its own license
  check, because the license of the Python package that loads a model is separate from the
  license of the model. The [ESMC specification](src/alambiq/resources/esmc-300m.json) and
  the [ESM3 specification](src/alambiq/resources/esm3-small.json) pin the model cards, the
  license file and the third-party notice files together with the model weights. The MIT
  license imposes no condition on embeddings or scores computed with the weights. The
  ESMC-300M model card and the Biohub README refer to Biohub's [acceptable-use
  policy](https://biohub.org/acceptable-use-policy/) (version of 2026-05-27), which
  prohibits uses that concern personal data, infringement of intellectual property,
  unreviewed high-risk decisions, biological and chemical weapons, sequences of concern and
  circumvention of safety controls. Commercial use is not among the prohibited uses. The
  [third-party notices](src/alambiq/resources/THIRD_PARTY_NOTICES.md) record the policy for
  both models and state that it applies independently of the MIT terms; the user is
  responsible for compliance.
- The [upstream release documentation](https://github.com/jwohlwend/boltz/blob/v2.2.1/README.md#license)
  of Boltz 2.2.1 declares its code and model weights MIT-licensed; the
  [code notice](https://github.com/jwohlwend/boltz/blob/v2.2.1/LICENSE) names
  Jeremy Wohlwend, Gabriele Corso and Saro Passaro. The
  [resource lock](environments/boltz2/resource-lock.osx-arm64.json) for macOS Apple
  silicon records the SHA-256 digest of each checkpoint file and of the molecular-resource
  tree. `alambiq resource provision` fetches the model from the upstream
  source. The [runtime recipes](environments/boltz2/README.md) define an
  official CPU and CUDA environment and an experimental Apple GPU environment. Keep the
  notices of the data and dependencies when redistributing optional resources.
- MMseqs2 release 18-8cc5c is MIT-licensed, copyright the MMseqs2 Development Team
  ([upstream license](https://github.com/soedinglab/MMseqs2/blob/18-8cc5c/LICENSE.md)).
  The UniProt and Pfam databases have separate terms: UniRef50 and UniProtKB/Swiss-Prot are
  CC BY 4.0, with attribution. The `build` field of the UniRef50, Swiss-Prot and Pfam resource
  specifications states the MMseqs2 release used to build each database.
- Example source material, whose particular sources and transformations each example
  records:
  - UniProt-derived material: CC BY 4.0, with attribution.
  - Pfam seed panels and deposited RCSB coordinates: CC0.
  - The pUC19 sequence (INSDC record M77789.2, submitted by J. Messing): the INSDC data-use
    policy attaches no use restriction or license to sequence records and asks users to
    cite the original submission, Yanisch-Perron, Vieira and Messing, Gene 33, 103-119
    (1985).
- The [SrIRED panel test fixture](tests/fixtures/enzyme-reaction-panel/source-accounting.json)
  contains the subset of the Gantz and colleagues lrDMS v2.0.0 dataset (CC BY 4.0) that is
  selected by sequence. `source-accounting.json` lists the source tables with their SHA-256
  digests and row counts, and the line and reason of each excluded row. The article that
  describes the dataset has its own publication terms.
- The OpenFF Sage 2.3.0 force-field files (openff-forcefields 2026.09.0) and the AshGC
  1.0.0 charge model (openff-nagl-models 2025.9.0, file `openff-gnn-am1bcc-1.0.0.pt`) are
  licensed under CC BY 4.0, copyright the Open Forcefield Group. Redistribution requires
  attribution, a link to the license and a statement of any changes.
- The [OpenFF environment](environments/openff-complex-preparation/README.md) contains
  MIT-licensed code and CC BY 4.0 force-field and model files. Its package inventories list
  the license of each package, including LGPL libraries (ParmEd, LGPL-2.1-or-later) and
  the GCC runtime libraries (GCC runtime exception). Distributing
  the environment requires keeping the attribution of the model and data files. CUDA
  components are licensed under NVIDIA terms.
- The [QM/MM environment](environments/qmmm/README.md) uses PySCF and GPU4PySCF, which are
  licensed under Apache-2.0.
  The CUDA build of GPU4PySCF 1.8.1 compiles NVIDIA CUTLASS v3.4.0 (commit 751eb9a), which is
  licensed under BSD-3-Clause, copyright 2017-2024 NVIDIA CORPORATION & AFFILIATES; a
  distributed wheel or image reproduces that notice. The same build compiles LibXC (commit
  b225c254 of a fork of LibXC 7.0.0), licensed under MPL-2.0; the CPU environments use
  conda-forge libxc-c 7.1.2. Distributing a built wheel or image requires making the
  source of the LibXC files, including the changes applied during the build, available
  under MPL-2.0.
- The `thermodynamics` extra installs equilibrator-api 0.8.0, equilibrator-cache 0.8.1,
  component-contribution 0.8.0 and equilibrator-cache-data 0.1.0, which are MIT-licensed
  code. The compound database (Zenodo 10.5281/zenodo.4128543), the model parameters, the
  training data and the group definitions are deposited under CC BY 4.0. The compound
  database is built from MetaNetX MNXref 3.1, which states that the licenses of its
  source databases continue to apply: KEGG and BiGG permit use for non-commercial or
  non-profit purposes only, and HMDB requires permission for commercial use and
  redistribution. The acid-base dissociation constants in the cache were
  computed with ChemAxon Marvin under an academic license. Commercial use of the extra,
  and of results that reproduce compound-database fields, follows the terms of these
  sources. The user installs these packages and data.
- Distributing a binary or container image, as opposed to building it locally, requires
  keeping its required notices and meeting the source-distribution obligations that apply
  to that image. This repository distributes container build recipes, not images.

The rights to each biological input, physical material, model and dependency that is used
or redistributed are separate from the Apache-2.0 license, which permits commercial use of
the code, documentation and configuration written for Alambiq.
