# Third-party notices

The wheel carries Alambiq's own code under Apache-2.0, one JavaScript bundle and the
records of the models and databases it can use. Model weights, databases and scientific
datasets are provisioned outside the wheel, under the terms in each record.

## Offline structure viewer

`alambiq/report/assets/3Dmol-min.js` is 3Dmol.js 2.5.5, University of Pittsburgh and
contributors, BSD-3-Clause; the upstream bundle incorporates MIT-licensed components. The
bundle's own banner, `3Dmol-min.js.LICENSE.txt` and `LICENSE` in the same directory carry
the notices and must travel with it. `README.md` there gives the source package,
upstream commit and digests. Reports embed the bundle only when they show a structure.

## Pinned model and database records

Each `*.json` file beside this one names the files of a resource, with sizes and
digests, and states its license. A record describes a resource; the resource files are provisioned separately.

- `esmc-300m.json`, `esm3-small.json`: Biohub ESM C 300M and ESM3-small, MIT with
  third-party notices. The model cards refer to Biohub's acceptable-use policy, which
  restricts use beyond the copyright grant and applies independently of the MIT terms.
- `thermompnn.json`: Kuhlman-Lab ThermoMPNN, MIT with third-party notices for the code it
  includes (ProteinMPNN, graph protein-design components, LightAttention); the individual
  notices and upstream URLs are in the record.
- `boltz2-2.2.1.json`: Boltz 2.2.1 code and weights, MIT.
- `p2rank-2.5.1.json`: P2Rank 2.5.1, MIT.
- `pfam-38.2-profiles.json`: Pfam 38.2 profiles, CC0 1.0.
- `uniref50-2026_02.json`: UniRef50 release 2026_02, UniProt Consortium, CC BY 4.0.
  Redistribution requires attribution.

## Examples

The example data are in the source checkout and the source distribution. Their sources, terms and transformations are listed in `docs/data-sources.md` and
in each example's README.
