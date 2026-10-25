# Examples

Each example directory holds a `campaign.yaml`, the input files that the campaign reads and
a README that states the question, the requirements, the outputs to inspect and the limits.
`tests/core/test_examples.py` runs the `campaign.yaml` of each example and checks values of
its results; `tests/core/test_connected_investigation.py` does the same for
`connected-investigation`. The campaigns `complex.yaml` (in `pockets-and-complex`) and
`embed.yaml` (in `representative-panel`) are not run by a test. Examples that need an
engine environment are skipped when the environment is not configured.

```bash
alambiq run examples/kinetics-from-measurements/campaign.yaml --report report.html
```

A dossier is the result of `protein.inspect`: the sequence of one protein (the query) with
the features of a UniProt record and the chains and residues of a structure, in the
numbering of the query. The `dossier` extra installs the structure file reader (Gemmi) that
`protein.inspect` needs. A round is one batch of variants that is made and measured
together.

Protein, variant and structure examples:

- [protein-first-look](protein-first-look/README.md): the `protein.inspect` dossier of one
  protein.
- [relatives-and-context](relatives-and-context/README.md): similar proteins, conservation
  and the genomic context of a gene. Needs MMseqs2.
- [lanmodulin-comparison](lanmodulin-comparison/README.md): the annotations and
  metal-bound structures of two lanmodulins, their similarity to a panel of 12 proteins and
  the genes around Mex-LanM. Needs MMseqs2.
- [representative-panel](representative-panel/README.md): 3 representative proteins of a
  6-protein panel, chosen from language-model embeddings.
- [pockets-and-complex](pockets-and-complex/README.md): predicted ligand-binding pockets of
  hemoglobin alpha (needs P2Rank) and a second campaign that predicts a protein-ligand
  complex (needs Boltz-2).
- [variant-to-construct](variant-to-construct/README.md): an assembled DNA construct design
  for one chosen variant.
- [ensemble-vs-nmr](ensemble-vs-nmr/README.md): a simulated ensemble compared with measured
  couplings. Needs OpenMM.

Measurement and operation examples:

- [kinetics-from-measurements](kinetics-from-measurements/README.md): the Michaelis constant
  (Km) and the maximum rate (Vmax) from initial rates.
- [measured-variant-rounds](measured-variant-rounds/README.md): the selection of variants to
  measure in the next round, from measured variants.
- [operation-from-progress-curves](operation-from-progress-curves/README.md): fitting
  progress curves of a reaction and simulating the fitted operation.
- [cofactor-regeneration](cofactor-regeneration/README.md): the effect of cofactor
  regeneration on the amount of product.

Reaction examples:

- [transformation-and-routes](transformation-and-routes/README.md): the thermodynamic
  feasibility and the synthetic routes of a reaction.
- [transformation-to-enzymes](transformation-to-enzymes/README.md): candidate enzyme
  sequences ranked and characterized for a reaction. Needs MMseqs2.

Example with generated inputs:

- [connected-investigation](connected-investigation/README.md): a campaign of 10 steps from
  a protein to the comparison of a simulated operation with a generated progress curve.

Application fields and their examples:

| Application field | Examples |
| --- | --- |
| Metal recovery | [lanmodulin-comparison](lanmodulin-comparison/README.md) |
| Renewable chemistry | [cofactor-regeneration](cofactor-regeneration/README.md), [transformation-and-routes](transformation-and-routes/README.md) |
| Environment | [transformation-to-enzymes](transformation-to-enzymes/README.md) |

Sources and terms of the data are in [the data-source index](../docs/data-sources.md).
