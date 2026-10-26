# Example data sources and attribution

This page lists the data used by each example in `examples/`. The licenses of the source
data, of the tables derived from them, of the model resources and of the Alambiq code are
independent of each other. Redistribution of the source material, or of an example built
from it, is subject to the license and attribution terms stated below.

- **protein-first-look, pockets-and-complex**: UniProtKB P69905 (record and sequence;
  UniProt Consortium, [CC BY 4.0](https://www.uniprot.org/help/license)) and wwPDB
  entry 1A3N (CC0).
- **relatives-and-context**: UniProtKB sequences of a 6-protein panel (CC BY 4.0;
  `representation-panel.sources.json` records each); an MMseqs2 alignment derived from
  them; and a contig of EMBL-EBI MGnify genome MGYG000000699, catalog
  human-gut-v2-0-2, under the [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/)
  (`subset.json` records the digests and the transformation).
- **lanmodulin-comparison**: UniProtKB records C5B164 and A0A4V2JDD3 and the sequence of
  P0DP23 (UniProt Consortium, [CC BY 4.0](https://www.uniprot.org/help/license)); wwPDB
  entries 8FNS and 8DQ2 (CC0); the 10 CDS features and proteins of ENA record CP001510.1
  (INSDC sequence data, no use restriction under the INSDC data-use policy). The README of
  the example records the digests and the derivation of the alignment.
- **representative-panel**: the 6-protein panel of relatives-and-context; the embeddings are
  ESM C 300M output (provider Biohub; license `MIT-with-third-party-notices` plus an
  acceptable-use policy; resource specification `src/alambiq/resources/esmc-300m.json`).
- **kinetics-from-measurements, cofactor-regeneration, transformation-and-routes**: the
  values in the example directory are generated or written for the example, as the README
  of each example states. The thermodynamic data of transformation-and-routes come from the
  eQuilibrator packages named in the report of that example. They are outside the example
  directory, and their terms are stated in
  [LICENSING.md](../LICENSING.md#terms-of-optional-software-models-and-data).
- **variant-to-construct**: pUC19 (INSDC record M77789.2, no use restriction under the
  INSDC data-use policy) and the mature hen egg-white lysozyme sequence (UniProtKB P00698,
  CC BY 4.0) with the S100R substitution.
- **connected-investigation**: the lysozyme sequence as above; the language-model scores
  are ESM C 300M output; the measurements are simulated from the model stated in the README
  of the example, with fixed seeds; the generating values are in `truth.json`.
- **measured-variant-rounds**: ProteinGym 1.3, dataset ESTA_BACSU_Nutschel_2020 (license MIT;
  [Zenodo 15293562](https://zenodo.org/records/15293562)); Nutschel and colleagues,
  doi:10.1021/acs.jcim.9b00954.
- **transformation-to-enzymes**: UniProtKB sequences (CC BY 4.0; accessions in
  `candidates.fasta` and `references.fasta`) in the HAD-S2 dehalogenase panel; the
  active-site residues are cited in doi:10.1074/jbc.271.34.20322.
- **operation-from-progress-curves**: Chen and colleagues, *Nature Communications*
  (2023), doi:10.1038/s41467-023-37921-y, Fig. 3d1 source data, CC BY 4.0;
  `source-measurements.json` records the cells and the original workbook digest.
- **ensemble-vs-nmr**: wwPDB entry 1P7E (CC0); scalar couplings of protein GB3 from
  Vögeli, Olsson, Riek and Güntert (2015), doi:10.1016/j.dib.2015.08.020, CC BY 4.0.

The test fixtures are listed, with their terms, in
[`tests/fixtures/README.md`](../tests/fixtures/README.md).
