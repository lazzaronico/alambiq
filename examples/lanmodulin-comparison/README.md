# Lanmodulin comparison

**Question.** Which proteins of a 12-protein panel are similar to Mex-LanM, the lanmodulin
of *Methylorubrum extorquens* AM1? How do the annotations and the metal-bound structures of
Mex-LanM and Hans-LanM (the lanmodulin of *Hansschlegelia quercus*) compare? Which genes lie
within 10 kb of the Mex-LanM gene?

The panel holds the 10 proteins of the Mex-LanM gene cluster (including Mex-LanM itself),
Hans-LanM and human calmodulin-1, an EF-hand protein that binds calcium. The campaign has
5 steps:

1. `mex-lanm` (`protein.inspect`) maps the UniProt record C5B164 and the neodymium-bound
   structure 8FNS onto the precursor sequence of Mex-LanM (UniProt C5B164, protein ACS39628.1
   of the chromosome record CP001510.1).
2. `hans-lanm` (`protein.inspect`) does the same for Hans-LanM (UniProt A0A4V2JDD3) with the
   lanthanum-bound structure 8DQ2.
3. `comparison` (`protein.search`) aligns Mex-LanM with MMseqs2 against `panel.fasta` at
   sensitivity 7.5. It keeps hits with an e-value of at most 0.001, an identity of at least
   0.2 and a query coverage of at least 0.6.
4. `alignment` (`sequence.inspect_msa`) reads `comparison.a3m`, the MMseqs2 alignment of
   Mex-LanM and Hans-LanM. It reports conservation per column in the numbering of the
   Mex-LanM precursor. `full_sequences` gives the whole precursor of each row, so that a
   column also maps to a position in the Hans-LanM precursor.
5. `cluster` (`genome.inspect_context`) places each protein of `lanm-pair.fasta` in
   `context.gff3` and lists the features within 10 kb on each side. The sequence of a protein
   in the file must equal the sequence of the CDS of the same identifier in
   `context-proteins.fasta`.

```bash
export ALAMBIQ_ENV_MMSEQS=/path/to/mmseqs2/bin/python   # see docs/install.md
alambiq run examples/lanmodulin-comparison/campaign.yaml --report report.html
```

**Needs.** The MMseqs2 environment (`ALAMBIQ_ENV_MMSEQS`, see
[installation](../../docs/install.md)) and the `dossier` extra (structure parsing).

**What to look at.**

- *Annotations and structures.* The UniProt record of Mex-LanM has 42 features, all mapped
  onto the query. 20 of them are binding sites for Nd(3+), 5 in each of the 4 EF-hand
  domains (residues 35–46, 59–70, 84–95 and 108–119). The record of Hans-LanM has 3
  features (signal peptide, chain and one EF-hand domain, residues 94–129) and no binding
  site. The record names the protein "EF-hand domain-containing protein". Structure 8FNS
  contributes 1 chain (precursor residues 29–133, 105 observed residues) and 4 neodymium
  atoms. Structure 8DQ2 contributes 4 chains (precursor residues 24–133 each, 440 observed
  residues in total) and 12 lanthanum atoms. All 5 chains map exactly. The structures differ
  in metal, construct and crystallization conditions.
- *Search.* Of the 12 panel proteins, 2 pass the thresholds: Mex-LanM itself (ACS39628.1,
  100 % identity, e-value 2.08e-81) and Hans-LanM (36.0 % identity over 100 aligned
  residues, 73 bits, e-value 3.64e-19, query coverage 0.752). The other 9 cluster
  proteins and calmodulin-1 are not returned. The query is a member of the panel, so it
  returns itself.
- *Alignment.* The alignment has 2 rows and 133 columns. Hans-LanM aligns to the query
  columns 33–132, and the first 32 columns and the last column are gaps for it. 36 columns
  are fully conserved, the same count as the identical residues of the search hit; the mean
  gap fraction is 0.124. With 2 rows a column has a consensus fraction of 0.5 or 1.
- *Gene cluster.* The Mex-LanM gene (locus `MexAM1_META1p1786`, protein ACS39628.1) lies
  on CP001510.1 at 1,858,038–1,858,439, on the plus strand. The window of 10 kb on each side
  is 1,848,038–1,868,439 and is not clipped, within the circular chromosome of 5,511,322 bp.
  The window holds 10 CDS features (`MexAM1_META1p1778` to `MexAM1_META1p1787`), all on the
  plus strand, and the step writes their 10 proteins to a FASTA file. Hans-LanM is reported
  as unmapped, because `context.gff3` holds the cluster of *M. extorquens* AM1 only. The
  record lists the product labels and qualifiers of each feature, including the label
  "conserved hypothetical protein; putative exported protein" for ACS39628.1.

**Limits.** The search covers the 12 supplied proteins. An e-value depends on the size of the
panel and compares only between searches of the same panel. Identity is measured over the
aligned region. The thresholds select sequence similarity and carry no information about
metal selectivity. Binding-site features are the annotation of the UniProt record. The
structures hold one metal each (neodymium in 8FNS, lanthanum in 8DQ2), so a difference
between the two structures combines a different protein, metal, construct and crystal
condition. The conservation in `comparison.a3m` applies to the 2 aligned proteins. The
annotation in `context.gff3` covers the 10 CDS features of the cluster, so the window lists
those features only, and the unmapped state of Hans-LanM reflects the supplied annotation.
Adjacency and strand are positional information.

**Data.**

- `query.fasta` is the precursor sequence of the protein ACS39628.1 of the chromosome record
  CP001510.1 (INSDC, no use restriction), which equals the sequence of the UniProt record
  C5B164. `hans.fasta` is the sequence of A0A4V2JDD3.
- `C5B164.json` and `A0A4V2JDD3.json` are UniProtKB records retrieved on 2026-09-07 (UniProt
  Consortium, CC BY 4.0), with SHA-256 digests
  `0a2c4a9c87d5bddc752dc5cbdf49df81f5ddf1997aac354a59e29fbdbc0f325f` and
  `6008f981e27028c3c224bc19a5b3f43aaa05a43c8d648cccd8440877008f36b9`.
- `8FNS.cif` and `8DQ2.cif` are the wwPDB coordinate files of the entries
  [8FNS](https://www.rcsb.org/structure/8FNS) and [8DQ2](https://www.rcsb.org/structure/8DQ2)
  (CC0), unmodified, with SHA-256 digests
  `bb593dc407de9f1d00138ea19c417a503b88bae3a2926967d53595da312b2d9f` and
  `d9de93986fbb902ff60c0162dd2baa175c53bcb349b0bca91aed0472596297e7`. Both entries are cited
  with doi:10.1038/s41586-023-05945-5.
- `context.gff3` and `context-proteins.fasta` hold the 10 CDS features
  `MexAM1_META1p1778` to `MexAM1_META1p1787` of the ENA record
  [CP001510.1](https://www.ebi.ac.uk/ena/browser/api/embl/CP001510.1) (INSDC, no use
  restriction), with the absolute chromosome coordinates, all qualifiers and the deposited
  translations. The chromosome DNA is omitted. The source file had 14,872,457 bytes and the
  SHA-256 digest `97fd22fd492668f77bc2b6ff7f395cf6fcc675d8ef8a2881dcfd66c22b11013c`; the
  deposited translations were checked against the genomic DNA with translation table 11.
- `panel.fasta` holds the 10 cluster proteins of `context-proteins.fasta`, Hans-LanM
  (A0A4V2JDD3) and human calmodulin-1 (P0DP23, UniProt, CC BY 4.0). `lanm-pair.fasta`
  holds the two lanmodulins.
- `comparison.a3m` is the MMseqs2 18-8cc5c alignment (`search` with `-s 7.5 -e 0.001
  -c 0.6 --cov-mode 0 -a`, then `result2msa --msa-format-mode 5`) of `query.fasta` against
  `panel.fasta`, with the row of the query against itself removed because an alignment holds
  each identifier once.

The UniProt record of Mex-LanM cites its characterization in doi:10.1021/jacs.8b09842, and the
study of the lanmodulin dimer and of Hans-LanM is doi:10.1038/s41586-023-05945-5. Sources and
terms of the data are in [the data-source index](../../docs/data-sources.md).
