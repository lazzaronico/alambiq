# Relatives and context

**Question.** Which proteins of a 6-protein panel pass the similarity thresholds for
hemoglobin alpha, which alignment columns are conserved among the query and 3 relatives, and
which proteins are encoded within 5 kb of each of 2 genes in a genome?

The campaign has 3 steps with separate inputs:

1. `protein.search` aligns hemoglobin alpha with MMseqs2 against the panel
   `representation-panel.fasta`, which contains 6 human proteins, and filters hits by
   e-value, identity and coverage.
2. `sequence.inspect_msa` reads the stored A3M alignment `globin-search.a3m` and reports
   conservation and entropy per column. The alignment is an input file and does not depend
   on step 1. It contains 4 sequences: the query P69905 and P68871, P69891 and P02144; its
   headers give the hit statistics of the 3 other sequences (P02144 has identity 0.263,
   below the threshold of step 1).
3. `genome.inspect_context` extracts the annotated genes within 5 kb on each side of each of
   two anchor genes of a gut metagenome-assembled genome. The request names the anchors.

```bash
export ALAMBIQ_ENV_MMSEQS=/path/to/mmseqs2/bin/python   # see docs/install.md
alambiq run examples/relatives-and-context/campaign.yaml --report report.html
```

**Needs.** The MMseqs2 environment (`ALAMBIQ_ENV_MMSEQS`, see [installation](../../docs/install.md)).

**What to look at.** Of the 6 panel proteins, 3 pass the thresholds (e-value 0.001, identity
0.3, coverage 0.8): the query itself (100 % identity), the beta chain at 43 % identity and
the gamma-1 chain at 42 %. The other 3 panel proteins (myoglobin, cytochrome c and actin) do
not pass the thresholds. The alignment has 4 sequences, 142 columns and 23 fully conserved
positions. Each of the 2 anchors matches exactly one locus. The step writes the 25 proteins
of the 2 windows (14 and 11, anchors included) to a FASTA file. The window of
`MGYG000000699_00080` is clipped at position 1 and the window of `MGYG000000699_00104` at
position 16709, the contig bounds declared in `context.gff3`; the report states this.

**Limits.** The search covers the 6 panel proteins. Conservation statistics apply to the 4 aligned
sequences. The genome subset is a single contig of 29 features around two ribosomal protein
genes. The rank of a hit reflects sequence similarity.

**Data.** Sequences are UniProtKB (CC BY 4.0); see `representation-panel.sources.json`. The
genome subset is from EMBL-EBI MGnify genome MGYG000000699 (catalog human-gut-v2-0-2),
under the EMBL-EBI terms of use; `subset.json` holds the digests and the transformation
(the contig rows are retained and the embedded DNA sequence is omitted).
