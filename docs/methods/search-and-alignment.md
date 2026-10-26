# Search and alignment

A homolog search finds proteins related to a protein of interest. `protein.search` finds
homologs in a panel, `sequence.generate_msa` builds an alignment, and
`sequence.inspect_msa` reads one. `enzyme.retrieve_candidates` and
`enzyme.characterize_candidates` rank candidates for a function against references.

## Searching a panel

`protein.search` aligns one protein against a panel supplied in the request and returns
the hits that meet the thresholds of the request. A hit is evidence of homology. Only
homologs in the panel can be found.

An expectation value applies to a panel of that size, so the same alignment has a better
expectation value against a small panel. Identity is read together with coverage.
Identity is measured over the aligned region, which can cover a fraction of either
protein, so a high identity over 20 residues is a different finding from the same
identity over the whole protein.

Set the thresholds before inspecting the hits. A higher `sensitivity` returns more distant
homologs, each with weaker evidence of homology. The MMseqs2 `sensitivity` of
`protein.search`, `enzyme.retrieve_candidates`, `enzyme.characterize_candidates`,
`sequence.generate_msa` and `protein.match_profiles` is from 1 to 7.5 inclusive. A value
outside this interval is refused. The default of `protein.search` is 5.7, the default of
MMseqs2.

## Building an alignment

`sequence.generate_msa` searches the pinned UniRef50 release with `iterations` search
iterations (1 to 5; default 3) and returns one A3M file per protein. The rows after the
query are sorted by identifier and then by sequence, so that the same search gives the
same file. The independence of the sorted rows from the thread count was tested with 1
and 4 threads on a toy database.

Each alignment is assessed for depth before use by three quantities:

- the number of distinct rows;
- the number of diverse rows: rows that cover at least `minimum_query_coverage` of the
  protein (default 0.5) and are at most 90 % identical to it;
- the median fraction of the protein covered by a row.

An alignment is marked `withheld` and stored for inspection when it has fewer distinct
rows than `minimum_alignment_sequences` (default 32), fewer diverse rows than
`minimum_diverse_sequences` (default 16), or a median row coverage below
`minimum_query_coverage` (default 0.5).

The depth quantities measure the amount of aligned sequence. UniRef50 stores one
representative per 50 % cluster, so close relatives are collapsed. A protein with few
relatives in UniRef50 has a shallow alignment at every setting of the capability. A
protein with a domain found in many unrelated proteins can have a deep alignment whose
rows cover only that domain. Each chain is aligned separately, so the alignment of a
complex has signal within each subunit only.

The capability offers no identity-ceiling setting, because the identity ceiling of
MMseqs2 leaves near-identical hits in the alignment. The representative of the query
itself can appear in its alignment.

A search loads the index files of the database (about 118 GB) before searching. The
timeout is 4 hours for each protein, and a step searches at most 8 proteins.

## Reading an alignment

`sequence.inspect_msa` reads a FASTA alignment or a query-centered A3M and reports
per-column entropy, gap fraction and consensus. The first row defines the query numbering.
In both formats it must be ungapped and in capitals, and the capability refuses the
alignment otherwise.

Counts are unweighted, so duplicate rows count each time. A column occupied by one row,
with gaps in all other rows, has a consensus equal to the residue in that row and a high
gap fraction. Read the entropy together with the gap fraction.

In A3M, lowercase residues are insertions relative to the query and add no columns. Row
positions count residues, including insertions. Match columns and query positions are
two further numberings. An exported match-column alignment omits the insertions, so
retain the original alignment when insertions matter.

`sequence.inspect_msa` reads at most 200 rows, 5,000 columns and 250,000 aligned cells
(rows times columns). Select a subset of rows before submitting a larger alignment.

## Ranking candidates against references

`enzyme.retrieve_candidates` scores each protein by its best alignment to a reference
other than the protein itself. It then assesses the ranking against named controls and
refuses to run without them.

- **Positive controls** are references that the request declares to have the function.
- **Negative controls** (named hard negatives in the result) are candidates that the
  request declares not to have the function and that are similar enough to the positives
  to test the ranking, for example a neighboring superfamily function, a related
  substrate or a misleading annotation.

Both declarations are taken as supplied. The result gives:

- the recall of the positives: the fraction that score at least `positive_min_bits`
  (default 40.0 bits);
- whether every negative control scores below every positive;
- the AUROC (area under the receiver operating characteristic curve);
- the AUROC of a score from protein length alone.

Similarity separates these controls better than protein length when the control AUROC
exceeds the length-only AUROC. With 3 positive and 3 negative controls and no tied
scores, the AUROC changes in steps of 1/9, so a perfect AUROC is weak evidence.

A good separation shows that similarity contains information for these controls. The
function of a high-ranking unknown is a hypothesis to test, because homologs can share
folds without sharing reactions. A missing hit is the absence of retrieval evidence.

`enzyme.characterize_candidates` checks whether the candidates retain the residues that
the literature associates with function. The request supplies the reference, its
residues, their roles and the citations. Each candidate is aligned to the reference and
the residues are read from the alignment. The decision for a candidate is one of three:

- **Rejected**: the candidate has no alignment to the reference, its retrieval score is
  below `minimum_retrieval_bits` (default 100.0 bits) or, among the candidates not
  withheld, the fraction of the named residues that are the expected amino acid is below
  `minimum_conserved_feature_fraction` (default 1.0). A candidate with no alignment to the
  reference has the single reason code `no-reference-alignment`, and its retrieval score
  is not tested.
- **Withheld**: the alignment covers less than `minimum_query_coverage` of the reference
  (default 0.8) or the fraction of the named residues inside the alignment, not on a gap,
  is below `minimum_mapped_feature_fraction` (default 1.0). Withholding is the decision for
  a candidate whose alignment lacks evidence, such as a partial alignment or one that
  places a catalytic residue in a deletion, and the decision records this.
- **Advances**: every other candidate.

Set the thresholds before reading the candidates and evaluate the controls first. The
thresholds separate the controls for a family when every positive control advances and
every negative control is rejected. The advancing candidates of a family are reported
with this result.

A candidate that advances retains at least the fraction
`minimum_conserved_feature_fraction` of the named residues at aligned positions. Folding,
expression and catalysis of an advancing candidate are determined by experiment. The
citations are supplied in the request and stored with the result as given.

Alignment-based transfer applies to candidates that align to the reference without
insertions, fusions or rearranged domains. At low identity or with such differences,
profile or structure comparison (`protein.match_profiles`, `structure.compare`) is
needed.
