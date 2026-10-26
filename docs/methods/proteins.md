# Proteins

Four capabilities cover protein inspection, family membership, predicted tracks and genomic
context: `protein.inspect`, `protein.match_profiles`, `protein.predict_tracks` and
`genome.inspect_context`. `alambiq capability show <name>` prints the request, result and
limits of each.

## Inspection

`protein.inspect` combines a sequence, an optional UniProt reference and an optional
structure into one record. The record states how the three correspond.

A reference feature is transferred to the query only when the whole query is a contiguous
subsequence of the reference that occurs exactly once, and the feature lies wholly inside
it. A structure chain maps only when its complete canonical polymer occurs exactly once in
the query, or the query occurs exactly once in the polymer. Every other case is stored as a
diagnostic marked withheld, with the reason: an internal substitution, an indel, an isoform
difference, a repeated match, an uncertain position or an unobserved residue. The
diagnostic is listed beside the feature it affects.

A feature that did not map is absent from the mapping and can still be present in the
protein; the diagnostics give the reason. The record stores three coordinate systems
separately: query positions, label-chain sequence ids, and author numbering with insertion
codes. Per-residue confidence is read only from a declared pLDDT metric on its 0 to 100
scale.

The optional per-residue annotations are computed from the supplied coordinates. A residue
with no observed coordinates has null values.

Secondary structure is eight-state DSSP: H, G and I are helices, E and B are strands and
bridges, T is a turn, S is a bend and a blank is a loop. It is assigned over runs of
consecutive polymer positions with complete backbones. A break is a distance above 2 Å
between the carbon of one residue and the nitrogen of the next.

Accessible area is computed by Shrake-Rupley with a 1.4 Å probe and 960 points per atom
over the observed heavy atoms, in square ångström. Hydrogens and water are excluded. The
other chains, ligands and ions in the structure occlude the surface, so two values are
comparable only when the models were prepared the same way.

## Family membership

`protein.match_profiles` searches a protein against sequence profiles: the pinned Pfam 38.2
library or Stockholm alignments supplied in the request. It reports every alignment with its
protein span, profile span, bits, expectation value and the share of the profile covered.
The matches have three properties:

- They are MMseqs2 profile matches. The profiles are built from seed alignments, without
  the curated gathering thresholds of Pfam, so the hits can differ from Pfam's own
  matches in both directions. A claim that depends on family membership uses the match
  from the family database itself (Pfam).
- Expectation values depend on the size of the library searched, so they are
  comparable within one library.
- Overlapping matches, including members of one clan (a Pfam group of related families),
  are all retained. The record lists them without assigning a region to one family.

`alternative_alignments` (0 to 20; default 0) is the number of further alignments of the
same profile to report. Set it above 0 when a family can repeat in a protein. At 0, only the
best alignment per profile is reported and a repeat appears as a single domain. Lowering
`min_profile_coverage` reports fragments of a family and increases the number of matches
that cover only part of a domain. A search that returns nothing shows that no profile in the
library matched at the stated thresholds.

## Predicted tracks

`protein.predict_tracks` runs ESM3-small on the sequence alone and decodes three tracks:
InterPro-labeled regions, function keywords and eight-class secondary structure. All three
are predictions, and the reported maximum probabilities are the output probabilities of the
model. The result stores the per-residue probability arrays and the region thresholds.
Other thresholds can be applied to the arrays. `protein.match_profiles` searches for the
family that corresponds to a predicted InterPro region.

`tests/core/test_esm3_capabilities.py` checks the decoding on one protein, ubiquitin (1UBQ).
It asserts that at least 10 of the residues 23 to 34 (the alpha helix) are labeled H, that at
least 5 residues in each of the strands 1 to 7, 10 to 17 and 66 to 72 are labeled E, and that
no residue after residue 72 is labeled H or E. The test is skipped when the ESM environment
is unavailable.

## Genomic context

`genome.inspect_context` shows the genes around chosen proteins from a GFF3 file supplied in
the request. An identifier is matched exactly against the attribute named in the request
(`ID`, `protein_id` or `locus_tag`). A candidate is withheld when its sequence differs from
the supplied sequence with the same identifier, or when it matches a feature and is absent
from the supplied proteins. An identifier on more than one feature receives a window for
each feature.

Genomic neighborhood is physical adjacency on the contig. An operon, shared regulation or
shared function requires evidence beyond adjacency. A window clipped at a contig boundary
ends at the assembly edge, which can differ from the end of the chromosome or plasmid. The
window of a circular contig ends at the origin and does not continue across it. The
proteins of the displayed features that are in the supplied `proteins` FASTA are written to
a FASTA file, which `protein.search` and `sequence.generate_msa` accept as input.
