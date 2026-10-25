# Variant to construct

**Question.** What assembled plasmid sequence encodes the lysozyme variant S100R with a
linker and a His6 tag?

The campaign has 3 steps, from a protein sequence to an assembled plasmid design:

1. `construct.compose_coding` joins the lysozyme variant S100R, a linker and a His6 tag
   into one coding sequence (429 bp: 142 codons and a stop codon) from a stated codon
   table, and checks each part against its expected protein.
2. `dna.optimize_coding` changes synonymous codons of the coding sequence, leaving the
   protein unchanged, to remove the SmaI and BsaI sites and to keep the GC content between
   0.35 and 0.65 in every 50 bp window.
3. `construct.assemble` inserts the coding sequence into pUC19 at the SmaI site and returns
   the product plasmid, the primers for the insert and the annotated features of the
   product.

```bash
alambiq run examples/variant-to-construct/campaign.yaml --report report.html
```

**Needs.** The `construct` and `coding-design` extras.

**What to look at.** The recode changes 13 codons. Before the recode, the sequence contains
1 match of an avoided SmaI or BsaI site and 49 violations of the GC window limits (window GC
from 0.46 to 0.72). After the recode, neither occurs (window GC from 0.46 to 0.64). The mean
GC of the whole sequence changes from 0.569 to 0.550. The product is 3,115 bp with 8
features and 2 primers whose melting temperatures are reported. The report states that 3
vector features of pUC19 are not transferred to the product. The stored record lists each of
them with the reason for the omission (`removed_vector_features`).

**Limits.** The product is a sequence design for synthesis and assembly. The product has no
promoter or ribosome-binding site chosen for the protein; expression in a host requires
them. The codon table was written for the example. Synthesis by a DNA supplier requires
checking the sequence against the constraints of the supplier.

**Data.** pUC19 (`puc19.gb`) is an INSDC database entry. The protein is mature hen
egg-white lysozyme (129 residues) with an N-terminal methionine and the substitution S100R
(numbering of the mature protein), the variant that
[connected-investigation](../connected-investigation/README.md) selects. The linker, tag and codon choices were written for the example.
