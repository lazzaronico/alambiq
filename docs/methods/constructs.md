# Constructs

Composition of a coding insert from parts, synonymous recoding under constraints, and
placement of the insert into a vector. The capabilities are
`construct.compose_coding`, `dna.optimize_coding` and `construct.assemble`. All three
operate on sequences.

## Composing an insert

Parts are joined in the order given, in one reading frame, with a single stop codon
appended. The complete insert is translated under genetic code 11 and compared with the
concatenated expected proteins. Each part records its span in both the DNA and the
protein, one-based and inclusive; the span excludes the appended stop codon.

Only the first codon of the fusion is translated as an initiator; an internal codon that
could act as an alternative start is translated as its ordinary amino acid. A start codon, a
tag, a linker, a signal peptide or any other segment, such as an initiation segment, is
entered as an explicit part (role `protein`, `tag` or `linker`) so that it appears in the
expected translation. The first part must begin with a start codon of genetic code 11; the
composition fails otherwise.

A part whose DNA ends with a stop codon declares `remove_terminal_stop` to remove it
before joining. A part declared without a terminal stop codon is an error. A stop codon
left inside the insert fails the translation check.

When the request supplies a codon table, every part is encoded from it and no part supplies
DNA: a part that has `coding_dna` together with the codon table is refused, and
`remove_terminal_stop` is ignored, because an encoded part holds no stop codon. The table
assigns one codon to each of the 20 standard amino acids, and each codon is checked against
genetic code 11. The codon table is part of the request, is recorded with the result and
sets the codon of every amino acid in every part. `dna.optimize_coding` can later change the
codons without changing the protein.

## Synonymous recoding

`dna.optimize_coding` changes synonymous codons only, keeping the first and last codon
fixed, to remove the motifs listed in the request and bring the GC content within stated
bounds. Motifs are literal ACGT strings, checked on both strands including overlapping
matches. An enzyme site is entered as its recognition sequence. GC bounds apply to the whole
sequence or to every window of a stated length.

The capability re-checks every constraint on the output of the search before reporting a
result: the translation, the terminal codons, each motif on both strands and the GC
fraction in every window. An output that violates a constraint raises an error. When the
search ends without a sequence that satisfies all constraints, the rejected candidate
is stored for inspection and the result has the status `failed`. The status
covers both a failed search and a constraint set that no synonymous sequence of the
protein satisfies.

The request field `seed` reproduces the result of a recoding run for the same package
versions, which the stored record lists. The search draws from the global random state of
NumPy; Alambiq sets it from `seed` and restores the previous state after the run.

Synonymous changes can alter translation efficiency, mRNA folding, internal regulatory
elements and co-translational folding; the constraints of the capability are the motifs
and the GC bounds.

## Placing an insert into a vector

`construct.assemble` simulates the linearization of a circular vector with a named
blunt-cutting enzyme, the amplification of the insert with designed tailed primers, and the
joining of the two by Gibson assembly with an overlap of 25 to 60 bases (default 35). It
reports the complete circular product: its sequence, its annotated GenBank file, the
primers, and the features retained or removed. The coding sequence of the product is
translated and compared with the expected protein, and an assembly that does not yield
exactly one unique circular product fails.

There are two input forms. A plain coding insert is one complete coding sequence, and any
tag or linker is present in both that sequence and the expected protein. An annotated
cassette, supplied as linear GenBank, is used for an insert with noncoding sequence; it
carries its own features, and the bases upstream and downstream of the annotated coding
sequence are not translated.

Vector features that the linearization or the join does not retain, because they overlap
the edit or cannot be transferred intact, are listed as removed and are not included in
the annotation of the product.

Melting temperatures are computed with `pydna.tm.tm_default`, a nearest-neighbor model,
on the annealing footprint alone. A computed melting temperature is a property of the
footprint; PCR performance of a primer is determined by experiment.

## Limits

`construct.compose_coding` composes and verifies a sequence; `dna.optimize_coding` and
`construct.assemble` design and verify sequences. The translation checks confirm that a
DNA sequence encodes the claimed protein. Folding, solubility, stability, activity and
expression are determined by experiment. `dna.optimize_coding` avoids the motifs given in
the request.

The assembly is simulated from the supplied vector and insert sequences, the named enzyme
and the designed primers; the product sequence is the simulated result of these inputs.
