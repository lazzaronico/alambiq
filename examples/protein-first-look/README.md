# Protein first look

**Question.** Which sequence, annotation and structure information does `protein.inspect`
report for this protein?

The campaign has one step, `protein.inspect`, with 3 inputs for human hemoglobin subunit
alpha (UniProt P69905): the sequence, the UniProt record (`uniprot`) and the deposited
structure 1A3N (`structure`). Without `uniprot` and `structure` the step returns a dossier
that contains sequence-derived information only. The UniProt record adds the reference
features and the structure adds the chain mapping table. With `coordinate_annotations: true`
the report also gives the secondary structure and the surface area.

```bash
alambiq run examples/protein-first-look/campaign.yaml --report report.html
```

**Needs.** The `quantitative` extra (the installation table lists `protein.inspect` under
it), the `dossier` extra (structure parsing) and the `trajectory` extra (secondary
structure and surface area from the coordinates). The command
`pip install -e ".[quantitative,dossier,trajectory]"` installs the three extras.

**What to look at.** The report opens with the sequence (142 residues) and the 214 UniProt
features, all of which are mapped onto the sequence. The structure section maps chains A and
C onto the query. Chains B and D are withheld with the reason "sequence mismatch or
ambiguous" (`sequence-mismatch-or-ambiguous` in the stored record): they are the beta
chains, whose sequences do not match the query. Secondary structure and surface area are
computed for all 572 observed residues of the 4 chains, including chains B and D, which are
withheld from the mapping; the mean surface area is given per observed residue.

**Limits.** A chain is mapped only when its polymer sequence contains the query exactly
once or is contained in the query exactly once. Chains are matched by exact sequence
containment, so a chain from a close homolog is withheld. The annotations are projected
onto the query from the UniProt record as given.

**Data.** `P69905.json` is the UniProtKB record (UniProt Consortium, CC BY 4.0),
`hemoglobin-alpha.fasta` the canonical sequence of P69905, and `1A3N.cif` the wwPDB
coordinates (CC0).
