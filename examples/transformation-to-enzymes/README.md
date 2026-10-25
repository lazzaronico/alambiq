# Transformation to enzymes

**Question.** Which candidate sequences rank above negative controls by similarity to
characterized (S)-2-haloacid dehalogenases, and do they conserve the active-site residues
reported in the literature?

The reaction is the hydrolysis of an (S)-2-haloacid (RHEA:11192). The campaign has 2 steps:

1. `enzyme.retrieve_candidates` scores 7 candidates and 3 experimentally characterized
   (S)-2-haloacid dehalogenases (the references) by the highest bit score of their
   alignments to a different reference. `candidates.fasta` holds 7 proteins: 2 without a
   control label (P24070, Q7DI63) and 5 negative controls. `references.fasta` holds the 3
   references, which are also the positive controls. The step evaluates the ranking against
   the controls named in the request: the 3 positive controls and the 5 negative controls
   (called hard negatives in the result).
2. `enzyme.characterize_candidates` maps 9 active-site residues of the reference Q53464
   (mutagenesis work cited in the campaign) onto each candidate through its alignment to
   the reference and assigns each candidate one of 3 statuses: advance, withhold or reject. The campaign
   uses the default thresholds of the capability. A protein is rejected when
   it has no alignment to the reference, scores below 100 bits in step 1 or conserves less
   than 1.0 of the 9 residues; it is withheld when the alignment covers less than 0.8 of the
   reference or less than 1.0 of the 9 residues lie inside the alignment.

```bash
export ALAMBIQ_ENV_MMSEQS=/path/to/mmseqs2/bin/python   # see docs/install.md
alambiq run examples/transformation-to-enzymes/campaign.yaml --report report.html
```

**Needs.** The MMseqs2 environment (`ALAMBIQ_ENV_MMSEQS`).

**What to look at.** All 3 positive controls rank above every negative control: the lowest
positive bit score is 186 and the highest negative bit score is 48, a margin of 138 bits.
The control AUROC is 1.0, against 0.8 for a baseline that scores sequence length alone. The
3 positive controls and the 2 candidates that are not controls (P24070, Q7DI63) advance,
each with all 9 residues conserved; the 5 negative controls are rejected. Four of them
(A0A140N5J7, O06995, P22643, Q8KLS9) have no alignment to Q53464 and score 0 bits in step 1;
P32662 scores 48 bits and conserves 4 of the 9 residues.

**Limits.** With 3 positive and 5 negative controls the AUROC takes values in steps of 1/30
(ties count one half), and 1.0 requires only that every positive control outscores every
negative control. A candidate that advances conserves all 9 residues; the status refers to sequence
similarity and residue conservation. The campaign lists the 9 residues from the cited
mutagenesis work, and the capability uses the list as given.

**Data.** UniProtKB sequences (CC BY 4.0). The sequences and controls belong to the HAD-S2
dehalogenase panel; the 9 residues are cited in doi:10.1074/jbc.271.34.20322.
