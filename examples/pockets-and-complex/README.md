# Pockets and complex

**Question.** Which residues form the predicted ligand-binding pockets of hemoglobin alpha?

`protein.inspect` produces the dossier of hemoglobin alpha from its sequence, its UniProt
record and the structure 1A3N, and `structure.detect_binding_pockets` runs P2Rank on that
structure. The report lists each predicted pocket with its score, probability and the
query positions of its residues in a table.

```bash
export ALAMBIQ_ENV_P2RANK=/path/to/p2rank/bin/python   # see docs/install.md
alambiq run examples/pockets-and-complex/campaign.yaml --report report.html
```

A second campaign, `complex.yaml`, predicts the pose of benzene in the engineered cavity of
T4 lysozyme L99A with `structure.predict_biomolecular_complex` (Boltz-2). The deposited
structure 181L permits comparison of the predicted pose with the experimental pose. The
campaign requires the Boltz environment and its resource. The run time depends on the
platform; the header of `complex.yaml` states "minutes on a GPU, much longer on a
CPU". The test suite does not run it.

**Needs.** The `dossier` extra, the P2Rank environment (`ALAMBIQ_ENV_P2RANK`; the
environment requires Java) and the `p2rank-2.5.1` resource. The `complex.yaml` campaign also
requires the Boltz environment and resource.

**What to look at.** The top-ranked pocket has a P2Rank score of 49.8 and a P2Rank
probability of 0.98. The pocket table lists, for each pocket, the number of residues and
their query positions (empty for pockets 2 and 4). The file named by `residues_table` in the
result lists the pocket membership of each of the 572 residues. The dossier lists which
chains map onto the query. The campaign `complex.yaml` predicts the pose of benzene; the
comparison with the ligand coordinates of 181L is made on the predicted structure, outside
the campaign.

**Limits.** A pocket is a prediction from structure-derived features by a model trained on
known ligand-binding sites. The accuracy of a predicted pose is assessed by comparison with
the experimental pose.

**Data.** As in [protein-first-look](../protein-first-look/README.md).
