# TbSADH: from the sequence to a simulated reactor

**Question.** Where does the predicted complex of the alcohol dehydrogenase TbSADH with its
substrate acetophenone place the substrate relative to the catalytic zinc and the NADP of the
deposited structure, and what does the measured progress of the same reaction predict for a
reactor?

One campaign follows one enzyme, the secondary alcohol dehydrogenase of *Thermoanaerobacter
brockii* (UniProt P14941, 352 residues, a homotetramer with a catalytic zinc and one NADP in
each subunit), and one substrate, acetophenone. The campaign has 11 steps in two branches.

| Step | Capability | Role |
| --- | --- | --- |
| `dossier` | `protein.inspect` | reads the sequence and the deposited structure 1YKF; its `sequence` is the input of `complex` |
| `pockets` | `structure.detect_binding_pockets` | predicts the pockets of the deposited tetramer (P2Rank) |
| `complex` | `structure.predict_biomolecular_complex` | predicts the tetramer with four NADP, four zinc ions and acetophenone (Boltz-2, seed 1), each chain with the alignment `P14941.a3m` |
| `predicted` | `protein.inspect` | reads the predicted structure against the same sequence |
| `agreement` | `structure.compare` | superposes chain A of the prediction on chain A of 1YKF |
| `site` | `structure.measure_ligand_site` | measures the substrate against the zinc and the NADP of the prediction, and against those of 1YKF after superposing the predicted tetramer on the deposited one |
| `free-first-order`, `free-saturation`, `hydrogel-first-order`, `hydrogel-saturation` | `operation.fit` | fit an effective first-order law and a saturation law to the progress of acetophenone reduction by free and hydrogel-assembled TbSADH |
| `reactor` | `operation.simulate` | simulates the fitted hydrogel operation |

The molecular branch and the laboratory branch share the enzyme and the substrate. No number
passes between them: the fits of the progress curves do not read the structure.

```bash
alambiq run examples/tbsadh-sequence-to-reactor/campaign.yaml --report report.html
```

**Needs.** The `dossier` and `quantitative` extras (the `site` step reads structures with Gemmi and fits them with NumPy). The molecular branch needs the P2Rank and
Boltz-2 environments and their resources (`ALAMBIQ_ENV_P2RANK`, `ALAMBIQ_ENV_BOLTZ`; see
[install](../../docs/install.md)) and a CUDA GPU for the prediction of the tetramer (1,408
residues). The laboratory branch needs no engine. The test suite runs the dossier, the laboratory branch and the chain assignment of the
superposition on 1YKF, and does not run the prediction.

**The alignment.** `P14941.a3m` is the result of `sequence.generate_msa` for `P14941.fasta`
(`msa.yaml`; UniRef50 release 2026_02, three iterations, 4,158 rows, of which 4,055 are diverse
sequences). The database is 138 GB and its search ran on a Mac in 646 s. The file is stored here
as an input of the campaign, and the four chains name it, so the campaign does not need the
database. Without an alignment, Boltz-2 predicted the tetramer with an ipTM of 0.17 and a
complex pLDDT of 0.45 (a prediction without cofactors).

**Run on one NVIDIA GB10 (the DGX Spark).** Wall time per step: `dossier` 0.1 s, `pockets` 3 s,
`complex` 1,765 s, `predicted` 0.1 s, `agreement` 0.3 s, `site` 0.4 s, the four fits 4.9, 17.5, 8.6 and 25.5 s,
`reactor` 0.02 s; 1,794 s for the campaign.

**What to look at.**

- **`pockets`**: P2Rank ranks four pockets with probabilities from 0.90 to 0.96 (scores 37.7,
  35.5, 30.4 and 24.3), and a fifth with 0.51. Each
  of the first four contains query positions 37, 59 and 150, the Cys37, His59 and Asp150 that
  bind the catalytic zinc.
- **`complex`**: ipTM 0.737, complex pLDDT 0.925, ligand ipTM 0.790. The four NADP are chains E
  to H and the four zinc ions chains I to L. The structure holds one acetophenone, chain ACP. A
  prediction of the same tetramer with the substrate alone had an ipTM of 0.959; the ipTM here
  includes the chains of the cofactors, and the chain pTM of each zinc ion is 0.
- **`agreement`**: chain A of the prediction superposes on chain A of 1YKF with an alpha-carbon
  RMSD of 0.51 Å over 352 residues.
- **`site`**: the predicted tetramer superposes on the deposited tetramer with an
  alpha-carbon RMSD of 0.72 Å over 1,408 residues; the predicted chains A, B, C and D
  correspond to the deposited chains B, A, C and D. The substrate is `LIG1` of chain ACP;
  Boltz-2 names its atoms by canonical rank, and the result lists each atom with the atoms it
  is bonded to, which identifies O9 as the carbonyl oxygen and C10 as the carbonyl carbon.
  - In the prediction, O9 lies 2.30 Å from the zinc of chain L and C10 3.52 Å from C4 of the
    nicotinamide of the NADP of chain H, the carbon that receives the hydride; their other
    subunits are 40 to 47 Å away. The nearest atom of the prediction outside the substrate is
    that zinc (to O9, 2.30 Å) and then Ser39 OG (to C14, 2.79 Å), Met285 CE, Trp110 CZ3, Leu294
    CD1 and His59 NE2 (2.98 to 3.88 Å); no atom is closer than 2.3 Å to the substrate.
  - Against 1YKF, O9 lies 2.51 Å from the zinc and C10 3.54 Å from C4 of the NADP of the
    subunit it is in (deposited chain D). The tetramer is nearly symmetric, and 4 of the 24
    assignments of predicted chains to deposited chains fit within 0.1 Å of the smallest RMSD;
    over them the nearest distance of O9 to a zinc of 1YKF ranges from 1.91 to 2.94 Å and that
    of C10 to C4 from 3.54 to 3.65 Å. The distances inside the prediction do not depend on the
    assignment.
  - The predicted zinc ions lie 0.5 to 0.7 Å from the deposited ones, and the predicted NADP
    deviates from the deposited NADP by 1.3 Å in RMSD over its 48 heavy atoms; the carboxamide
    of the nicotinamide is rotated (N7N 5.1 Å and O7N 3.8 Å from the deposited atoms). Because
    of that, the nearest atoms of 1YKF to the substrate are N7N and O7N of its NADP, at 0.72 to
    1.2 Å for four of the nine atoms, and `contacts` of the result lists them.
- **Laboratory branch**: the numbers of
  [operation-from-progress-curves](../operation-from-progress-curves/README.md): the standardized
  RMSE of the hydrogel saturation fit is 1.00 on the fitted points and 1.85 on the evaluation
  points, and the simulation ends at 480 s with 0.224 mM acetophenone and 4.776 mM converted
  substrate.

**Rerun.** A second campaign in a separate run directory wrote the same predicted structure
(byte for byte), the same confidences, the same fits, reactor simulation and `site` record and
fitted structure, with the same step keys.

**Limits.**

- TbSADH, or a close homologue with a substrate, may be among the training data of Boltz-2, so
  the agreement with 1YKF does not show that the model predicts new complexes. It checks the
  chain on a known answer.
- 1YKF holds NADP and zinc and no substrate. The distances in the prediction place the
  substrate relative to the cofactor and the metal that the prediction holds, with a Zn–O
  distance of 2.30 Å and a C–C4N distance of 3.52 Å; they are the geometry of one sample, and
  do not show that this is the productive binding mode.
- The prediction places the carboxamide of the NADP differently from 1YKF, so the substrate
  overlaps the deposited carboxamide; the comparison with 1YKF is a comparison of the
  tetramer, the zinc ions and the nicotinamide ring.
- The distances of `site` against 1YKF depend on which of the equivalent chain assignments is
  used, as stated above; the campaign reports the one with the smallest RMSD and the range
  over all of them.
- The prediction is one sample (seed 1) with one alignment. The campaign runs the prediction
  once, and the number of rows of the alignment depends on the database release.
- The fits describe one assay at fixed conditions (5 mM substrate, 1 mg/mL enzyme, pH 7.4,
  30 °C) with four fitted points for each saturation fit; see the limits of
  [operation-from-progress-curves](../operation-from-progress-curves/README.md).

**Data.** `1YKF.cif` is the deposited structure of TbSADH with NADP (RCSB PDB 1YKF, CC0;
Korkhin et al., J. Mol. Biol. 1998). `P14941.fasta` is the UniProt sequence P14941 (CC BY 4.0;
the identifier of the header is reduced to the accession, because the alignment capability
accepts letters, digits, underscore, dot and hyphen). The deposited sequence and the UniProt
sequence are identical. The progress curves and the definition files are those of
[operation-from-progress-curves](../operation-from-progress-curves/README.md) (Chen et al.,
*Nature Communications* 2023, doi:10.1038/s41467-023-37921-y, Fig. 3d1 source data, CC BY 4.0).
