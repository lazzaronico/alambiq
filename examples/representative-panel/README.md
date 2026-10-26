# Representative panel

**Question.** Which 3 proteins of a panel are the most widely separated in the
language-model embedding space?

`protein.explore_embeddings` takes mean-pooled language-model vectors for a panel (here 6
human proteins embedded with ESM C 300M). It projects the vectors with PCA and lists the
nearest neighbor of each protein by cosine distance. It recommends 3 proteins as
representatives.

```bash
alambiq run examples/representative-panel/campaign.yaml --report report.html
```

The vectors are stored in `embeddings.json`, so the campaign needs neither the ESM
environment nor the model. The campaign `embed.yaml` regenerates the vectors with
`protein.embed`. It requires the ESM engine environment and the ESM C resource
([installation](../../docs/install.md)).

**Needs.** The `quantitative` extra.

**What to look at.** The first two principal components explain 65.5 % and 17.1 % of the
variance, respectively. The report recommends hemoglobin gamma-1 (P69891), actin (P60709)
and myoglobin (P02144). The nearest-neighbor table lists the 3 hemoglobin subunits (P69905,
P68871, P69891) at cosine distances of 0.010 to 0.030 from their nearest neighbors.

**Limits.** The layout describes the model embeddings of the sequences. Mean pooling
discards positional information. The recommendation and the nearest neighbors are computed
from the full 960-dimensional vectors, and the 2-dimensional projection explains 82.6 % of
the variance. The recommendation selects the medoid (the protein with the smallest mean
cosine distance to all proteins) first. It then repeatedly selects the protein whose cosine
distance to the nearest chosen protein is largest. It uses the vectors only.

**Data.** UniProtKB sequences (CC BY 4.0), listed in `representation-panel.sources.json`.
The ESM C 300M weights are distributed under the license `MIT-with-third-party-notices`
plus an acceptable-use policy; `embeddings.json` states the model, revision and digests.
