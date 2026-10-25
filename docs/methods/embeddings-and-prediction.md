# Embeddings and learned predictions

Representation of proteins as vectors, exploration of a panel in that space, proposal of new
residues, and prediction of the stability effect of substitutions. The capabilities are
`protein.embed`, `protein.explore_embeddings`, `protein.complete` and
`protein.predict_stability_change`. `protein.embed` produces vectors.
`protein.explore_embeddings` computes a layout, neighbors, a hierarchy and representatives
from stored vectors and runs no model. `protein.complete` and
`protein.predict_stability_change` produce hypotheses.

## Embeddings

`protein.embed` processes each protein separately with ESM C 300M or ESM3-small. The
vector is the mean over residues of the last hidden state, excluding the start and end
tokens: after the final normalization for ESM C 300M and before the final normalization
for ESM3-small. The vectors have 960 and 1,536 values, respectively. A record names its
model, revision, layer and pooling, and holds the vectors of one model.

Vectors from different models, or from the same model at a different layer or pooling,
belong to different spaces. Vectors are comparable only within one space, whatever their
dimensions. Vectors are compared and combined only within one space.
`protein.explore_embeddings` refuses to merge records that differ and names the difference.
Within one space, interpretation rests on neighbor ranks and selections; raw distances are
on an uncalibrated scale.

A vector is the internal representation of a sequence in the model; mean pooling discards
residue order. Folding, stability and activity are measured properties of a protein, and the
vector is one input to a model of them. The models were trained on natural proteins;
designed or unusual sequences can fall in regions that the training data cover poorly. The
default device is the CPU; vectors computed on different devices can differ numerically.

## Panel layout

`protein.explore_embeddings` computes a two-dimensional layout of a panel with PCA or a
seeded UMAP, finds the nearest neighbor of each protein by cosine distance, builds an
average-linkage hierarchy, and proposes representatives: the medoid first, then the protein
farthest from those already chosen. Distances, neighbors and representatives are computed
from the full vectors, not from the two-dimensional layout.

A two-dimensional layout preserves part of the pairwise relations among the vectors. PCA
axes are the directions of greatest variance; the explained fractions are reported, and a
note is added when the first two axes explain less than 50 % of the variance. In a UMAP
layout only neighborhood membership is interpretable. The layout depends on its seed and
neighbor count; repeating the layout with a different seed checks a cluster. Representatives
are a coverage heuristic for choosing what to inspect or test first. For a panel of fewer
than 5 proteins the capability adds a note that the layout and the hierarchy carry little
information.

## Residue proposals

`protein.complete` masks the regions named in the request and fills them with ESM3-small,
once per seed. Everything outside the regions must be returned unchanged. The capability
checks every proposal independently of the model for three conditions: unchanged residues outside
the named regions, unchanged length, and standard residues only. A failure is stored with
its reason and is not offered as a candidate.

A proposal is a sequence sampled from ESM3-small conditioned on the residues outside the
named regions. The capability reports sampled sequences without a score. The model was
trained on natural proteins, and a sampled change can abolish activity. Catalytic residues
are protected by leaving them outside the named regions. A test with ubiquitin (regions 2 to
4, 3 proposals, seed 1) asserts that a proposal without changed residues is flagged; when
the test was written, the model restored the original residues of ubiquitin. The result
counts accepted proposals, including duplicates; a proposal that repeats an earlier one has
`duplicate_of` set. Different seeds can return the same sequence. Annotations and
measurements of the original protein describe the original; a proposal has annotations and
measurements only after they are made for it.

## Stability of substitutions

`protein.predict_stability_change` returns, for each single substitution on the backbone
of one chain, a predicted change in folding free energy in kcal/mol, mutant minus wild
type, with a positive value meaning destabilizing. A substitution to the same residue is a
zero-change control and returns 0 kcal/mol. The wild-type residue given for every
substitution is checked against the structure at the mapped position.

The chain must match the construct sequence residue for residue, and every residue must
have exactly one N, CA, C and O atom, each with an occupancy above 0 and no
alternate-location label, and the distance from the C of one residue to the N of the next
must be 1.0 to 1.8 Å. A structure with
missing residues is refused, because padding would let an author-numbering
offset shift a prediction onto the wrong residue. The structure is mapped by author
residue number and optional insertion letter, which are distinct from the residue indices
of the model.

The model takes the backbone of that chain as input. The prediction depends on the
backbone of the selected chain only; temperature, pH, ligands and oligomeric state are not
inputs. ThermoMPNN is a network trained on measured stability changes.

In the benchmark in this repository, on a beta-glucosidase, 128 substitutions were scored
and 78 had a finite measured thermal midpoint. The Spearman correlation of the negated
predicted change with that midpoint was 0.16 (position-bootstrap 95 % interval -0.10 to
0.38), and both criteria, fixed on 2026-10-04 before any prediction, failed
(`environments/thermompnn/BGLB_BENCHMARK.md`). The first criterion required the lower end of
the 95 % position-bootstrap interval of the correlation to exceed 0. The second required the
number of variants with a midpoint at least 1 °C above wild type among the 10 highest-ranked
of the 78 variants to exceed the random expectation, at a one-sided hypergeometric
probability of at most 0.05; the observed number was 3, against 1.28 expected, with a
probability of 0.113. The midpoint in that benchmark is derived from the activity measured
after a 30 min heat challenge between 30 and 50 °C (pNPG readout), so the result applies to
one enzyme and one functional endpoint. The capability returns a point estimate for each
substitution, without an interval; the error of the estimates is measured by a benchmark
such as the one above. The predicted quantity is folding stability; activity is a separate
property.
