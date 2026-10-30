# Embeddings and learned predictions

A protein is represented as a vector from a language model. Five capabilities use these
models:

- `protein.embed` produces vectors.
- `protein.explore_embeddings` computes a layout, neighbors, a hierarchy and
  representatives from stored vectors and runs no model.
- `protein.complete` proposes new residues.
- `protein.predict_stability_change` predicts the stability effect of substitutions.
- `protein.predict_melting_temperature` estimates the melting temperature of a protein from
  its sequence.

The last two produce hypotheses.

## Embeddings

`protein.embed` processes each protein separately with ESM C 300M or ESM3-small. The
vector is the mean over residues of the last hidden state, excluding the start and end
tokens. The hidden state is taken after the final normalization for ESM C 300M and before
it for ESM3-small. The vectors have 960 and 1,536 values, respectively. A record names its
model, revision, layer and pooling, and holds the vectors of one model.

Vectors from different models, or from the same model at a different layer or pooling,
belong to different spaces. Vectors are compared and combined only within one space,
whatever their dimensions. `protein.explore_embeddings` refuses to merge records that
differ and names the difference. Within one space, interpretation rests on neighbor ranks
and selections. Raw distances are on an uncalibrated scale.

A vector is the internal representation of a sequence in the model. Mean pooling discards
residue order. Folding, stability and activity are measured properties of a protein, and
the vector is one input to a model of them. The models were trained on natural proteins.
Designed or unusual sequences can fall in regions that the training data cover poorly.
The default device is the CPU. Vectors computed on different devices can differ
numerically.

## Panel layout

`protein.explore_embeddings` computes a two-dimensional layout of a panel with PCA or a
seeded UMAP, finds the nearest neighbor of each protein by cosine distance, and builds an
average-linkage hierarchy. It proposes representatives: the medoid first, then the protein
farthest from those already chosen. Distances, neighbors and representatives are computed
from the full vectors, not from the two-dimensional layout.

A two-dimensional layout preserves part of the pairwise relations among the vectors. PCA
axes are the directions of greatest variance. The explained fractions are reported, and a
note is added when the first two axes explain less than 50 % of the variance. In a UMAP
layout only neighborhood membership is interpretable. The layout depends on its seed and
neighbor count, and repeating the layout with a different seed checks a cluster.
Representatives are a coverage heuristic for choosing what to inspect or test first. For
a panel of fewer than 5 proteins the capability adds a note that the layout and the
hierarchy carry little information.

## Residue proposals

`protein.complete` masks the regions named in the request and fills them with ESM3-small,
once per seed. Everything outside the regions must be returned unchanged. A proposal is a
sequence sampled from ESM3-small conditioned on the residues outside the named regions.
The capability reports sampled sequences without a score.

The capability checks every proposal independently of the model for three conditions:
unchanged residues outside the named regions, unchanged length, and standard residues
only. A failure is stored with its reason and is not offered as a candidate.

The model was trained on natural proteins, and a sampled change can abolish activity.
Catalytic residues are protected by leaving them outside the named regions. A test with
ubiquitin (regions 2 to 4, 3 proposals, seed 1) asserts that a proposal without changed
residues is flagged; when the test was written, the model restored the original residues
of ubiquitin.

The result counts accepted proposals, including duplicates. A proposal that repeats an
earlier one has `duplicate_of` set. Different seeds can return the same sequence.
Annotations and measurements of the original protein describe the original. A proposal has
annotations and measurements only after they are made for it.

## Stability of substitutions

`protein.predict_stability_change` returns, for each single substitution on the backbone
of one chain, a predicted change in folding free energy in kcal/mol, mutant minus wild
type. A positive value means destabilizing. A substitution to the same residue is a
zero-change control and returns 0 kcal/mol. The wild-type residue given for every
substitution is checked against the structure at the mapped position.

The model takes the backbone of that chain as input. ThermoMPNN is a network trained on
measured stability changes. The prediction depends on the backbone of the selected chain
only; temperature, pH, ligands and oligomeric state are not inputs. The capability
returns a point estimate for each substitution, without an interval. The predicted
quantity is folding stability; activity is a separate property.

The chain must match the construct sequence residue for residue. Every residue must have
exactly one N, CA, C and O atom, each with an occupancy above 0 and no alternate-location
label. The distance from the C of one residue to the N of the next must be 1.0 to 1.8 Å.
A structure with missing residues is refused, because padding would let an author-numbering
offset shift a prediction onto the wrong residue. The structure is mapped by author
residue number and optional insertion letter, which are distinct from the residue indices
of the model.

The error of the estimates is measured by a benchmark. In the benchmark in this
repository, on a beta-glucosidase, 128 substitutions were scored and 78 had a finite
measured thermal midpoint. The Spearman correlation of the negated predicted change with
that midpoint was 0.16 (position-bootstrap 95 % interval -0.10 to 0.38). Both criteria,
fixed on 2026-10-04 before any prediction, failed
(`environments/thermompnn/BGLB_BENCHMARK.md`).

- The first criterion required the lower end of the 95 % position-bootstrap interval of
  the correlation to exceed 0.
- The second required the number of variants with a midpoint at least 1 °C above wild type
  among the 10 highest-ranked of the 78 variants to exceed the random expectation, at a
  one-sided hypergeometric probability of at most 0.05. The observed number was 3, against
  1.28 expected, with a probability of 0.113.

The midpoint in that benchmark is derived from the activity measured after a 30 min heat
challenge between 30 and 50 °C (pNPG readout), so the result applies to one enzyme and one
functional endpoint.

## Melting temperature

`protein.predict_melting_temperature` returns, for each protein of a FASTA, the melting
temperature in degrees Celsius that each of the three replicas of TemBERTure Tm predicts, with
their mean and standard deviation. The input is the sequence alone. Ligands, metals, the pH and
the oligomeric state are not inputs.

TemBERTure Tm is the ProtBERT-BFD protein language model with an adapter and a regression head.
The authors trained it on melting temperatures of the Meltome Atlas (13 organisms), with
the data split by clusters of sequences at 50 % identity. The weights are the pinned
resource `temberture-tm`; `alambiq resource provision temberture-tm --yes` downloads them
(about 1.7 GB: the ProtBERT-BFD model from Hugging Face and the adapters from GitHub).

The model reads one residue per token and at most 512 tokens, which is the first 510 residues.
A longer protein is marked `truncated` and the result carries a note. A protein of 10 to 5,000
residues from the 20 amino acids and X is accepted. The estimate is the mean of the replicas,
and the spread between them is the indicator of reliability to read first: the authors report
that the replicas, which differ in their random seeds, vary markedly. A spread above 15
degrees Celsius adds a note.

The authors report a Pearson correlation of 0.78 between predicted and measured melting
temperatures on their test set, and observe that the predictions are bimodal: concentrated
below 60 degrees Celsius and above 80 degrees Celsius. Use the values to rank proteins that
differ widely, for instance homologs from organisms that grow at different temperatures. No test of the model on the variants of one protein is
part of this capability, so a difference between a variant and its wild type is not validated.

The reference code of the authors leaves the network in training mode, which keeps the 0.3
dropout of the regression head active, so that its predictions vary from call to call. The
capability evaluates the network in evaluation mode, so that its predictions are deterministic.
The two agree on average: for the example enzyme of the repository, the evaluation-mode value of the
first replica is 77.9 degrees Celsius and two training-mode calls gave 77.7 and 72.4. The
authors' repository pins Adapters 0.1.1 and PyTorch 1.10.2; the capability runs their files in
Adapters 1.3.0 and PyTorch 2.15.0. The tests check that the model loads, that a protein gives
the same values alone and in a batch, that residues beyond the 510th do not change the result,
and that a repeated run reproduces the values.
