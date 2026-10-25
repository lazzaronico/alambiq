# Methods

One page per method area. Each page covers the capabilities of one area: what their
results state and the limits of those results. `alambiq capability show <id>` prints the
request, result, units and limits of one capability.

- [Measurements](measurements.md): measurement tables, replicate structure, response
  models, the four uncertainties (coefficient standard error, mean-response interval,
  future-observation interval and measured predictive error), and projection of a fit onto
  a batch target.
- [Operation](operation.md): reaction simulation, fitting of parameters to measured
  progress, comparison of predictions with measurements, and accounting of output,
  catalyst use and consumed inputs.
- [Candidate models](candidate-models.md): partition of a candidate panel into training,
  evaluation and pool, comparison of model error with the training-mean baseline, feature
  representations, and batch selection.
- [Constructs](constructs.md): composition of a coding insert from parts, synonymous
  recoding, and placement into a vector.
- [Proteins](proteins.md): protein inspection, family membership by profile, predicted
  tracks, and genomic context.
- [Search and alignment](search-and-alignment.md): panel search, alignment construction and
  reading, and ranking of candidates against references with controls.
- [Embeddings and learned predictions](embeddings-and-prediction.md): embedding vectors and
  their spaces, panel layout, residue proposals, and prediction of the stability effect of
  substitutions.
- [Reactions](reactions.md): reaction equations, thermodynamic estimates, route search, and
  prediction of activity across enzymes and reactions.
- [Structure and dynamics](structure-and-dynamics.md): structure comparison and prediction,
  system preparation, molecular dynamics, trajectory inspection, and comparison of replicas
  with each other and with measured couplings.
- [Binding free energy and QM/MM](binding-and-qmmm.md): relative binding free energies from
  OpenFE legs and their diagnostics, and construction, sampling and analysis of a QM/MM
  reaction from a modeled enzyme to a free-energy profile.
- [Proposing variants](variant-proposal.md): substitution scoring with a language model,
  ranking with stability and homolog evidence, a benchmark on measured variants, and
  selection among candidates.
