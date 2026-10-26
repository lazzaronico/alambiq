# Reports

A report is one HTML file that summarizes a run. It contains its stylesheet inline and
requests no external resource, so a browser opens it from the local file system without a
server. It contains a script only when a structure is shown.

`alambiq run` writes a report when it receives `--report PATH`:

```bash
alambiq run campaign.yaml --run-dir runs/kinetics --report report.html
```

From Python, `outcome.report("report.html")` writes it.

## Contents

A navigation panel states the name of the campaign, the number of steps, whether the run
was complete or incomplete, and a link to each step. One section per step follows. Each
section contains:

- **Step name and timing**: the step id and either its duration in seconds or a `reused`
  badge. `reused` means that the stored result was used because the key of the step was
  unchanged.
- **Capability**: the id of the capability and its one-line summary.
- **Notes**: texts that a capability records about its result. A note leaves the status of
  the step unchanged. `alambiq run` prints each note on a line that starts with `note:`.
  Examples: a fit that finished at a bound (`operation.fit`), scenarios that extrapolate
  the fitted curve (`assay.balance`), and more draws discarded for a negative coefficient
  than 20 % of the requested samples (`operation.propagate`).
- **Result**: the result as presented by the capability. Each capability determines what is
  shown. A fit shows coefficients with their standard errors and units. A simulation shows
  final concentrations and the largest species-ledger balance error in M (labeled
  "species-ledger balance error" in the report). A selection shows the chosen batch and,
  for each reason for which a candidate was not chosen, the number of candidates. A
  capability without a renderer shows its outputs as JSON in a collapsible panel, cut at
  20,000 characters.
- **Error**: the message of a step that failed, shown in place of a result.

## Charts and views

Charts are drawn as inline SVG and need no script or external file. The plotted numbers can
be read from the SVG source. Each marker has `data-x` and `data-y` attributes, and each
line lists its points in `data-points`. The root element states the axis ranges and scales
(`data-x-low`, `data-x-high` and `data-x-scale`, and the same for y) and the pixel box of
the plot (`data-px-left`, `data-px-right`, `data-px-top` and `data-px-bottom`). A line
lists at most 2000 points and a series shows at most 5000 markers.

- **Fits** show the measurements, the fitted curve, and two shaded bands: the 95 % interval
  of the fitted curve (`mean_interval_95`) and the 95 % prediction interval of a new
  measurement (`prediction_interval_95`).
- **Time courses** show the concentration of each species of a simulation against time, for
  at most 8 species. For `operation.fit` and `operation.compare` they show the measured
  values beside the predictions at the same times, in one chart per experiment, quantity
  and target. At most 6 charts are drawn for a step; the remaining quantities are in the
  stored record.
- **Selections** show the selection value of each chosen candidate, in the order of
  selection.
- **Embeddings** show every protein at its coordinates in a two-dimensional projection (PCA
  or UMAP, as requested), with the representative proteins marked.
- **Alignments** show a plot of entropy and gap fraction along the alignment, above the
  aligned rows in blocks, with residues shaded blue where they equal the query's and
  orange where they differ. An alignment shows at most 100 rows and 3000 columns.
- **Trajectories and free-energy profiles** show an observable against time and the free
  energy against the coordinate. The profile of `simulation.analyze_reaction_free_energy`
  has a band of one standard error across replicas.
- **Structures** are drawn with a rotatable viewer (3Dmol.js, BSD-3-Clause, bundled in the
  package). Its script is added to a report only when a structure is shown. The renderers
  of `structure.prepare_protein_system`, `structure.prepare_protein_ligand_system` and
  `structure.prepare_predicted_complex_for_dynamics` remove the water and simple ions of a
  PDB structure before display. A structure larger than 4,000,000 bytes is left out with a
  note.

## Run history

When the run directory holds at least 2 recorded runs of the campaign, the report ends with
a table of those runs, newest first: start time (UTC), outcome, steps computed, reused and
failed, and duration in seconds. [Run records](concepts.md#run-records) describes the run
records, `alambiq history` and `alambiq replay`.

```bash
alambiq history runs/kinetics
alambiq replay runs/kinetics fit --out replay-fit
alambiq run replay-fit/campaign.yaml
```

## Reading a report

`alambiq capability show ID` prints the method, the assumptions and the limits of a
capability.

A `reused` badge describes the run: a reused step returns the result computed earlier from
the same inputs. A run marked incomplete has at least one step that failed or was skipped.
A skipped step is bound to a failed step or to a step that was not run. The sections show
which steps failed. Steps that do not depend on a failed step contain their results.

## Stored records and limits

Charts are drawn from the stored record of each step. If the record is missing or
unreadable, the chart is omitted and the rest of the report is written. The complete records
(every row, every residual, every point of a trajectory) are files in the artifact store of
the run directory, stored under their SHA-256 digest. The result of each step gives the path
to its own record.

The capability version, the resolved request and the reuse key of a step are stated in
`run.json`.

Some renderers show a truncated SHA-256 digest. `protein.inspect` shows the first 16
characters of the digest of the input sequence, `protein.embed` the same for each protein,
and `construct.assemble` the same for the assembled product sequence.
`simulation.run_qmmm_dynamics` shows the first 12 characters of the digest of the
Hamiltonian.

## Storage

A report is written to the path given by `--report` or `outcome.report`. A new run with the
same `--run-dir` and `--report PATH` reuses every unchanged step and writes a new report.
