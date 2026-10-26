# Python

`alambiq.api` runs campaigns and capabilities from Python. A campaign written in Python
is the same object as one loaded from YAML.

```python
from alambiq.api import run, step

outcome = run("examples/kinetics-from-measurements/campaign.yaml", run_dir="runs/kinetics")
outcome.ok                       # True when every step is completed or reused
outcome.outputs("fit")["parameters"]
outcome.report("report.html")    # the report, see Reports
```

`outcome.outputs(step_id)` returns the fields of the result of that step as a dictionary.
`run` accepts a campaign as a file path, as a dictionary or as a `Campaign` object. A
dictionary has the same keys as the YAML:

```python
outcome = run(
    {
        "id": "inspect",
        "steps": [
            {
                "id": "protein",
                "capability": "protein.inspect",
                "params": {"fasta": "my_protein.fasta"},
            }
        ],
    },
    run_dir="runs/inspect",
)
```

A relative path in a request is resolved against the directory of the campaign file. For a
dictionary, it is resolved against the `base` key, or against the working directory when
the dictionary has no `base` key.

## One capability

`step` runs one capability as a one-step campaign and returns the result object. The
keyword arguments `preset`, `run_dir` and `base` select a preset, the run directory and the
directory against which relative paths are resolved. Every other keyword argument sets the
request field of that name.

```python
result = step("protein.inspect", fasta="my_protein.fasta", run_dir="runs/step")
result.sequence_length
```

## Inspection and validation

```python
from alambiq.api import capability_list, describe, validate

capability_list()                  # every capability with its id, family, summary and engine environment
describe("assay.fit")              # the request, the result, and the method and limits
validate("campaign.yaml")          # a list of strings, one per problem found; empty when no problem is found
```

`validate` checks a campaign without running it. The checks are listed in
[Validation](concepts.md#validation). A missing engine environment is reported for that
step when it runs.

## Reuse

A rerun in the same `run_dir` reuses every step whose key is unchanged. `fresh=True`
recomputes all steps. After a step is added to an otherwise unchanged campaign, the rerun
computes only the new step. The key is defined in [Reuse](concepts.md#reuse).

## History and replay

```python
from alambiq.api import history, replay

history("runs/kinetics")                         # one summary per recorded run
replay("runs/kinetics", "fit", "replay-fit")     # writes replay-fit/campaign.yaml
```

Every run writes the run record `run.json` in its run directory, with a copy under
`history/`. [Run records](concepts.md#run-records) lists its contents. `replay` writes one
step of the run record as a campaign that runs that step again, with its input files copied
beside it.

## Failures

`outcome.steps` maps each step id to a `StepResult`. Its attributes include `status`
(`completed`, `reused`, `failed` or `skipped`), `error`, `notes` and `outputs`. A failed
step has its message in `error`. A step bound to a failed step, or to a step that was not
run, has the status `skipped`, and its `notes` name that step. The function `step` raises
`RuntimeError` when its capability fails.
