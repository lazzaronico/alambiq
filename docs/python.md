# Python

The Python interface builds the same campaign object as the YAML loader and passes it to
the same execution function.

```python
from alambiq.api import run, step

outcome = run("examples/kinetics-from-measurements/campaign.yaml", run_dir="runs/kinetics")
outcome.ok                       # True when every step is completed or reused
outcome.outputs("fit")["parameters"]
outcome.report("report.html")    # the report, see Reports
```

`outcome.outputs(step_id)` returns the fields of the result of that step as a dictionary. `run` accepts
a campaign as a file path, as a dictionary or as a `Campaign` object. A dictionary has the
same keys as the YAML:

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

Alambiq resolves a relative path in a request against the directory of the campaign file.
For a dictionary, it resolves the path against the `base` key of the dictionary, or against
the working directory when the dictionary has no `base` key.

## One capability

`step` runs one capability as a one-step campaign and returns the result object. The keyword arguments `preset`, `run_dir` and `base` select a preset, the run directory and the directory against which relative paths are resolved; every other keyword argument sets the request field of that name.

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

`validate` reads the campaign and checks, without running anything:

- that every step names an existing capability;
- that every binding names a step of the campaign;
- that the steps have no circular dependency;
- the form of each request as far as it is known before the run: an unknown field, a
  missing required field, a value of the wrong type or a value outside the allowed
  choices.

A missing engine environment is reported for that step when it runs. See
[validation](concepts.md#validation).

## Reuse

A rerun in the same `run_dir` reuses every step whose capability version, request, input
files (name and SHA-256 digest of each), engine environment and resources are unchanged;
`fresh=True` recomputes all of them. After a step is added to an otherwise unchanged
campaign, the rerun computes only the new step. See [reuse](concepts.md#reuse).

## History and replay

```python
from alambiq.api import history, replay

history("runs/kinetics")                         # one summary per recorded run
replay("runs/kinetics", "fit", "replay-fit")     # writes replay-fit/campaign.yaml
```

Every run writes the run record `run.json` in its run directory, with a copy under
`history/`. The contents of the run record are listed in
[Run records](concepts.md#run-records). `replay` writes one step of the run record as a campaign
that runs that step again, with its input files copied beside it.

## Failures

In a campaign run, `outcome.steps` maps each step id to a `StepResult`. Its attributes
include `status` (`completed`, `reused`, `failed` or `skipped`), `error`, `notes` and
`outputs`. A
failed step has its message in `error`. A step bound to a failed step, or to a step that
was not run, has the status `skipped`, and its `notes` name that step. The function `step`
raises `RuntimeError` when its capability fails.
