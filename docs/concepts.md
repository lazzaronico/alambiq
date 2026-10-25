# Concepts

The terms below have the same meaning in YAML and in Python.

## Capabilities

A capability is a function with a declared request type, a declared result type and documented limits. It is identified by `prefix.action`, for example `assay.fit`, `operation.simulate` or `candidate.select`. The prefix is the part of the identifier before the dot; the family is a separate attribute of the capability. Each capability has one family: characterize, investigate, engineer, measure or operate. The capability `assay.fit` is in the measure family.

`alambiq capability list` prints every capability with its family and a one-line summary. `alambiq capability show assay.fit` prints the request, the result, and the method and limits documented in the function.

## Campaigns and steps

A campaign is an identifier and a list of steps, written in YAML or Python. A step is one use of a capability in a campaign, with an id, a capability, parameters (`params`) and bindings (`bind`). A binding is the assignment of an output of one step to a field of the request of another.

The campaign in `examples/kinetics-from-measurements/campaign.yaml` has two steps. The step `fit` takes its field `measurements` from the output `record` of the step `intake`.

```yaml
id: kinetics-from-measurements
steps:
  - id: intake
    capability: assay.inspect
    params:
      table: rates.csv
      conditions:
        c0: Substrate 0.05 mM
        c1: Substrate 0.1 mM
        c2: Substrate 0.25 mM
        c3: Substrate 0.5 mM
        c4: Substrate 1 mM
        c5: Substrate 2.5 mM
        c6: Substrate 5 mM
        c7: Substrate 8 mM
        c8: Substrate 20 mM
        c9: Substrate 30 mM
      condition_values:
        c0: {substrate: {value: 0.05, unit: mM}}
        c1: {substrate: {value: 0.1, unit: mM}}
        c2: {substrate: {value: 0.25, unit: mM}}
        c3: {substrate: {value: 0.5, unit: mM}}
        c4: {substrate: {value: 1, unit: mM}}
        c5: {substrate: {value: 2.5, unit: mM}}
        c6: {substrate: {value: 5, unit: mM}}
        c7: {substrate: {value: 8, unit: mM}}
        c8: {substrate: {value: 20, unit: mM}}
        c9: {substrate: {value: 30, unit: mM}}
  - id: fit
    capability: assay.fit
    bind:
      measurements: intake.record
    params:
      sample_id: enzyme
      metric: initial-rate
      x_quantity: substrate
      x_unit: mM
      y_unit: uM/min
      model: michaelis_menten
```

A binding is written `field: step.output`, where `output` is the name of a field of the result of that step, or `field: step.path` for a path into a nested field such as `fit.operations.my-reaction`. A number in a path selects an item of a list: `align.chains.0.alignment` is the alignment of the first chain. The field on the left accepts the same notation, so a request field that holds a list of items can be assembled from several steps: `replicas.0.details: first.details` sets the `details` field of the first item of `replicas`. The other fields of that item come from `params`. Missing lists and items are created. An index may be at most one past the end of the list; a larger index is an error.

The order of steps in the file does not affect execution. The bindings determine which step precedes which, and steps with no dependency between them run concurrently, at most 4 at a time.

A preset is a named set of request-field values declared by a capability. A step selects a preset with its `preset` key. A field set in `params` overrides the same field of the preset.

A campaign is run from Python with `alambiq.api.run`, which accepts a file path, a dictionary with the same keys as the YAML, or a `Campaign` object. [Python](python.md) describes the interface.

## Requests, results and records

A request is the input of a capability, made of fields. The fields of a request are set by the `params` and the `bind` of a step. A result is the typed output of a capability. An output of a step is a field of its result. A record is the JSON file that holds the full detail of a result: every row of a table, every point of a trajectory, every residual.

The artifact store is the directory `artifacts/` of the run directory; it holds each stored file under the SHA-256 digest of its content. A capability whose result has a `record` field writes the record to the artifact store, and the field holds the path of the record. A binding such as `measurements: intake.record` therefore passes the path of a file to the next step. The result of a capability with a `record` field holds summary values; the complete detail is in the record. Reuse compares the digests of the files that a request names.

## Runs

A run is one execution of a campaign. `alambiq run campaign.yaml --run-dir runs/kinetics --report report.html` executes the steps, prints one line per step with its state and duration, prints each note on a line that starts with `note:`, and, with `--report`, writes the report to the given path. A note is a text that a capability records about its result; Python stores notes in the `notes` of a step.

The run directory is the directory given by `--run-dir`. It holds the artifact store (`artifacts/`), a working directory per step, the step cache and the run record (`run.json`, see [Run records](#run-records)). Running a campaign again with the same `--run-dir` uses the step cache of that directory (see [Reuse](#reuse)), replaces `run.json` and adds a copy to `history/`. Without `--run-dir`, a new directory is created under the system temporary directory and printed on the line that starts with `artifacts:`.

A failed step prevents only the steps that depend on it from starting. Independent steps still run. A step bound to a failed step, or to a step that was not run, is not started: its state is `skipped`, and its note reads `STEP was not run because UPSTREAM failed` or `STEP was not run because UPSTREAM was not run`. The run is marked incomplete in the report and in `alambiq history`, and `alambiq run` exits with status 1. The error of a failed step is printed once, not repeated for the steps that depend on it.

## Run records

The run record is `run.json` in the run directory. It describes the latest run; a copy of every run, including the latest, is stored under `history/`. The run record contains the campaign as submitted, the start time, the duration, whether all steps completed or were reused, the versions of Alambiq and Python, the platform and, for each step, the capability and its version, the request after the bindings were resolved, the reuse key (see [Reuse](#reuse)), the duration and the outcome. The version of a capability is the number printed by `alambiq capability show`.

`alambiq history` lists the runs of a run directory. `alambiq replay RUN_DIR STEP --out DIR` writes the resolved request of one step as a one-step campaign, with the files that the step reads copied beside it. It fails for a step whose request was not built.

## Validation

Alambiq checks a request against the declared request type of the capability. A step fails with a message that names the step and the field when its request contains a field the capability does not accept, lacks a required field or contains a value of the wrong type. For an unknown field the message lists the fields the capability accepts. For a missing field it states that the field is required. For a value of the wrong type it states the expected type.

`alambiq validate campaign.yaml` checks a campaign without running it:

- every capability exists;
- every binding names a step in the campaign;
- the steps have no circular dependency;
- the form of each request, as far as it is known before the run: an unknown field, a missing required field, a value of the wrong type or outside the allowed choices.

The value that a binding supplies is checked when the step runs. Input files and data are read when the step runs, so a missing file or a unit that disagrees with the table is reported then.

Validation checks the form of a request. A fit with a model that does not describe the data is computed; the result reports the standard errors, and the stored record holds the residuals. A well-formed value that is wrong, such as a failed reading entered as 0 and marked measured, is used as given. `assay.inspect` records a note when readings of exactly 0 are marked measured.

## Reuse

Reuse means that a step whose key is unchanged is not computed again; its stored result is used. The key of a step is built from the capability and its version, the request after the bindings were resolved, and the name and SHA-256 digest of every input file the request names. A reused step is marked `reused`; later steps receive its stored result. `--fresh` recomputes every step and ignores the step cache.

In the same run directory:

- Adding a step computes only the new step.
- Changing the `model` of the step `fit` of the example reuses the step `intake`.
- Editing the input table recomputes every step downstream of it, because the digest of the file changed.

The key uses the name and the SHA-256 digest of each input file, not the directory that holds it. The same file in another directory is the same input; a copy with another name is a different input. A run directory can be moved or copied to another machine and its stored results remain usable, because the run record and the stored results refer to the artifact store by a placeholder, not by an absolute path.

The key of a step that runs an engine, or reads a pinned model or database, also includes the Python and package versions of the engine environment, the versions of the executables declared for the engine environment, and the version and file digests of each pinned resource. Upgrading MMseqs2 or changing the model weights therefore invalidates the earlier results, which were computed with a different MMseqs2 version or different model weights. Such a step is reused only where its engine environment is configured, because the identity of the engine environment cannot be determined otherwise.

The key depends only on the declared request. A capability must read only inputs named in its request, because a change to any other input does not change the key and the stored result would be reused.

## Engine environments

An engine is the module that runs inside an engine environment. An engine environment is a separate Python environment with pinned packages, named by the variable `ALAMBIQ_ENV_<NAME>`, for example `ALAMBIQ_ENV_OPENMM`. Capabilities that need OpenMM, MMseqs2, PySCF or a protein language model run their engines in such environments. The request is passed to the engine as a JSON file, and the engine writes its result as a JSON file.

`alambiq env check` reports which engine environments are present and what they contain, and, for each environment that is missing, names its recipe: the path of the document that describes how to build the environment. A step whose capability needs a missing engine environment fails before the capability starts, with a message that names the variable and, when the variable is unset, the recipe.

## Resources

A resource is a pinned model or database, verified before use. A capability that reads a resource names it. The name identifies a resource specification, a JSON file in `src/alambiq/resources/` that states the version, the license, and the size and SHA-256 digest of every file. A result depends on the resource specification: a changed version or file is a different resource, and earlier results are not reused.

Resources are installed before a run.

- `alambiq resource list` shows the known resources and which are installed.
- `alambiq resource show NAME` prints the license, size and location of a resource.
- `alambiq resource verify NAME` checks every installed file against the specification.
- `alambiq resource provision NAME --yes` downloads a resource whose specification lists a source for each file. All files are checked before any of them is installed.

A database that a specification describes but that is built locally, such as the UniRef50 and Pfam search databases, is built with the steps stated in the `build` field of its specification and is then verified in the same way.

Resources are stored under `ALAMBIQ_RESOURCE_HOME` when it is set, otherwise under `~/.alambiq/resources`, in directories named by the SHA-256 digest of the list of files in the specification. `ALAMBIQ_RESOURCE_<NAME>` (the name in capitals, punctuation as underscores) sets the location of one resource. The first verification hashes every file. A receipt of each file's size and modification time, stored beside the resources or in the directory named by `ALAMBIQ_RESOURCE_RECEIPTS`, lets later verifications skip the hash until a file changes.

## Reports

A report is a single HTML file that summarizes a run. It is written when `alambiq run` receives `--report PATH` or when `RunResult.report(path)` is called. The contents of a report are specified in [Reports](reports.md).
