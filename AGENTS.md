# Alambiq agent instructions

Alambiq is open-source software for biotechnology R&D, organized as capabilities.

## Sources of truth

- `docs/concepts.md` defines the terms capability, campaign, step, binding, request,
  result, run, run record, reuse, engine environment and resource; `docs/methods/` explains how to
  choose and read the capabilities of each method area.
- A capability is declared in code: `alambiq capability show ID` prints its request
  fields, result, method and limits from that declaration. Do not restate the request fields, result or limits
  of a capability in another file, such as a manifest, a schema or an agent skill file.
  Methods pages explain the choice and interpretation of capabilities and link to
  `alambiq capability show`.
- Document implemented behavior and measured limits; do not add development forecasts
  or release promises.

## Required discipline

- Organize reusable code by scientific capability, never by application sector or
  release number. Implement scientific computation in functions that do not depend on
  the capability registry, the run context or Prefect, so that tests call them
  directly. The capability function reads the request, calls such a function and builds
  the result.
- A campaign written in YAML and one written in Python are built into the same
  `Campaign` model and run by the same code.
- The researcher plans the research. Alambiq runs the requests that the researcher
  writes and does not generate requests; do not add an autonomous scientific planner.
- Prefect is the sole workflow runtime, and no parallel executor is added. Code for
  requests, resource identities and result presentation does not import Prefect;
  `src/alambiq/core/execution.py` is the module that does.
- Engines run in their own pinned environments. An engine receives a JSON request file
  and writes a JSON result file; engine modules import only the standard library and
  their own engine.
- Preserve domain-specific types; do not introduce one type for every biological
  entity.
- Add a dependency or an abstraction only for a need stated in the change description.
  Remove superseded code. Shared viewers and exports are part of the report file and
  must not require an externally hosted service.
- A Alambiq installation runs on one machine. Do not add scheduler management or
  distributed execution unless a need for it is stated.
- Report results and source identities. A diagnostic names what is wrong (the field,
  column, unit, file, step or variable). The researcher interprets results and makes
  research decisions. A scientifically implausible value is a finding in the result,
  never a validation error.
- Keep resource files, engine environments and generated artifacts outside Git.
- Select scientific software and model weights whose licenses permit commercial use
  without fees, individually negotiated agreements, or academic-only or
  noncommercial-only restrictions. Prefer MIT, BSD and Apache-2.0; the pattern `ALLOWED` in
  `tools/scan_licenses.py` defines the other accepted licenses. Check the license of the
  code, the model weights and the data separately, at the version in use: the license of a
  Python package that calls a model is separate from the license of the model. Avoid GPL scientific
  executables in the recipes under `environments/`. Preserve third-party
  notices.
- Obtain explicit approval from the repository owner before downloading a resource,
  calling a paid service, running an expensive computation, contacting an external
  party, publishing or pushing.
- When a test fails, find the cause and fix it. Do not weaken the test to make it pass.
- Test each change in the behavior of a method against a known answer, and test the
  failure cases that apply. Preserve attribution, the identity of inputs and outputs
  (content digests), units, source type and applicable licensing information.
- In documentation, help text, reports and comments, state implemented behavior, input
  requirements, assumptions and observations directly, in affirmative sentences and not
  as negations of what is absent.

## Writing

Repository text is reference documentation, written like a technical manual or the methods
section of a paper. `tests/core/test_writing.py` enforces the mechanical rules.

- **Voice.** Impersonal and in the present tense. Procedures use the imperative
  (`Install the core.`). No second or first person, no contractions, no rhetorical
  questions, no exclamation marks.
- **Headings.** Sentence-case noun phrases naming the subject: "Run records", not "What a
  run keeps". Bare imperatives only in step-by-step guides and recipes ("Install", "Build").
  A heading never asks a question or promises a result.
- **Definitions.** A term is defined once, by what it is and what distinguishes it, and used
  the same way afterwards. Example: "A capability is a function with a declared request
  type, a declared result type and documented limits. It is identified by `prefix.action`."
- **Vocabulary.** "Alambiq" names the software. The terms are capability, campaign, step,
  binding, request, result, run, run record, reuse, engine environment and resource. Do not use "workbench",
  "platform", "tool" or "workflow" for the software or its parts (OpenMM's platforms and
  Prefect's workflows keep their own names).
- **Stable text.** Prose does not state counts of the repository's own contents (capabilities,
  examples, pages, tests, extras) or other details that a listing command or a test already
  gives; the command is the source. Numbers that belong to the method (limits, defaults,
  ranges, results) and dated evidence (a verification receipt) stay.
- **Precision.** Quantify or state the condition instead of "a few", "usually", "roughly",
  "often". State a limit as a condition: "The interval assumes independent residuals."
  Every number has its unit. Counts are digits ("3 replicas").
- **Scope.** Text states what the software does and the conditions under which a result
  holds, as positive conditions: "The interval assumes independent residuals", "Applies to
  single substitutions of a natural enzyme", "Requires a structure without ligands". It does
  not list what the software does not do, claim, establish or replace ("does not establish",
  "makes no claim", "is not a substitute for", "not cleared"). A negative sentence stays only
  where a reader would otherwise use a result where it does not hold and no positive
  condition says so; it is one sentence, at the place where the result is used. License text
  states the terms that apply.
- **Plain prose.** No evaluative adjectives or adverbs ("simply", "carefully", "robust"),
  no metaphors ("lands", "keeps", "carries"), no sentence about the page itself, no
  asides set off with dashes. Qualities are shown by tests, never claimed in text.
- **Spelling.** American ("analyze", "license", "modeled", "behavior", "center").
- **Release labels.** Text describes what exists. It contains no release names, forecasts
  or promises.

## Validation

Run these commands before proposing a change, in a Python environment where the project
is installed (or with `PYTHONPATH=src` set). The last command writes its run to a new
temporary directory.

```bash
python -m pytest tests/core
ruff check src tests tools
python -m alambiq validate examples/kinetics-from-measurements/campaign.yaml
python -m alambiq run examples/kinetics-from-measurements/campaign.yaml
```

`tests/core` is the whole suite. A test that needs an engine environment declares it and
is skipped when `ALAMBIQ_ENV_<NAME>` (for example `ALAMBIQ_ENV_OPENMM`) does not point
at that environment's Python; the skip message names the missing prerequisite.

Besides the method tests, `tests/core` contains these checks of the whole repository:

- `test_examples.py`: every example runs from its `campaign.yaml`, and the test asserts
  the values the example was built to return.
- `test_report_lint.py`: every report parses, loads nothing from the network, shows no
  Python or NumPy representation, `NaN`, `inf` or `None` in its text, and in each numeric
  column the header states the unit, or states that the column is a count, an index, a
  rank or a fraction. The outline of each report (sections, headings, notes, table
  headers and collapsed sections) matches the snapshot in `tests/snapshots/reports/`;
  `ALAMBIQ_UPDATE_SNAPSHOTS=1` rewrites the snapshots.
- `test_first_mistakes.py`: each input mistake that a first-time user makes (listed in
  the test) produces a message that names the field, column, unit, file, step or variable
  at fault, a nonzero exit status and no traceback.
- `test_determinism.py`: the steps that draw random numbers or start from random points
  (candidate selection, codon recoding, fits to progress curves) return identical outputs
  when an example runs twice in new run directories, and a different seed changes a
  random selection.
- `test_runtime_budgets.py`: no step of an engine-free example exceeds its time budget,
  which is about 10 times the time measured on a laptop, with a minimum of 5 s.
- `test_documentation.py`: every relative link resolves, every capability appears on a
  methods page that the methods index lists, every path and `alambiq` subcommand named
  in a shell block exists, and every example has a README with the same sections and
  appears in the index of examples.
- `test_writing.py`: the mechanical rules above.
- `test_licenses.py`: every resource specification states a license from the accepted set
  and the size and SHA-256 digest of each file, the third-party notices name every
  resource and the bundled viewer, and no installed dependency has a license that
  `tools/scan_licenses.py` classifies as refused or unknown.
- `test_core_imports.py`: all capabilities register and are listed when the optional
  packages (NumPy, SciPy, RDKit and others) cannot be imported.
- `test_conformance.py`: every capability states its limits, describes each request field
  and its result, has a test that exercises each request field, except the justified
  exemptions listed in the test, and has a function that renders its result in the report.

When a change breaks one of these checks, find the cause; do not relax the check. Check a
known answer in every method test: an analytic result, a quantity recomputed
independently from the written files, or a transformation whose answer is fixed (for
example, the difference between a structure and itself is zero).

Checks outside the suite:

- `python tools/check_distribution.py` builds the wheel and the source distribution,
  inspects their contents and installs the wheel into a new environment.
- `python tools/scan_licenses.py` lists each installed dependency whose license is not
  classed as allowed; `--all` lists every dependency.
- `python tools/check_getting_started.py` runs the commands of `docs/getting-started.md`
  in a copy of the repository made for the check.
- `python -I tools/check_installed_package.py --wheel PATH --receipt FILE` runs the tests
  against an installed wheel and records the module origins and the skipped tests.

Build local wheels from a fresh checkout or clear `build/` first: a wheel built from a
`build/` directory that still holds a module removed from the source tree contains that
module.
