# Operation file formats

The operation capabilities read JSON files: an operating definition, a template, a
fit specification, an observation file, an assessment and a comparison file. This page
lists every field of each file with its type, unit, whether it is required, the
values it accepts and the validation rules of the capability that reads it. The model
that the fields describe is in [Operation](methods/operation.md).

The files and the capabilities that read them:

- An **operating definition** is read by `operation.simulate`. Each entry of
  `experiments` in a fit specification is an operating definition, and `operation.fit`
  writes its fitted definitions in the same layout.
- A **template** is read by `operation.define` and `operation.propagate`. It is an
  operating definition in which the rate parameters of the named channels are
  placeholders.
- A **fit specification** is read by `operation.fit`.
- An **observation file** is read by `operation.compare` and, without
  `operation_id`, by `operation.compare_ensemble`.
- An **assessment** is read by `operation.summarize`.
- A **comparison file** is read by `operation.compare_resources`.

`alambiq capability show ID` prints the other fields of each capability's request,
such as `solver`. `operation.simulate` and `operation.fit` take `solver` (`Radau`, the
default, or `DOP853`) in the `params` of the step, not as a field of the file.

## Conventions

- Each file is one JSON object. A step passes the file path, relative to the
  campaign file or absolute, in the `params` entry that the capability declares:
  `definition` for `operation.simulate` and `operation.fit`, `template` for
  `operation.define` and `operation.propagate`, `comparison` for
  `operation.compare` and `operation.compare_ensemble` (an observation file) and for
  `operation.compare_resources` (a comparison file), and `assessment` for
  `operation.summarize`.
- A JSON object accepts exactly the fields listed for it. A missing required field
  or an unknown field is refused, and the message lists the accepted fields. A
  misspelled optional field is therefore refused.
- A number is a finite JSON number. A string, a boolean or `null` is refused where a
  number is expected. "Nonnegative" means zero or more, "positive" means above zero,
  and a "fraction" lies between zero and one inclusive.
- An `id` is a non-blank single-line string of at most 200 characters. Ids are
  unique within their list, and other fields refer to them.
- Free text (`conditions`, `source`, `basis`, `scope`, and the `product` text of a
  comparison file) is a non-blank single-line string of at most 3000 characters. The
  `source` fields are required.
- The field name fixes the unit: concentrations in M (mol/L), time in s, volume in
  L, amounts in mol, rates in M/s. Where a field accepts a unit, the sections below
  name the accepted units. The amount of a catalyst is stated in `amount_unit`, a
  string of at most 100 characters, and every other amount of that catalyst uses the
  same unit. `operation.summarize` converts the catalyst amount to moles of active sites
  (see Assessment). `operation.define` and `operation.propagate` convert the unit of the
  assay enzyme concentration to the `amount_unit` of the catalyst when both are mass
  units.
- The capability that reads a file checks it when its step runs.
  The check covers structure, units, references between fields and the rules below.
- `alambiq validate` checks the structure of a campaign. An error in one of these files
  is reported when the step that reads it runs.
- Two inputs are produced by earlier steps and are not written by hand: the
  `operation` request field of `operation.compare` and `operation.summarize` is the
  `record` of an `operation.simulate` step (the stored JSON file of its result), and
  the fitted definitions are the `operations` result of an `operation.fit` step.

## Operating definition

An operating definition describes one experiment: a well-mixed liquid containing
named species, catalysts (possibly none), rate channels, a sequence of phases, discrete
exchanges (events, possibly none) and the times at which the state is recorded.

A field not marked optional is required.

- `id`: string, the name of the experiment. The `operation_id` of every observation
  file and assessment of this experiment equals this `id`.
- `volume_L`: positive number, the initial liquid volume in L.
- `conditions`: free text describing the physical and assay conditions. An
  observation file or assessment of this experiment repeats it unchanged.
- `species`: list of 1 to 200 objects.
- `catalysts`: list of 0 to 50 objects; `[]` when no channel names a catalyst.
- `channels`: list of 1 to 200 objects.
- `phases`: list of 1 to 100 consecutive objects.
- `events`: list of 0 to 200 objects; `[]` for uninterrupted operation.
- `times_s`: list of 2 to 10000 numbers, the times in s at which the state is
  recorded.
- `collections` (optional): list of 0 to 200 objects.
- `state` (optional): the operating conditions, an object with the temperature and pH
  of the operation.

### Species

Each object has `id`, `initial_M` and `pools`.

- `initial_M`: nonnegative number, the concentration at time zero in M.
- `pools`: object that maps a pool name to a nonnegative number; at most 100 pools
  per species, and `{}` for none. A pool is a named quantity that every channel must
  leave unchanged, for example a cofactor or a chemical moiety. The number is the
  amount of that quantity in one molecule of the species, in a unit chosen by the
  author and the same for all species of the pool. Every channel conserves every
  pool: the sum over species of stoichiometric coefficient times pool number equals
  exactly zero. The check uses exact rational arithmetic without a tolerance. A channel that changes a
  pool is refused, and the message names the pool.

### Catalysts

Each object has `id`, `amount`, `amount_unit`, `reference_amount`,
`active_fraction`, `deactivation_per_s` and `washout_fraction`, and optionally
`deactivation_temperature`.

- `amount_unit`: string of at most 100 characters, such as `mg` or `U`. It is the
  unit of `amount`, `reference_amount`, the `amount_per_L` of the catalyst feed, the
  `amount` of the catalyst additions of an event and any observation or threshold
  about this catalyst.
- `amount`: nonnegative number in `amount_unit`, the total catalyst at time zero.
- `reference_amount`: positive number in `amount_unit`, the amount at which the
  channels that name this catalyst state their capacity.
- `active_fraction`: fraction, the share of `amount` that is active at time zero.
- `deactivation_per_s`: nonnegative number in 1/s, the first-order rate constant of
  the loss of activity.
- `washout_fraction`: fraction, the ratio of the catalyst concentration in the
  outflow to the catalyst concentration in the reactor. 0 retains the catalyst
  completely; 1 removes it at the rate of the liquid outflow.
- `deactivation_temperature`: object `{activation_energy_J_mol, reference_K}`, with a
  finite activation energy in J/mol and a positive reference temperature in K. It
  makes `deactivation_per_s` depend on the temperature in `state`; `reference_K` is
  the temperature at which `deactivation_per_s` was measured.

### Channels

A channel is the conversion of species in one direction by one rate law. A
reversible reaction is two channels, one per direction; `reaction_id` names the
reaction. The fields:

- `id`, `reaction_id`: strings naming the channel and its reaction.
- `stoichiometry`: object that maps species ids to signed coefficients, negative for
  consumed species and positive for produced species. It contains at least two
  species, all declared, with at least one consumed and one produced. A coefficient
  is an integer or a string of at most 32 characters in integer, fraction or decimal
  notation (`"1/2"`, `"-3/2"`, `"0.5"`). Its magnitude is above 0 and at most 10000,
  and the denominator of its fraction in lowest terms is at most 1000000. A float is
  refused.
- `capacity_M_s`: nonnegative number in M/s. With a `catalyst` it is the rate at that
  catalyst's `reference_amount`, full activity and the initial volume; without a
  `catalyst` it is a volumetric rate at the current volume.
- `factors`: list of objects that define the concentration dependence, described
  below.
- `source`: free text giving the origin of the rate.
- `catalyst` (optional): the id of a declared catalyst. The rate of the channel is
  multiplied by the ratio of the active catalyst amount to `reference_amount` and by
  the ratio of the initial volume to the current volume.
- `inhibitors_M` (optional): object that maps species ids to positive inhibition
  constants in M. The whole rate is divided by 1 plus the sum over the inhibitors of
  concentration divided by constant.
- `denominator` (optional): a coupled denominator `{constant, terms}` by which the rate
  is divided, described below (see also
  [Operation](methods/operation.md#rate-factors)).
- `temperature` (optional): object `{activation_energy_J_mol, reference_K}`, as for
  the `deactivation_temperature` of a catalyst. It multiplies `capacity_M_s` by
  exp(-Ea/R (1/T - 1/`reference_K`)), with Ea the activation energy, R the gas
  constant and T the `temperature_K` of `state`.
- `ph` (optional): object `{pKa1, pKa2, reference_pH}` of three finite numbers. It
  multiplies `capacity_M_s` by a(pH) / a(`reference_pH`), with
  a(x) = 1 / (1 + 10^(pKa1 - x) + 10^(x - pKa2)) and pH taken from `state`. `pKa1` and `pKa2` are used in
  the order given.

Each object of `factors` has `species`, `kind` and `scale_M`.

- The list holds at least one factor and at most one factor per species. The species
  of the factors include every species with a negative coefficient. The rate of
  the channel is therefore zero when any consumed species has zero concentration.
- `kind`: `mass-action` or `saturation`. `scale_M`: positive number in M.
- `power` (optional): integer 1 or 2, default 1, the power of the numerator.
- For a `saturation` factor only: `denominator_power` (1 or 2, default 1),
  `competitive_inhibitors_M` and `uncompetitive_inhibitors_M` (objects that map
  species ids to positive constants in M) and `self_inhibition_M` (positive number in
  M). A `mass-action` factor that contains any of these four fields is refused.

A `denominator` is `{constant, terms}`.

- `constant`: positive number. `terms`: list of 1 to 100 objects, each with an `id`
  (unique among the terms), a positive `coefficient` and `factors`, a list of
  `{species, scale_M, power}` with a positive `scale_M` and an optional `power` of 1
  or 2. A species occurs at most once in a term.
- All `factors` of a channel with a `denominator` are `mass-action`.

The rate laws and the role of each inhibition constant are in
[Operation](methods/operation.md#rate-factors).

### State

`state` is `{temperature_K, pH}`. Both fields are optional: `temperature_K` is a
positive number in K, and `pH` is any finite number. A channel with `temperature`, or
a catalyst with `deactivation_temperature`, requires `temperature_K`; a channel with
`ph` requires `pH`. A definition that declares such a dependence without the
corresponding `state` field is refused. A `state` without any dependence is recorded
and changes no rate.

### Phases

Phases run one after another from time zero. Each phase has `end_s`, `feed_M`,
`active_channels` and the flow, and optionally `solute_input_mol_s` and
`catalyst_feed`.

- `end_s`: positive number in s, greater than the `end_s` of the previous phase. The
  last phase ends at the last entry of `times_s`.
- `feed_M`: object that maps species ids to nonnegative concentrations in M of the
  combined inlet. A species that is absent has zero inlet concentration. `{}` when
  nothing is fed.
- `active_channels`: list of unique declared channel ids. An empty list is a phase
  without reaction.
- The flow is either `dilution_per_s` (nonnegative number in 1/s) or both `inlet_L_s`
  and `outlet_L_s` (nonnegative numbers in L/s). Both forms together, or only one
  field of the pair, are refused. `dilution_per_s` sets the inlet flow and the outlet
  flow to that rate times the volume at the start of the phase, so the volume is
  constant. Separate flows are constant during the phase and change the volume. A phase whose final
  volume is not positive is refused. Different phases may use different forms.
- `solute_input_mol_s`: object that maps species ids to nonnegative amount rates in
  mol/s, added independently of the liquid flow. A physical addition is stated once,
  as feed or as solute input; stating it as both counts its amount twice.
- `catalyst_feed`: object that maps declared catalyst ids to
  `{amount_per_L, active_fraction}`, both required: `amount_per_L` is the nonnegative
  concentration of that catalyst in the combined inlet, in its `amount_unit` per L,
  and `active_fraction` is the fraction of it that is active. A catalyst that is
  absent is not fed.

### Events

An event replaces part of the liquid at one instant. All fields are required except
the last.

- `time_s`: positive number in s, greater than the `time_s` of the previous event and
  at most the last entry of `times_s`. It need not be an entry of `times_s`.
- `exchange_fraction`: fraction of the current liquid that is removed and replaced by
  fresh liquid of the same volume.
- `feed_M`: object that maps species ids to the nonnegative concentrations of the
  fresh liquid, in M.
- `catalyst_retention`: object with one fraction for every declared catalyst and no
  other key, the share of that catalyst retained in the reactor. `{}` when there are
  no catalysts.
- `catalyst_additions`: object that maps declared catalyst ids to
  `{amount, active_fraction}`, the fresh catalyst added after the exchange, with
  `amount` in the amount unit of the catalyst and `active_fraction` a fraction. `{}`
  for none.
- `catalyst_activity_retention` (optional): object with one fraction for every
  declared catalyst, the share of the retained active catalyst that remains active
  after handling. The default is 1 for each catalyst.

### Collections and times

- Each entry of `collections` has `id`, `start_s` and `end_s` (numbers in s), with
  `0 <= start_s < end_s` and `end_s` at most the last entry of `times_s`. The window
  collects a positive outlet volume. Windows may overlap, and their boundaries need
  not be entries of `times_s`. An `effluent-concentration` observation refers to a
  window by `id` in `collection_id`.
- `times_s` starts at 0 and increases strictly. The state is recorded at each time;
  when an event falls on one of these times, the recorded state is the one after
  the exchange. The states immediately before and after every event are recorded as well.

### Operating definition examples

`batch.json`: a first-order conversion of S to P in a batch with one catalyst. The
rate constant is 0.002 M/s x 1 / 0.01 M = 0.2/s. The examples of the other sections
refer to this file. `operation.simulate` returns 0.0013534 M of S and 0.0086466 M of P
at 10 s, the values of 0.01 exp(-0.2 t) M and 0.01 (1 - exp(-0.2 t)) M.

```json
{
  "id": "first-order-reference",
  "volume_L": 0.01,
  "conditions": "Declared constant assay conditions",
  "species": [
    {"id": "S", "initial_M": 0.01, "pools": {"substrate-moiety": 1}},
    {"id": "P", "initial_M": 0, "pools": {"substrate-moiety": 1}}
  ],
  "catalysts": [
    {"id": "E", "amount": 2, "amount_unit": "mg", "reference_amount": 2,
     "active_fraction": 1, "deactivation_per_s": 0, "washout_fraction": 0}
  ],
  "channels": [
    {"id": "forward", "reaction_id": "S-to-P", "stoichiometry": {"S": -1, "P": 1},
     "capacity_M_s": 0.002,
     "factors": [{"species": "S", "kind": "mass-action", "scale_M": 0.01}],
     "source": "Analytic first-order test, k = 0.2/s", "catalyst": "E"}
  ],
  "phases": [{"end_s": 10.0, "dilution_per_s": 0.0, "feed_M": {}, "active_channels": ["forward"]}],
  "events": [],
  "times_s": [0, 1, 2, 3, 5, 7, 10]
}
```

`fed-batch.json`: a saturation channel with a competitive and a self-inhibition
constant, a catalyst with deactivation, washout and a temperature dependence, a
stated temperature and pH, a first phase that doses substrate at a constant rate in
a closed volume, a second phase with separate inlet and outlet flows that also feeds
catalyst, one exchange with partial catalyst retention and a collection window. The
rate values are stated for the example. `operation.simulate`
returns 0.008680 M of S and 0.001986 M of P at 1200 s.

```json
{
  "id": "fed-batch",
  "volume_L": 0.01,
  "conditions": "A 10 mL reactor, dosed in the first phase and flowing in the second",
  "state": {"temperature_K": 310.15, "pH": 7.5},
  "species": [
    {"id": "S", "initial_M": 0.01, "pools": {"moiety": 1}},
    {"id": "P", "initial_M": 0, "pools": {"moiety": 1}}
  ],
  "catalysts": [
    {
      "id": "E",
      "amount": 10,
      "amount_unit": "mg",
      "reference_amount": 10,
      "active_fraction": 0.9,
      "deactivation_per_s": 1e-05,
      "washout_fraction": 0.5,
      "deactivation_temperature": {"activation_energy_J_mol": 80000, "reference_K": 298.15}
    }
  ],
  "channels": [
    {
      "id": "convert",
      "reaction_id": "S-to-P",
      "stoichiometry": {"S": -1, "P": 1},
      "capacity_M_s": 2e-06,
      "catalyst": "E",
      "factors": [
        {
          "species": "S",
          "kind": "saturation",
          "scale_M": 0.002,
          "competitive_inhibitors_M": {"P": 0.05},
          "self_inhibition_M": 0.5
        }
      ],
      "temperature": {"activation_energy_J_mol": 50000, "reference_K": 298.15},
      "ph": {"pKa1": 6, "pKa2": 9, "reference_pH": 7.5},
      "source": "Stated for this example"
    }
  ],
  "phases": [
    {
      "end_s": 600,
      "dilution_per_s": 0,
      "feed_M": {},
      "solute_input_mol_s": {"S": 1e-08},
      "active_channels": ["convert"]
    },
    {"end_s": 1200, "inlet_L_s": 1e-06, "outlet_L_s": 5e-07, "feed_M": {"S": 0.02},
     "catalyst_feed": {"E": {"amount_per_L": 10, "active_fraction": 0.8}},
     "active_channels": ["convert"]}
  ],
  "events": [
    {"time_s": 1000, "exchange_fraction": 0.5, "feed_M": {"S": 0.01},
     "catalyst_retention": {"E": 0.9}, "catalyst_activity_retention": {"E": 0.95},
     "catalyst_additions": {"E": {"amount": 0.5, "active_fraction": 1}}}
  ],
  "collections": [{"id": "effluent", "start_s": 700, "end_s": 1100}],
  "times_s": [0, 300, 600, 900, 1000, 1200]
}
```

## Template

A template is an operating definition, as described in
[Operating definition](#operating-definition), whose named channels hold placeholders. `operation.define` replaces the `capacity_M_s` and the
`scale_M` of the saturation factor of each channel named in the `kinetics` field of
its request with values computed from an assay fit, and rewrites the `source` of that
channel. It can also set the `deactivation_per_s` of a catalyst, and it adds the
`temperature`, `ph` and `deactivation_temperature` blocks when the request includes
temperature, pH or inactivation fits. All other fields of the template keep their
values, including the channels that `kinetics` does not name. `operation.propagate`
applies the same replacement to one channel for every coefficient set that it draws.

A template follows the rules of an operating definition, with these additions. A
placeholder capacity is a valid number, for example 0.

- A channel named in `kinetics` has a declared `catalyst` and consumes the substrate
  that the entry gives as `species`. Unless it is the forward channel of a reversible
  pair (below), it has a `saturation` factor on that species with `power` 1 and no
  `denominator`. The capacity refers to the `reference_amount` and `amount_unit` of
  the catalyst and to the `volume_L` of the template.
- A channel appears at most once in `kinetics`.
- For a reversible pair given in `equilibria`, both channels name the same
  catalyst. The forward channel consumes the substrate and produces the product, and
  the reverse channel the opposite. Each has one `mass-action` factor on its consumed
  species with `power` 1, and a `denominator` with `constant` 1 and 2 terms whose
  factors are on the substrate and the product only (in the layout of
  `reversible-template.json`, one term on the substrate and one on the product, each
  with 1 factor). The forward channel also needs its own entry in `kinetics`. The scales and
  the reverse capacity are set from the forward fit, the product's Michaelis
  constant and the equilibrium constant.

`template.json`: the template of `examples/connected-investigation`. Its channel
`hydrolysis` has a placeholder capacity of 0 and a placeholder scale of 1:

```json
{
  "id": "hydrolysis",
  "volume_L": 0.5,
  "conditions": "Assay conditions of the kinetics measurement",
  "species": [
    {"id": "S", "initial_M": 0.01, "pools": {"moiety": 1}},
    {"id": "P", "initial_M": 0, "pools": {"moiety": 1}}
  ],
  "catalysts": [
    {"id": "E", "amount": 10, "amount_unit": "mg", "active_fraction": 1,
     "reference_amount": 10, "deactivation_per_s": 0, "washout_fraction": 0}
  ],
  "channels": [
    {"id": "hydrolysis", "reaction_id": "S-to-P", "stoichiometry": {"S": -1, "P": 1},
     "capacity_M_s": 0.0, "catalyst": "E",
     "factors": [{"species": "S", "kind": "saturation", "scale_M": 1.0}],
     "source": "Set by operation.define from the measured kinetics"}
  ],
  "phases": [
    {
      "end_s": 3600.0,
      "dilution_per_s": 0.0,
      "feed_M": {},
      "active_channels": ["hydrolysis"]
    }
  ],
  "events": [],
  "times_s": [0.0, 300.0, 600.0, 1200.0, 1800.0, 2700.0, 3600.0]
}
```

`reversible-template.json`: a template for a reversible pair. It also shows the
layout of a coupled `denominator` in an operating definition. The `equilibria` entry
that sets it is `{"forward": "forward", "reverse": "reverse", "substrate": "S",
"product": "P", "product_km": {"value": 2, "unit": "mM"}, "equilibrium_constant": 4}`,
together with a `kinetics` entry for `forward`.

```json
{
  "id": "reversible",
  "volume_L": 0.5,
  "conditions": "Assay conditions of the kinetics measurement",
  "species": [
    {"id": "S", "initial_M": 0.01, "pools": {"moiety": 1}},
    {"id": "P", "initial_M": 0, "pools": {"moiety": 1}}
  ],
  "catalysts": [
    {"id": "E", "amount": 10, "amount_unit": "mg", "active_fraction": 1,
     "reference_amount": 10, "deactivation_per_s": 0, "washout_fraction": 0}
  ],
  "channels": [
    {
      "id": "forward",
      "reaction_id": "S-to-P",
      "stoichiometry": {"S": -1, "P": 1},
      "capacity_M_s": 1e-05,
      "catalyst": "E",
      "factors": [{"species": "S", "kind": "mass-action", "scale_M": 0.001}],
      "denominator": {
        "constant": 1,
        "terms": [
          {
            "id": "substrate",
            "coefficient": 1,
            "factors": [{"species": "S", "scale_M": 0.001}]
          },
          {
            "id": "product",
            "coefficient": 1,
            "factors": [{"species": "P", "scale_M": 0.001}]
          }
        ]
      },
      "source": "Placeholder; set by operation.define"
    },
    {
      "id": "reverse",
      "reaction_id": "P-to-S",
      "stoichiometry": {"P": -1, "S": 1},
      "capacity_M_s": 1e-05,
      "catalyst": "E",
      "factors": [{"species": "P", "kind": "mass-action", "scale_M": 0.001}],
      "denominator": {
        "constant": 1,
        "terms": [
          {
            "id": "substrate",
            "coefficient": 1,
            "factors": [{"species": "S", "scale_M": 0.001}]
          },
          {
            "id": "product",
            "coefficient": 1,
            "factors": [{"species": "P", "scale_M": 0.001}]
          }
        ]
      },
      "source": "Placeholder; set by the equilibrium constant"
    }
  ],
  "phases": [
    {
      "end_s": 3600,
      "dilution_per_s": 0,
      "feed_M": {},
      "active_channels": ["forward", "reverse"]
    }
  ],
  "events": [],
  "times_s": [0, 600, 1800, 3600]
}
```

## Fit specification

A fit specification contains the experiments to simulate, the fitted parameters and the
fields each parameter sets, the starting values, the measurements and the uncertainty
mode of `operation.fit`. All fields are required.

- `id`: string, the name of the fit specification.
- `experiments`: list of 1 to 20 operating definitions with distinct ids, each a
  complete operating definition as described above.
- `parameters`: list of 1 to 12 objects, described below.
- `starts`: list of 1 to 10 objects.
- `observations`: list of 1 to 10000 objects.
- `uncertainty`: `descriptive` or `independent-residuals`. With `independent-residuals`,
  the result contains local 95 % intervals when the Jacobian rank equals the number of
  parameters and no parameter is at a bound; with `descriptive` it contains none (see
  [Operation](methods/operation.md#fitting-parameters-to-measured-progress)).
- `max_nfev`: integer from 1 to 2000, the maximum number of function evaluations of
  the optimizer for each start.

### Parameters

A parameter is a number adjusted by the fit. It sets every field listed in its
`targets`, in one or more experiments. The fields:

- `id`: string, the name used by `starts` and in the results.
- `unit`: string, the unit of the number. It equals the unit of every target.
- `lower`, `upper`: the bounds, in `unit`. `lower` is nonnegative, `upper` is
  positive, and `lower` is below `upper`. A parameter that sets a concentration
  scale, an inhibition constant or a denominator coefficient has `lower` above zero,
  and a parameter that sets a fraction has `upper` of at most 1. Negative values are
  not accepted, and activation energies and pKa values cannot be fitted.
- `scale`: positive number in `unit`. The optimizer works with the parameter divided
  by `scale`. The bounds divided by `scale` must be finite and must differ by enough
  that a step of 1e-4, or a quarter of the scaled interval if that is smaller, changes
  the scaled bound in floating-point arithmetic.
- `targets`: list of 1 to 200 objects `{experiment_id, path}`. A field is the target
  of exactly one parameter.
- `profile_values` (optional): list of at most 50 unique values inside the bounds. The
  fit is repeated at each value with the parameter held fixed, and the result reports
  the sum of squared standardized residuals (`sse`) at each value.

### Target paths

A `path` is a list of 3 to 8 strings and integers that addresses a number in the
experiment. Names are identical to the ids of the definition. An event is addressed by its
zero-based index in `events`, a factor by its species and a denominator term by its
`id`. The field exists in the definition as written: to fit `self_inhibition_M`, for
example, the factor already contains it. The targetable paths and their units:

- `["channels", channel, "capacity_M_s"]`: M/s.
- `["channels", channel, "factors", species, "scale_M"]`: M.
- `["channels", channel, "factors", species, "self_inhibition_M"]`: M.
- `["channels", channel, "factors", species, "competitive_inhibitors_M", inhibitor]`
  and the same with `"uncompetitive_inhibitors_M"`: M.
- `["channels", channel, "inhibitors_M", inhibitor]`: M.
- `["channels", channel, "denominator", "terms", term, "coefficient"]`: `1`, a
  dimensionless coefficient.
- `["channels", channel, "denominator", "terms", term, "factors", species, "scale_M"]`: M.
- `["catalysts", catalyst, "amount"]`: the catalyst's `amount_unit`.
- `["catalysts", catalyst, "deactivation_per_s"]`: 1/s.
- `["catalysts", catalyst, "active_fraction"]` and `"washout_fraction"`: `fraction`.
- `["events", index, "catalyst_retention", catalyst]` and
  `"catalyst_activity_retention"`: `fraction`.

All other fields keep the value given in the definition: the volume, flows,
feeds, catalyst feed, solute input, initial concentrations, powers, the constant of a
denominator, `state` and the temperature and pH dependences.

### Starts

Each start is an object with one number for every parameter id and no other key, each
number inside the bounds of its parameter. Every start is run. The reported fit is
that of the start with the lowest sum of squared standardized residuals among the
starts whose optimizer reports success. If no start succeeds, the step fails.

### Observations

Each observation has `id`, `experiment_id`, `time_s`, `state`, `quantity`, `target`,
`unit`, `value`, `scale`, `partition` and `source`, and optionally `reference_M` and
`collection_id`.

- `quantity` determines the kind of `target` and the `unit`:
  - `concentration`: a species, in `M`.
  - `conversion`: a species, in `fraction`, with a positive `reference_M`. The value is
    1 minus the concentration divided by `reference_M`.
  - `removed`: a species, in `mol`, the cumulative amount that has left through the
    outflow and the exchanges.
  - `effluent-concentration`: a species, in `M`, with a `collection_id`. The value is
    the flow-weighted mean concentration of that collection window.
  - `catalyst-total` and `catalyst-active`: a catalyst, in its `amount_unit`.
- The `state` field of an observation is `after` or `before` and selects the side of an
  exchange; it is unrelated to the `state` object of a definition. Together with `time_s`
  it selects one recorded state of the trajectory. With `state: "after"`, `time_s` is an
  entry of the experiment's `times_s` or the time of one of its events, and the
  state is the one after any exchange at that time. With `state: "before"`, `time_s`
  is the time of an event and the state is the one immediately before the exchange. An
  `effluent-concentration` observation uses `state: "after"` and a `time_s` equal to
  the `end_s` of its collection.
- `reference_M` is required for `conversion` and accepted for no other quantity;
  `collection_id` is required for `effluent-concentration` and accepted for no other
  quantity.
- `value`: any finite number in `unit`. Negative values are accepted, for example for
  readings corrected for a blank.
- `scale`: positive number in `unit`. The residual that the fit minimizes
  is the prediction minus the value, divided by `scale`.
- `partition`: `fit` or `evaluation`. Only `fit` observations enter the
  optimization. Their number must exceed the number of entries in `parameters`. Both
  groups are reported separately.
- `source`: free text giving the origin of the measurement.

### Fit specification examples

`fit.json`: a fit of the capacity of the channel in `batch.json` to 6 concentrations
of S, 4 for fitting and the last 2 held out for evaluation. `operation.fit` returns a
capacity of 0.002 M/s, the capacity of the experiment, with 4 fitted and 2 evaluated
observations:

```json
{
  "id": "analytic-progress",
  "experiments": [
    {
      "id": "first-order-reference",
      "volume_L": 0.01,
      "conditions": "Declared constant assay conditions",
      "species": [
        {"id": "S", "initial_M": 0.01, "pools": {"substrate-moiety": 1}},
        {"id": "P", "initial_M": 0, "pools": {"substrate-moiety": 1}}
      ],
      "catalysts": [
        {"id": "E", "amount": 2, "amount_unit": "mg", "reference_amount": 2,
         "active_fraction": 1, "deactivation_per_s": 0, "washout_fraction": 0}
      ],
      "channels": [
        {"id": "forward", "reaction_id": "S-to-P", "stoichiometry": {"S": -1, "P": 1},
         "capacity_M_s": 0.002,
         "factors": [{"species": "S", "kind": "mass-action", "scale_M": 0.01}],
         "source": "Analytic first-order test, k = 0.2/s", "catalyst": "E"}
      ],
      "phases": [{"end_s": 10.0, "dilution_per_s": 0.0, "feed_M": {}, "active_channels": ["forward"]}],
      "events": [],
      "times_s": [0, 1, 2, 3, 5, 7, 10]
    }
  ],
  "parameters": [
    {
      "id": "capacity",
      "unit": "M/s",
      "lower": 0,
      "upper": 0.02,
      "scale": 0.002,
      "targets": [
        {
          "experiment_id": "first-order-reference",
          "path": ["channels", "forward", "capacity_M_s"]
        }
      ]
    }
  ],
  "starts": [{"capacity": 0.001}, {"capacity": 0.01}],
  "observations": [
    {"id": "t-1", "experiment_id": "first-order-reference", "time_s": 1, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M",
     "value": 0.008187307530779819, "scale": 0.001, "partition": "fit",
     "source": "Analytic first-order control"},
    {"id": "t-2", "experiment_id": "first-order-reference", "time_s": 2, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M",
     "value": 0.006703200460356393, "scale": 0.001, "partition": "fit",
     "source": "Analytic first-order control"},
    {"id": "t-3", "experiment_id": "first-order-reference", "time_s": 3, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M",
     "value": 0.005488116360940264, "scale": 0.001, "partition": "fit",
     "source": "Analytic first-order control"},
    {"id": "t-5", "experiment_id": "first-order-reference", "time_s": 5, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M",
     "value": 0.0036787944117144234, "scale": 0.001, "partition": "fit",
     "source": "Analytic first-order control"},
    {"id": "t-7", "experiment_id": "first-order-reference", "time_s": 7, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M",
     "value": 0.0024659696394160645, "scale": 0.001, "partition": "evaluation",
     "source": "Analytic first-order control"},
    {"id": "t-10", "experiment_id": "first-order-reference", "time_s": 10,
     "state": "after", "quantity": "concentration", "target": "S", "unit": "M",
     "value": 0.0013533528323661271, "scale": 0.001, "partition": "evaluation",
     "source": "Analytic first-order control"}
  ],
  "uncertainty": "descriptive",
  "max_nfev": 100
}
```

The `operations` result of the fit maps each experiment id to the path of the fitted
definition, a file in the layout of an operating definition. A later step simulates
it by binding `definition` to `fit.operations.first-order-reference` (see
[Example campaigns](#example-campaigns)).

`fed-batch-fit.json`: a fit specification with the definition `fed-batch.json` as its only
experiment. Its parameters target a capacity, a saturation scale, a deactivation rate
and an event retention, and its observations use the quantities `concentration`,
`conversion`, `effluent-concentration` and `catalyst-active`, with both states. The
`experiments` list contains that definition as printed above (its id is `fed-batch`)
and is omitted here; the other fields are:

```json
{
  "id": "fed-batch-fit",
  "parameters": [
    {
      "id": "capacity",
      "unit": "M/s",
      "lower": 1e-07,
      "upper": 0.0001,
      "scale": 1e-06,
      "targets": [{"experiment_id": "fed-batch", "path": ["channels", "convert", "capacity_M_s"]}]
    },
    {
      "id": "Km",
      "unit": "M",
      "lower": 0.0001,
      "upper": 0.1,
      "scale": 0.001,
      "targets": [
        {
          "experiment_id": "fed-batch",
          "path": ["channels", "convert", "factors", "S", "scale_M"]
        }
      ],
      "profile_values": [0.001, 0.003, 0.01]
    },
    {
      "id": "deactivation",
      "unit": "1/s",
      "lower": 0,
      "upper": 0.001,
      "scale": 1e-05,
      "targets": [{"experiment_id": "fed-batch", "path": ["catalysts", "E", "deactivation_per_s"]}]
    },
    {
      "id": "retention",
      "unit": "fraction",
      "lower": 0.5,
      "upper": 1,
      "scale": 0.1,
      "targets": [{"experiment_id": "fed-batch", "path": ["events", 0, "catalyst_retention", "E"]}]
    }
  ],
  "starts": [
    {"capacity": 2e-06, "Km": 0.002, "deactivation": 1e-05, "retention": 0.9},
    {"capacity": 5e-06, "Km": 0.005, "deactivation": 3e-05, "retention": 0.7}
  ],
  "observations": [
    {"id": "s-300", "experiment_id": "fed-batch", "time_s": 300, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M", "value": 0.008994,
     "scale": 0.0001, "partition": "fit",
     "source": "Illustrative value"},
    {"id": "s-600", "experiment_id": "fed-batch", "time_s": 600, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M", "value": 0.008055,
     "scale": 0.0001, "partition": "fit",
     "source": "Illustrative value"},
    {"id": "s-900", "experiment_id": "fed-batch", "time_s": 900, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M", "value": 0.007264,
     "scale": 0.0001, "partition": "fit",
     "source": "Illustrative value"},
    {"id": "s-1000-before", "experiment_id": "fed-batch", "time_s": 1000,
     "state": "before", "quantity": "concentration", "target": "S", "unit": "M",
     "value": 0.007024, "scale": 0.0001, "partition": "fit",
     "source": "Illustrative value"},
    {"id": "s-1000-after", "experiment_id": "fed-batch", "time_s": 1000, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M", "value": 0.008512,
     "scale": 0.0001, "partition": "fit",
     "source": "Illustrative value"},
    {"id": "conversion-1200", "experiment_id": "fed-batch", "time_s": 1200,
     "state": "after", "quantity": "conversion", "target": "S", "unit": "fraction",
     "value": 0.1879, "scale": 0.01, "partition": "fit",
     "source": "Illustrative value", "reference_M": 0.01},
    {"id": "effluent-p", "experiment_id": "fed-batch", "time_s": 1100, "state": "after",
     "quantity": "effluent-concentration", "target": "P", "unit": "M", "value": 0.003107,
     "scale": 0.0001, "partition": "fit",
     "source": "Illustrative value", "collection_id": "effluent"},
    {"id": "e-active-1000-after", "experiment_id": "fed-batch", "time_s": 1000,
     "state": "after", "quantity": "catalyst-active", "target": "E", "unit": "mg",
     "value": 6.819, "scale": 0.1, "partition": "fit",
     "source": "Illustrative value"},
    {"id": "e-active-1200", "experiment_id": "fed-batch", "time_s": 1200, "state": "after",
     "quantity": "catalyst-active", "target": "E", "unit": "mg", "value": 6.693,
     "scale": 0.1, "partition": "evaluation",
     "source": "Illustrative value"},
    {"id": "s-1200", "experiment_id": "fed-batch", "time_s": 1200, "state": "after",
     "quantity": "concentration", "target": "S", "unit": "M", "value": 0.008121,
     "scale": 0.0001, "partition": "evaluation",
     "source": "Illustrative value"}
  ],
  "uncertainty": "descriptive",
  "max_nfev": 100
}
```

## Observation file

An observation file holds measurements to be compared with the simulation stored in
the record of an `operation.simulate` step. It is read by `operation.compare`
together with that record, and by `operation.compare_ensemble` together with 1 to
1000 such records. All fields are required. `operation_id` is required in a file for
`operation.compare` and refused in a file for `operation.compare_ensemble`.

- `id`: string, the name of the observation file.
- `operation_id`: the `id` of the simulated definition. A different value is refused.
  `operation.compare_ensemble` takes the id from each member and refuses a file that
  contains `operation_id`.
- `conditions`: the `conditions` of the simulated definition, repeated unchanged.
- `observations`: list of 1 to 10000 objects with unique ids.

Each observation has `id`, `time_s`, `state`, `quantity`, `target`, `unit`, `value`,
`partition` and `source`, and optionally `reference_M`, `collection_id` and
`reported_error`. The quantities, their units, the selection of time and state and
the use of `reference_M` and `collection_id` are those of the observations of a fit
request, described above, and they are checked against the recorded simulation. The
observation differs from that of a fit specification in four respects.

- There is no `experiment_id`; the observation belongs to the simulated definition.
- There is no `scale`.
- `partition`: `calibration`, `evaluation` or `reference`. The value is a label that
  the result uses to group the residuals.
- `reported_error` (optional): object `{kind, value}`, with `kind` `SD` (standard
  deviation) or `SE` (standard error) and a
  nonnegative `value` in `unit`. It records the error that the source reported; the residuals are unweighted. It is omitted when the source reported none.

The result groups the residuals (prediction minus measurement) by partition,
quantity, target, unit and `reference_M`, and gives the root-mean-square error of
each group.

For `operation.compare_ensemble`, every observation has to correspond to a recorded
state in every member, with the same units and the same `conditions`, and an
effluent observation needs the same collection interval and volume in every member.

`observations.json`: an observation file for `batch.json`. `operation.compare`
reports a root-mean-square error below 1e-12 M in both partitions:

```json
{
  "id": "later-measurements",
  "operation_id": "first-order-reference",
  "conditions": "Declared constant assay conditions",
  "observations": [
    {"id": "t-0.0", "time_s": 0.0, "state": "after", "quantity": "concentration",
     "target": "S", "unit": "M", "value": 0.01, "partition": "calibration",
     "source": "Independent analytic first-order reference",
     "reported_error": {"kind": "SD", "value": 0.0}},
    {"id": "t-1.0", "time_s": 1.0, "state": "after", "quantity": "concentration",
     "target": "S", "unit": "M", "value": 0.008187307530779819, "partition": "calibration",
     "source": "Independent analytic first-order reference",
     "reported_error": {"kind": "SD", "value": 0.0}},
    {"id": "t-5.0", "time_s": 5.0, "state": "after", "quantity": "concentration",
     "target": "S", "unit": "M", "value": 0.0036787944117144234,
     "partition": "calibration", "source": "Independent analytic first-order reference",
     "reported_error": {"kind": "SD", "value": 0.0}},
    {"id": "t-10.0", "time_s": 10.0, "state": "after", "quantity": "concentration",
     "target": "S", "unit": "M", "value": 0.0013533528323661271, "partition": "evaluation",
     "source": "Independent analytic first-order reference",
     "reported_error": {"kind": "SD", "value": 0.0}}
  ]
}
```

`fed-batch-observations.json`: an observation file for `fed-batch.json` that uses the
quantities `concentration`, `conversion`, `removed`, `effluent-concentration` and
`catalyst-active` and both states. The values are illustrative:

```json
{
  "id": "fed-batch-measurements",
  "operation_id": "fed-batch",
  "conditions": "A 10 mL reactor, dosed in the first phase and flowing in the second",
  "observations": [
    {"id": "s-600", "time_s": 600, "state": "after", "quantity": "concentration",
     "target": "S", "unit": "M", "value": 0.00889, "partition": "calibration",
     "source": "Illustrative value",
     "reported_error": {"kind": "SD", "value": 0.0002}},
    {"id": "s-1000-before", "time_s": 1000, "state": "before", "quantity": "concentration",
     "target": "S", "unit": "M", "value": 0.00783, "partition": "calibration",
     "source": "Illustrative value"},
    {"id": "s-1000-after", "time_s": 1000, "state": "after", "quantity": "concentration",
     "target": "S", "unit": "M", "value": 0.00917, "partition": "evaluation",
     "source": "Illustrative value"},
    {"id": "conversion-1200", "time_s": 1200, "state": "after", "quantity": "conversion",
     "target": "S", "unit": "fraction", "reference_M": 0.01, "value": 0.115,
     "partition": "evaluation", "source": "Illustrative value"},
    {"id": "removed-s-1200", "time_s": 1200, "state": "after", "quantity": "removed",
     "target": "S", "unit": "mol", "value": 4.55e-05, "partition": "evaluation",
     "source": "Illustrative value"},
    {"id": "effluent-p", "time_s": 1100, "state": "after",
     "quantity": "effluent-concentration", "target": "P", "unit": "M",
     "collection_id": "effluent", "value": 0.00227, "partition": "reference",
     "source": "Illustrative value"},
    {"id": "enzyme-active-1200", "time_s": 1200, "state": "after",
     "quantity": "catalyst-active", "target": "E", "unit": "mg", "value": 8.0,
     "partition": "reference", "source": "Illustrative value"}
  ]
}
```

## Assessment

An assessment states which species is the product of the simulation stored in the
record of an `operation.simulate` step, which amounts count as output, the molecular
basis of the catalysts and the lifetime thresholds. It is read by
`operation.summarize` together with that record. All fields are required except
`additional_time`.

- `id`: string, the name of the assessment.
- `operation_id` and `conditions`: those of the simulated definition, repeated
  unchanged.
- `product`: object `{species, recovery_fraction, accepted_fraction, source}`.
  `species` is a declared species. The two fractions are fractions: the share of the
  selected output that is recovered, and the share of the recovered amount that is
  accepted. They are handling assumptions stated in the file.
- `output`: object `{final_reactor, exchanges_s, collections}` that selects the
  amounts that count as output.
  - `final_reactor`: boolean, true to count the contents at the end.
  - `exchanges_s`: list of unique event times, each the `time_s` of an event of the
    definition; the amount removed at that exchange counts. `[]` for none.
  - `collections`: list of unique collection ids from the definition. The selected
    windows do not overlap. `[]` for none.
- `catalysts`: list of 0 to 50 objects `{id, active_sites_per_molecule, source}` and
  optionally `molecular_mass_g_mol`. `id` is a declared catalyst,
  `active_sites_per_molecule` is a positive number, and the mass is a positive
  number in g/mol. The entries convert the catalyst amount to moles of active sites;
  a catalyst without an entry has no turnover figures in the result. A catalyst
  whose `amount_unit` is `mol`, `mmol`, `umol` or `nmol` converts directly; one in
  `kg`, `g`, `mg` or `ug` also requires `molecular_mass_g_mol`; with any other unit,
  or without the mass, `supplied_active_site_mol` and the turnover fields per supplied
  active site are null and the diagnostics of the result name the catalyst.
- `lifetime`: list of 0 to 50 objects `{id, catalyst, minimum_active_amount, source}`.
  `catalyst` is a declared catalyst and `minimum_active_amount` is a positive number
  in the `amount_unit` of that catalyst. The result gives one `status` per threshold:
  `observed_below_threshold` with the interval between the two adjacent recorded states
  in which the active amount first falls below the threshold (an exact time when the
  fall occurs at an event), `below_at_start` when the active amount is below the
  threshold at time zero, or `above_at_all_recorded_states` when it stays at or above
  the threshold up to the last recorded time.
- `additional_time` (optional): object `{value, unit, source}`, the handling time
  beyond the simulated time. `value` is a nonnegative number and `unit` is `s`, `min`
  or `h`. Without it the result contains no productivity over the whole process.

The selected output cannot exceed the sum of the final amount of the product species in
the reactor and the amount of it removed during the simulation.

`assessment.json`: an assessment for `batch.json`. The active amount of E stays at
2 mg, so the result reports `above_at_all_recorded_states` for the lifetime threshold,
with a horizon of 10 s:

```json
{
  "id": "performance",
  "operation_id": "first-order-reference",
  "conditions": "Declared constant assay conditions",
  "product": {
    "species": "P",
    "recovery_fraction": 0.8,
    "accepted_fraction": 0.9,
    "source": "Explicit conditional product handling"
  },
  "output": {"final_reactor": true, "exchanges_s": [], "collections": []},
  "catalysts": [
    {
      "id": "E",
      "molecular_mass_g_mol": 50000,
      "active_sites_per_molecule": 1,
      "source": "Synthetic known protein reference"
    }
  ],
  "lifetime": [
    {
      "id": "useful-E",
      "catalyst": "E",
      "minimum_active_amount": 0.5,
      "source": "Authored active-amount threshold, in the catalyst's mg basis"
    }
  ],
  "additional_time": {"value": 2, "unit": "s", "source": "Explicit synthetic handling time"}
}
```

`fed-batch-assessment.json`: an assessment for `fed-batch.json` that selects the
exchange and the collection as well as the final contents:

```json
{
  "id": "fed-batch-performance",
  "operation_id": "fed-batch",
  "conditions": "A 10 mL reactor, dosed in the first phase and flowing in the second",
  "product": {
    "species": "P",
    "recovery_fraction": 0.9,
    "accepted_fraction": 1,
    "source": "Stated handling assumption for this example"
  },
  "output": {"final_reactor": true, "exchanges_s": [1000], "collections": ["effluent"]},
  "catalysts": [
    {
      "id": "E",
      "molecular_mass_g_mol": 50000,
      "active_sites_per_molecule": 1,
      "source": "Stated molecular basis for this example"
    }
  ],
  "lifetime": [
    {
      "id": "above-half",
      "catalyst": "E",
      "minimum_active_amount": 5,
      "source": "Stated threshold in mg of active enzyme"
    }
  ],
  "additional_time": {"value": 30, "unit": "min", "source": "Stated handling and isolation time"}
}
```

## Comparison file

A comparison file lists alternatives. Each alternative states a product amount, a
recovery range, an accepted-fraction range, a duration and an amount of each listed
consumable. `operation.compare_resources` multiplies each amount by the shared price of
its consumable, sums the products and divides the sum by the useful product, the
product amount multiplied by the recovery and the accepted fraction. It needs no
simulation. All fields are required.

- `id`: string, the name of the comparison.
- `product`: free text identifying the product and its form, shared by the
  alternatives.
- `cost_unit`: string, the currency or credit in which the prices are stated.
- `scope`: free text stating which consumables are included.
- `resources`: list of 1 to 64 objects `{id, unit, price}`, one per consumable. `unit` is one of `mol`,
  `mmol`, `umol` (amount), `kg`, `g`, `mg` (mass), `L`, `mL`, `uL` (volume), `h`,
  `min`, `s` (time) or `kWh`, `Wh`, `kJ` (energy). `price` is a range, in `cost_unit`
  per `unit`.
- `alternatives`: list of 1 to 20 objects, described below.

A **range** is `{low, base, high, source}` with three nonnegative numbers in that
order (`low <= base <= high`) and a source. A **quantity** is `{value, unit, source}`
with a nonnegative value and a unit of the same kind as the field it fills.

Each alternative has these required fields:

- `id` and `basis`: `id` is a string; `basis` is free text stating what the
  alternative is, such as one batch or multiple reuse cycles.
- `product`: quantity in an amount unit (`mol`, `mmol`, `umol`), the amount of
  product that the alternative makes.
- `recovery` and `accepted`: ranges with values between 0 and 1.
- `duration`: quantity with a positive value in a time unit (`h`, `min`, `s`).
- `resources`: object with one quantity for every consumable id of the file and no
  other key. A consumable that the alternative does not use is stated with `value` 0.
  The unit of each quantity must be of the same kind as the unit of the consumable (a
  mass unit for a consumable in `g`, for example) and is converted to it.

All alternatives use the same price for each consumable. A product amount of zero, or a
recovery or accepted-fraction range with `low` of zero, is accepted. A per-product
result is then undefined where the useful product is zero.

`resources.json`: a comparison of two alternatives. Both have a cost per useful
millimole of 0.3 (low), 0.75 (base) and 2.4 (high) illustrative-credit, and the
crossover prices are 2 illustrative-credit/g for `enzyme` and 0.2
illustrative-credit/mmol for `feed`:

```json
{
  "id": "batch-comparison",
  "product": "Exact product form shared by both batches",
  "cost_unit": "illustrative-credit",
  "scope": "Selected enzyme and feedstock inputs per declared batch",
  "resources": [
    {
      "id": "enzyme",
      "unit": "g",
      "price": {"low": 1, "base": 2, "high": 4, "source": "Illustrative scenario"}
    },
    {
      "id": "feed",
      "unit": "mmol",
      "price": {"low": 0.1, "base": 0.2, "high": 0.4, "source": "Illustrative scenario"}
    }
  ],
  "alternatives": [
    {
      "id": "A",
      "basis": "One explicit reaction batch",
      "product": {"value": 10, "unit": "mmol", "source": "Explicit amount per declared batch"},
      "recovery": {"low": 0.5, "base": 0.8, "high": 1, "source": "Illustrative scenario"},
      "accepted": {"low": 1, "base": 1, "high": 1, "source": "Illustrative scenario"},
      "duration": {"value": 2, "unit": "h", "source": "Explicit amount per declared batch"},
      "resources": {
        "enzyme": {"value": 2, "unit": "g", "source": "Explicit amount per declared batch"},
        "feed": {"value": 10, "unit": "mmol", "source": "Explicit amount per declared batch"}
      }
    },
    {
      "id": "B",
      "basis": "One explicit reaction batch",
      "product": {"value": 10, "unit": "mmol", "source": "Explicit amount per declared batch"},
      "recovery": {"low": 0.5, "base": 0.8, "high": 1, "source": "Illustrative scenario"},
      "accepted": {"low": 1, "base": 1, "high": 1, "source": "Illustrative scenario"},
      "duration": {"value": 2, "unit": "h", "source": "Explicit amount per declared batch"},
      "resources": {
        "enzyme": {"value": 1, "unit": "g", "source": "Explicit amount per declared batch"},
        "feed": {"value": 20, "unit": "mmol", "source": "Explicit amount per declared batch"}
      }
    }
  ]
}
```

## Example campaigns

The campaign below uses `batch.json`, `observations.json`, `assessment.json`,
`fit.json` and `resources.json` from the sections above, saved under these names in the
folder of the campaign file. Its steps give the results stated in those sections:

```yaml
id: operation-files
steps:
  - id: simulate
    capability: operation.simulate
    params:
      definition: batch.json

  - id: compare
    capability: operation.compare
    bind:
      operation: simulate.record
    params:
      comparison: observations.json

  - id: summarize
    capability: operation.summarize
    bind:
      operation: simulate.record
    params:
      assessment: assessment.json

  - id: fit
    capability: operation.fit
    params:
      definition: fit.json

  - id: simulate-fitted
    capability: operation.simulate
    bind:
      definition: fit.operations.first-order-reference

  - id: resources
    capability: operation.compare_resources
    params:
      comparison: resources.json
```

`operation.compare_ensemble` takes its simulations as a mapping from a member name to
the `record` of an `operation.simulate` step, and the shared observation file without
`operation_id`. In the campaign below, `member-0.json` and `member-1.json` are
operating definitions like `batch.json` without a catalyst and with the `id` `member-0`
and `member-1`, each with the `conditions` text of `ensemble-observations.json`, and
with a `capacity_M_s` of 0.002 and 0.003 M/s, and `ensemble-observations.json` is
`observations.json` without its `operation_id`. The root-mean-square error of
`member-0` is below 1e-12 M in both partitions, and that of `member-1` is 9.5e-4 M
(calibration) and 8.6e-4 M (evaluation):

```yaml
id: operation-ensemble
steps:
  - id: member-0
    capability: operation.simulate
    params:
      definition: member-0.json

  - id: member-1
    capability: operation.simulate
    params:
      definition: member-1.json

  - id: ensemble
    capability: operation.compare_ensemble
    bind:
      members.member-0: member-0.record
      members.member-1: member-1.record
    params:
      comparison: ensemble-observations.json
      ensemble_id: two-rows
      source: Two parameter rows of one study
```
