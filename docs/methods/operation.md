# Operation

Operation is the behavior of a reaction over time under stated operating conditions. The
capabilities simulate a reaction, fit its parameters to measured progress, compare a
simulation with later measurements, account for the output, and compare the consumables
that alternatives use.

The capabilities are `operation.define`, `operation.propagate`, `operation.simulate`,
`operation.fit`, `operation.compare`, `operation.compare_ensemble`, `operation.summarize`
and `operation.compare_resources`. The files they read are described field by field in
[Operation file formats](../operation-formats.md).

An operating definition (a definition) is the input of `operation.simulate`. It lists
species, catalysts, rate channels, phases, a state and events (see [Operating
definition](../operation-formats.md#operating-definition)). A rate channel is a reaction
with stoichiometric coefficients, a capacity and rate factors.

## Model scope

The reactor is a well-mixed liquid phase. Alambiq integrates the supplied rate channels.
A rate channel can have mass-action, saturation, inhibition and coupled-denominator
factors (see Rate factors). A definition can also contain:

- conserved pools: cofactor and chemical moiety quantities that every channel conserves;
- feed and withdrawal, with separate inlet and outlet flows;
- catalyst feed, deactivation and washout;
- discrete exchange events.

Every recorded state has an amount ledger. The ledger balances the initial, fed, formed,
consumed, removed and final amounts of a species, and the conserved pools are checked
against it.

The temperature and the pH are constant during a simulation. They affect a rate only
through dependences declared from a measured series (see Operating conditions). A rate
without a declared dependence is the same at every stated condition. Rate parameters are
supplied or fitted, and a trajectory describes the conditions its parameters came from.

A channel capacity is in moles per liter per second at the reference amount of its
catalyst, full activity and the initial volume. Three ratios scale it: the rate factors,
the active amount over the reference amount, and the initial volume over the current
volume. A channel without a named catalyst has a volumetric capacity at the current
volume, so a first-order rate constant does not change with dilution. Kinetics that
depend on the amount of enzyme require a named catalyst.

## Rate factors

Each rate factor is a function of the concentration C of one species. It is written with
x = C / scale, where `scale` is a reference concentration in molar. Every consumed species
needs a factor, so that a depleted species stops its channel.

A mass-action factor is x raised to the power 1 or 2.

A saturation factor is x^p / D^q, where p is the power of the numerator (1 or 2) and q is
the power of the denominator (1 or 2), with D = 1 + x + sum(I/Kic) + x * sum(J/Kiu) +
x * C/Ksi. I and J are the concentrations of the competitive and uncompetitive
inhibitors, Kic and Kiu are their constants, and Ksi is a self-inhibition constant, all in
molar. A competitive inhibitor adds I/Kic to the term of D that does not depend on the
substrate. An uncompetitive inhibitor adds J/Kiu to the coefficient of x. A species can
have both kinds of inhibitor at the same time.

A channel-level inhibition list divides the rate by an additional factor 1 + sum(I/Ki),
where Ki is the inhibition constant of each listed inhibitor.

A multi-substrate rate law can have one coupled denominator instead,
D = constant + sum(coefficient * product of (C/scale)^power). D is positive because its
constant is strictly positive. A cross term such as (A/Ka)(B/Kb) is one term with two
factors. A reversible rate law (Vf A/Ka - Vr P/Kp)/D is two directed channels that share
D. At equilibrium the gross extents of the forward and reverse channels remain separate,
and the net conversion is their difference.

## Flow, feed and catalyst supply

For inlet flow Qi, outlet flow Qo, volume V, the amount n of a species, its concentration
C = n/V and a direct solute input J in mol/s, a phase obeys dV/dt = Qi - Qo and
dn/dt = V * sum(coefficient * channel rate) + Qi * Cfeed - Qo * C + J.

A concentration can decrease through reaction, through inlet dilution or through both.
The recorded amounts and channel extents distinguish them. Each physical addition is
described either as a feed concentration or as a direct input; stating both counts its
amount twice. The feed ledger sums continuous feed, direct input and discrete exchange
additions.

A catalyst feed describes the combined inlet. Weight each stock stream by its inlet flow
fraction before entering it. A zero inlet flow supplies no catalyst at any stated
concentration. Let F be the catalyst supplied per second (inlet flow times amount per
liter), w = Qo/V times the washout fraction, f the inlet active fraction and k the
deactivation rate. The total amount T and the active amount A obey dT/dt = F - w T and
dA/dt = f F - (w + k) A. The total amount and the active amount each have a ledger whose
residual is zero within the numerical tolerance.

At a discrete exchange, a catalyst can be partly retained. Damage from handling is stated
as the fraction of the retained active amount that survives. The remainder is recorded as
deactivated, and fresh catalyst is added afterwards.

Collected effluent is the continuous outlet integrated over its window. The collected
amount is the integral of Qo * C dt. The mean concentration is that amount divided by the
integral of Qo dt, so a change of flow weights the mean by volume and a zero-flow period
does not contribute. An observation of a collected fraction is stated on the basis of the
fraction as collected; the model ends at collection.

## Numerical integration

Two integrators are offered: Radau (implicit, the default) and DOP853 (explicit).
Integrating the same definition with both tests whether the trajectory depends on the
integrator.

The relative tolerance is 1e-9 (dimensionless) and the absolute tolerance on concentration
is at most 1e-12 M. The tolerances control the numerical error of the integration. A step
fails with a diagnostic and produces no trajectory when a state is below -100 times the
absolute tolerance, or when a material or catalyst ledger does not close to a residual of
1e-9 relative to its scale. The uncertainty of fitted parameters is handled by
`operation.propagate`, and the comparison of predictions with observations by
`operation.compare`.

## From a measured fit to a definition

`operation.define` sets the rate parameters of a template from an `assay.fit` result. A
template is a definition whose named channels hold placeholders. For each named channel,
the capability takes the Vmax and Km of the fit and converts the measured rate to the
substrate the channel consumes, using the rate basis of the fit and the stoichiometry of
the channel. It divides by the enzyme concentration of the assay and the active fraction
of that enzyme, and scales to the reference amount of the catalyst in the initial volume:

    capacity (M/s) = Vmax_substrate / (assay enzyme per L x active fraction)
                     x reference amount / initial volume

The saturation scale is the Km of the fit, unchanged. Everything else in the template
remains as written.

The researcher supplies three inputs to this expression: the amount of enzyme in the
assay, the fraction of it that was active, and the assumption that the temperature, pH and
composition of the reactor are those of the assay. If the catalyst amount is stated as a
mass, the enzyme concentration is accepted in any mass per volume. If it is stated in
activity units, the enzyme concentration is given per liter in that unit.

A Michaelis-Menten curve fitted at low concentrations determines saturation and
inhibition only within the range it covered. If the template starts or feeds substrate
above that range, the channel is flagged as an extrapolation.

## Operating conditions

A definition states its operating conditions in `state`: a temperature in kelvin and a pH.
It can also declare the dependence of channel capacities and catalyst deactivation rates
on temperature and pH.

A channel's capacity is written at reference conditions, the conditions its kinetics were
measured at. Two factors scale it. The Arrhenius factor `exp(-Ea/R (1/T - 1/Tref))`
covers temperature. The bell-shaped activity at the stated pH divided by the activity at
the reference pH, `1 / (1 + 10^(pKa1 - pH) + 10^(pH - pKa2))`, covers pH. The deactivation
rate of a catalyst has its own Arrhenius factor. Every multiplier is reported with the
simulation.

The dependences are derived from measured series fitted with `assay.fit`:

- At least 4 rate constants at 3 or more distinct temperatures give the activation energy
  (`arrhenius`).
- Activity remaining over time at one temperature gives the inactivation rate and its
  half-life (`first_order_decay`).
- The inactivation rate constants of the same series repeated across temperatures give
  the activation energy of the inactivation (`arrhenius`).
- Activity across pH gives the two pKa values and the optimum (`ph_bell`).

`operation.define` converts these fits into the declared dependences when given the
temperature and pH of the kinetics assay.

Thermal inactivation and the temperature dependence of the catalytic rate are separate
processes, measured separately, each with its own activation energy.

Data determine a dependence only within the range over which it was measured.
`operation.simulate` applies it at any stated temperature and pH, so the researcher
compares the stated conditions with the measured range. The limits of the bell-shaped
profile are stated in [Measurements](measurements.md#response-models).

## Reversible reactions and equilibrium

A reversible reaction needs a forward and a reverse channel with equal equilibrium
positions. The rate law is `v = (Vf S/Ks - Vr P/Kp) / (1 + S/Ks + P/Kp)`. S and P are the
concentrations of substrate and product, Vf and Vr are the forward and reverse
capacities, and Ks and Kp are the Michaelis constants of substrate and product. The
channels reach equilibrium where `P/S = (Vf/Ks) / (Vr/Kp)`, the Haldane relation.

`operation.define` takes the forward kinetics from a fit, the Michaelis constant of the
product and the equilibrium constant. It derives the reverse capacity so that the Haldane
relation holds. The equilibrium ratio P/S then equals the equilibrium constant for any
Michaelis constant of the product; that constant changes only the rate at which
equilibrium is approached.

The equilibrium constant can be given directly or as the `ln_k_prime` reported by
`reaction.estimate_thermodynamics`. That estimate applies at 298.15 K.
`operation.define` uses the constant uncorrected for temperature, because a correction
requires the reaction enthalpy. The capability reports this when the definition states
another temperature.

## Prediction uncertainty

`operation.propagate` converts the uncertainty of a fitted Michaelis-Menten rate into a
spread of predicted concentrations. It draws coefficient sets from the fit, builds and
simulates the definition for each set, and reports the quantiles at the requested levels
(default 0.05, 0.5 and 0.95) at the recorded times.

The coefficient sets come from the covariance of the fit (a normal approximation that
assumes jointly normal coefficients) or from the sets resampled by a cluster bootstrap,
which are used directly.

Only the fitted coefficients vary. The enzyme amount of the assay, its active fraction,
the other rates of the template, the dependences on temperature and pH, and the model
itself are taken as exact, so the spread is a lower bound on the uncertainty of the
prediction.

The result reports the number of draws discarded because Km or Vmax was not positive. A
note is issued when the discarded draws exceed 20 % of the samples. The call fails when
more than 98 % of the draws are discarded.

## Definition requirements

Before assigning a capacity, establish the direction of the reaction, every participant,
the concentration units, and the quantity the measured rate was measured against. Forward
and reverse channels have separate rates and separate sources. A consumed species without
a concentration-dependent factor is consumed regardless of its amount.

Conserved pools are declared for the chemical moieties and cofactors that balance across
all channels. A declared pool is checked when the definition is read. Alambiq refuses a
channel whose stoichiometric coefficients, weighted by the pool numbers of the species,
do not sum to 0, and the message names the pool and the channel. The integrated ledger is
checked separately (see Numerical integration).

Catalyst amount, unit, reference amount, active fraction and deactivation rate are
separate fields. Measured retained activity combines deactivation with physical loss.
Separating the two requires independent measurements or a stated scenario.

## Fitting parameters to measured progress

`operation.fit` adjusts named fit parameters of 1 or more definitions to minimize the
difference between predictions and the supplied observations. The observations are
concentrations, conversions against a reference concentration, collected effluent as a
flow-weighted mean over its collection window, or catalyst amounts.

A fit parameter can be assigned to several fields, so that one physical constant is shared
across experiments, exchanges, or both directions of a reaction. Fields not assigned to a
fit parameter remain as given. These fields can be assigned:

- rate capacities;
- saturation and inhibition constants;
- denominator coefficients and scales;
- the amount, active fraction, deactivation and washout of a catalyst;
- the retention fractions of an exchange.

The volume, the liquid flows, the catalyst inlet and the direct solute inputs remain as
declared, so a rate fitted in a fed batch is conditional on the stated dosing and its
normalization basis.

The request assigns each observation to fitting or to evaluation before the fit.
Evaluation observations are excluded from the fit. The fit error and the evaluation error
are each reported as the root-mean-square of the residuals divided by the `scale` of each
observation (the standardized RMSE). The evaluation error estimates predictive
performance.

The fit reports four diagnostics:

- A parameter at a bound (within 1e-4 of the bound, in units of the parameter scale) is
  reported as a limit of the search range.
- A rank of the Jacobian below the number of parameters means that the data do not
  determine at least one combination of parameters. Singular values below 1e-7 times the
  largest are counted as zero.
- The scaled condition number is the largest over the smallest singular value of the
  Jacobian. No threshold is applied to it. A value above 1e7 coincides with a rank below
  the number of parameters; a larger value means that at least one combination of
  parameters is poorly determined.
- The profile of a parameter with `profile_values` (up to 50 values) gives the sum of
  squared standardized residuals with that parameter fixed at each value and the other
  parameters re-fitted; a flat profile means that the objective does not change along the
  parameter.

The fit reports local 95 % intervals when `uncertainty` is `independent-residuals`, the
rank equals the number of parameters and no parameter is at a bound. The intervals assume
independent residuals with a common variance after division by the observation `scale`.
The 95 % coverage applies when residuals are independent. Residuals of progress curves
from one experiment are correlated in time. Overlapping collection windows, and several
species measured in one collected fraction, share measurement errors, which departs from
the same assumption.

The optimizer is bounded trust-region least squares on scaled parameters. It runs from
every supplied start and reports the converged start with the lowest objective. A
converged start gives a local solution, and different starts or bounds can give a
different result. The fit fails if no start converges.

If an active fraction and a catalytic capacity are estimated together, the data determine
only their product. Recovery and deactivation parameters are separable only with direct
measurements of recovery and of activity.

## Comparing a simulation with measurements

`operation.compare` matches each observation to the recorded state it describes and
reports residuals grouped by partition, quantity, target and unit. A residual is the
prediction minus the measurement. The prediction is the saved one, so a comparison uses
the rate parameters and the conditions of the simulation it reads.

Before matching, the capability checks that:

- the `operation_id` and the `conditions` text of the observation file equal those of the
  simulation;
- each observation time is a recorded state (before an exchange only at an exchange
  time);
- each unit is the unit of its quantity.

It refuses observations that fail any check, including observations for a different
operating definition. Observations taken before and after an exchange describe different
states. A conversion needs its reference concentration.

A conversion is computed from concentration as one minus the concentration divided by its
reference concentration. When the volume changes, the conversion includes dilution by the
inlet. The recorded amounts and channel extents give the quantity the reactor converted.

Reported standard deviations and standard errors are stored as distinct quantities.
Unreported errors stay unreported. The residual summary is unweighted.

An ensemble comparison matches observations to each member of a set of saved simulations,
for example one simulation per row of source parameters. It reports the pointwise median
and range across members beside the errors of each member. The range describes the
members that were included, and adding members widens it. Every observation must map to a
recorded state of every member with the same units. A collected-effluent observation must
have the same collection interval and volume in every member.

## Accounting for output and catalyst use

`operation.summarize` builds a ledger from a saved simulation and an assessment. The
assessment is the input that selects the output and states the recovery and accepted
fractions (see [Assessment](../operation-formats.md#assessment)). The ledger holds:

- the initial, fed, formed, consumed, removed and final amounts;
- the selected output from the final reactor contents, discrete exchanges and continuous
  collections;
- the useful output after the stated recovery and accepted fractions, which are handling
  assumptions.

Channel extent and net product differ for reversible reactions and for reaction cascades,
and the ledger reports both.

Turnover per supplied active site requires an active-site count above 0 and, for a
catalyst counted in a mass unit, a molecular mass. Otherwise, and for any other amount
unit, it is reported as unresolved. The molecular mass, the amount and the active-site
count must refer to the same protein form or oligomer.

A usefulness threshold is a minimum active catalyst amount (`minimum_active_amount`, in
the `amount_unit` of the catalyst). The capability reports the time at which the active
amount first falls below it. A crossing between recorded states is reported as the
interval between them. A crossing at a discrete exchange has the time of the exchange.
For a threshold never reached, the result states that it is not reached within the
recorded time span. Denser recording narrows both. Replenishment can restore activity, so
every transition is stored, not only the first. The threshold applies to the active
catalyst amount only.

The continuous collection windows chosen for the selected output must not overlap.

## Comparing the consumables used by alternatives

A consumable is an amount of material, time or energy that has a price per unit in a
comparison file. The `resources` field of the file lists the consumables.

`operation.compare_resources` sums the listed consumable amounts multiplied by shared
prices. It reports the cost per useful millimole as low, base and high values over the
declared ranges of the prices and of the recovery and accepted fractions. It also gives
the price at which two alternatives cost the same, with all other prices and the recovery
and accepted fractions at their base values. The low, base and high values are scenarios
chosen by the researcher.

An alternative declares the total amount of each consumable and the total product amount
over the cycles it covers. Each alternative lists every consumable, with a value of 0 for
a consumable it does not use. All alternatives use the same prices, so a change in a price
changes the cost of both alternatives in a pairwise comparison.

The comparison covers the consumables that were listed. Capital, labor, utilities and
waste treatment enter only when they are entered as consumables.

A crossover price outside the declared range is reported and marked as outside it. When a
range includes zero useful product, the per-product quantities are undefined and the
result states this.

## Limits

A trajectory describes the parameters and conditions it was given. The exception is the
dependence on temperature and pH declared from measured series, which applies within the
range those series covered. A result applies where mixing, mass transfer, heat and reactor
geometry do not limit the process.

The intervals of a fit assume independent residuals, which the residuals of progress
curves violate when they are correlated in time. An ensemble range describes the supplied
members. A cost comparison describes the listed consumables.
