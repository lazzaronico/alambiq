# Operation

The behavior of a reaction over time under stated operating conditions: simulation,
fitting of parameters to measured progress, comparison of a simulation with later
measurements, accounting of output, and comparison of the consumables that alternatives
use. The capabilities are `operation.define`, `operation.propagate`,
`operation.simulate`, `operation.fit`, `operation.compare`, `operation.compare_ensemble`,
`operation.summarize` and `operation.compare_resources`. The files they read are
described field by field in [Operation file formats](../operation-formats.md).

An operating definition (a definition) is the input of `operation.simulate`: species,
catalysts, rate channels, phases, a state and events (see [Operating
definition](../operation-formats.md#operating-definition)). A rate channel is a reaction
with stoichiometric coefficients, a capacity and rate factors.

## Model scope

The reactor is a well-mixed liquid phase. Supplied rate channels are integrated. A rate
channel can have mass-action, saturation, inhibition and coupled-denominator factors (see
Rate factors). A definition can also contain conserved pools (cofactor and chemical
moiety quantities that every channel conserves), feed and withdrawal with separate inlet
and outlet flows, catalyst feed, deactivation and washout, and discrete exchange events.
Every recorded state has an amount ledger, which is the balance of the initial, fed,
formed, consumed, removed and final amounts of a species, and the conserved pools are
checked against it.

Temperature and pH affect a rate only through dependences declared from a measured series
(see Operating conditions). A rate without a declared dependence is the same at every
stated condition. A trajectory describes the conditions its parameters came from.
The temperature and the pH are constant during a simulation. Rate parameters are
supplied or fitted.

A channel capacity is in moles per liter per second at the reference amount of its
catalyst, full activity and the initial volume. It is scaled by the rate factors, by the
ratio of active amount to reference amount, and by the ratio of initial volume to current
volume. A channel without a named catalyst has a volumetric capacity at the current
volume, so a first-order rate constant does not change with dilution. Kinetics that
depend on the amount of enzyme require a named catalyst.

## Rate factors

Each rate factor is a function of the concentration C of one species, written with
x = C / scale, where `scale` is a reference concentration in molar. A mass-action factor
is x raised to the power 1 or 2. A saturation factor is x^p / D^q, where p is the power
of the numerator (1 or 2) and q is the power of the denominator (1 or 2), with
D = 1 + x + sum(I/Kic) + x * sum(J/Kiu) + x * C/Ksi. I and J are the concentrations of
the competitive and uncompetitive inhibitors, Kic and Kiu are their constants, and Ksi is
a self-inhibition constant, all in molar. A competitive inhibitor adds I/Kic to the term
of D that does not depend on the substrate; an uncompetitive inhibitor adds J/Kiu to the
coefficient of x. A species can have competitive and uncompetitive inhibitors at the same
time. A channel-level inhibition list divides the rate by an additional factor
1 + sum(I/Ki), where Ki is the inhibition constant of each listed inhibitor. Every
consumed species needs a factor, so that a depleted species stops its channel.

A multi-substrate rate law can instead have one coupled denominator, D = constant +
sum(coefficient * product of (C/scale)^power). D is positive because its constant is
strictly positive. Cross terms such as (A/Ka)(B/Kb) are one term with two factors, and a
reversible rate law (Vf A/Ka - Vr P/Kp)/D is two directed channels that share D. The
gross extents of the forward and reverse channels remain separate at equilibrium; the net
conversion is their difference.

## Flow, feed and catalyst supply

For inlet flow Qi, outlet flow Qo, volume V, the amount n of a species, its concentration
C = n/V and a direct solute input J in mol/s, a phase obeys dV/dt = Qi - Qo and dn/dt =
V * sum(coefficient * channel rate) + Qi * Cfeed - Qo * C + J. A concentration can decrease
through reaction, through inlet dilution or through both; the recorded amounts and channel
extents distinguish them. Each physical addition is described either as a feed
concentration or as a direct input; stating both counts its amount twice. The feed ledger
sums continuous feed, direct input and discrete exchange additions.

A catalyst feed describes the combined inlet. Weight each stock stream by its inlet flow
fraction before entering it. A zero inlet flow supplies no catalyst at any stated
concentration. With F the catalyst supplied per second (inlet flow times amount per
liter), w = Qo/V times the washout fraction, f the inlet active fraction and k the
deactivation rate, the total amount T and the active amount A obey dT/dt = F - w T and
dA/dt = f F - (w + k) A. At a discrete exchange, a catalyst can be partly retained.
Damage from handling is stated as the fraction of the retained active amount that
survives; the remainder is recorded as deactivated, and fresh catalyst is added
afterwards. The total amount and the active amount each have a ledger whose residual is
zero within the numerical tolerance.

Collected effluent is the continuous outlet integrated over its window. The collected
amount is the integral of Qo * C dt, and the mean concentration is that amount divided by
the integral of Qo dt, so a change of flow weights the mean by volume and a zero-flow
period does not contribute. An observation of a collected fraction is stated on the basis
of the fraction as collected; the model ends at collection.

## Numerical integration

Two integrators are offered: Radau (implicit, the default) and DOP853 (explicit).
Integrating the same definition with both tests whether the trajectory depends on the
integrator. The relative tolerance is 1e-9 (dimensionless) and the absolute tolerance on
concentration is at most 1e-12 M. A state below -100 times the absolute tolerance, or a
material or catalyst ledger that does not close to a residual of 1e-9 relative to its scale,
fails the step with a diagnostic and produces no trajectory. The tolerances control the
numerical error of the integration; `operation.propagate` propagates the uncertainty of
fitted parameters, and `operation.compare` compares predictions with observations.

## From a measured fit to a definition

`operation.define` sets the rate parameters of a template (a definition whose named
channels hold placeholders) from an `assay.fit` result. For each named channel it takes
the Vmax and Km of the fit and converts the measured rate to the substrate the channel
consumes, using the rate basis of the fit and the stoichiometry of the channel. It
divides by the enzyme concentration of the assay and the active fraction of that enzyme,
and scales to the reference amount of the catalyst in the initial volume:

    capacity (M/s) = Vmax_substrate / (assay enzyme per L x active fraction)
                     x reference amount / initial volume

The saturation scale is the Km of the fit, unchanged. Everything else in the template
remains as written.

Three inputs to this expression are statements supplied by the researcher: the amount of
enzyme in the assay, the fraction of it that was active, and the assumption that the
temperature, pH and composition of the reactor are those of the assay. If the catalyst
amount is stated as a mass, the enzyme concentration is accepted in any mass per volume;
if it is stated in activity units, the enzyme concentration is given per liter in that
unit. If the template starts or feeds substrate above the range the fit covered, the
channel is flagged as an extrapolation: a Michaelis-Menten curve fitted at low
concentrations determines saturation and inhibition only within that range.

## Operating conditions

A definition can state its operating conditions (`state`: a temperature in kelvin and a
pH) and declare the dependence of channel capacities and catalyst deactivation rates on
temperature and pH. A channel's capacity is written at reference conditions, the
conditions its kinetics were measured at. It is scaled by the Arrhenius factor
`exp(-Ea/R (1/T - 1/Tref))` for temperature, and by the bell-shaped activity at the
stated pH divided by the activity at the reference pH,
`1 / (1 + 10^(pKa1 - pH) + 10^(pH - pKa2))`. The deactivation rate of a catalyst has its
own Arrhenius factor. Every multiplier is reported with the simulation.

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

A dependence is determined by data only within the range over which it was measured.
`operation.simulate` applies it at any stated temperature and pH; the researcher compares
the stated conditions with the measured range. The limits of the bell-shaped profile are
stated in [Measurements](measurements.md#response-models). Thermal inactivation and the
temperature dependence of the catalytic rate are separate processes, measured separately
and each with its own activation energy.

## Reversible reactions and equilibrium

A reversible reaction needs a forward and a reverse channel with equal equilibrium
positions. With the rate law `v = (Vf S/Ks - Vr P/Kp) / (1 + S/Ks + P/Kp)`, where S and P
are the concentrations of substrate and product, Vf and Vr are the forward and reverse
capacities, and Ks and Kp are the Michaelis constants of substrate and product, the
channels reach equilibrium where `P/S = (Vf/Ks) / (Vr/Kp)`, the Haldane relation.
`operation.define` takes the forward kinetics from a fit, the Michaelis constant of the
product and the equilibrium constant, and derives the reverse capacity so that the
Haldane relation holds. The equilibrium ratio P/S then equals the equilibrium constant
for any Michaelis constant of the product, which changes only the rate at which
equilibrium is approached. The constant can be given directly or as the `ln_k_prime`
reported by `reaction.estimate_thermodynamics`. That estimate applies at 298.15 K.
`operation.define` uses the constant uncorrected for temperature; a correction requires
the reaction enthalpy, and the capability reports this when the definition states another
temperature.

## Prediction uncertainty

`operation.propagate` converts the uncertainty of a fitted Michaelis-Menten rate into a
spread of predicted concentrations. It draws coefficient sets from the fit (its
covariance, or the sets resampled by a cluster bootstrap), builds and simulates the
definition for each set, and reports the quantiles at the requested levels (default 0.05,
0.5 and 0.95) at the recorded times. Only the fitted coefficients vary. The enzyme amount
of the assay, its active fraction, the other rates of the template, the dependences on
temperature and pH, and the model itself are taken as exact, so the spread is a lower
bound on the uncertainty of the prediction. The normal approximation assumes jointly
normal coefficients. The number of draws discarded because Km or Vmax was not positive is
reported. A note is issued when the discarded draws exceed 20 % of the samples, and the
call fails when more than 98 % of the draws are discarded. The cluster bootstrap uses the
resampled coefficient sets directly.

## Definition requirements

Before assigning a capacity, establish the direction of the reaction, every participant,
the concentration units, and the quantity the measured rate was measured against.
Forward and reverse channels have separate rates and separate sources. Every consumed
species needs a concentration-dependent factor; without one it is consumed regardless of
its amount.

Conserved pools are declared for the chemical moieties and cofactors that balance across
all channels. A declared pool is checked when the definition is read: Alambiq refuses a
channel whose stoichiometric coefficients, weighted by the pool numbers of the species,
do not sum to 0, and the message names the pool and the channel. The integrated ledger is
checked separately (see Numerical integration).

Catalyst amount, unit, reference amount, active fraction and deactivation rate are
separate fields. Measured retained activity combines deactivation with physical loss;
separating the two requires independent measurements or a stated scenario.

## Fitting parameters to measured progress

`operation.fit` adjusts named fit parameters of 1 or more definitions to minimize the
difference between predictions and the supplied observations: concentrations, conversions
against a reference concentration, collected effluent as a flow-weighted mean over its
collection window, or catalyst amounts. A fit parameter can be assigned to several
fields, so that one physical constant is shared across experiments, exchanges, or both
directions of a reaction. Fields not assigned to a fit parameter remain as given. Only
rate capacities, saturation and inhibition constants, denominator coefficients and
scales, the amount, active fraction, deactivation and washout of a catalyst, and the
retention fractions of an exchange can be assigned. The volume, the liquid flows, the
catalyst inlet and the direct solute inputs remain as declared, so a rate fitted in a fed
batch is conditional on the stated dosing and its normalization basis.

The request assigns each observation to fitting or to evaluation before the fit;
evaluation observations are excluded from the fit. The fit error and the evaluation
error are each reported as the root-mean-square of the residuals divided by the `scale`
of each observation (the standardized RMSE); the evaluation error estimates predictive
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
Residuals of progress curves from one experiment are correlated in time; the 95 %
coverage of the intervals applies when residuals are independent. Overlapping collection
windows, and several species measured in one collected fraction, share measurement
errors, which departs from the same assumption.

The optimizer is bounded trust-region least squares on scaled parameters. A start that
the solver reports as converged gives a local solution, and the fit fails if no start
converges. It runs from every supplied start and reports the converged start with
the lowest objective; different starts or bounds can give a different result. If an
active fraction and a catalytic capacity are estimated together, the data determine only
their product. Recovery and deactivation parameters are separable only with direct
measurements of recovery and of activity.

## Comparing a simulation with measurements

`operation.compare` matches each observation to the recorded state it describes and
reports residuals grouped by partition, quantity, target and unit. The prediction is the
saved one, so a comparison uses the rate parameters and the conditions of the simulation
it reads. Observations for a different operating definition are refused. A residual is the
prediction minus the measurement.

A conversion is computed from concentration as one minus the concentration divided by its
reference concentration. When the volume changes, the conversion includes dilution by the
inlet. The recorded amounts and channel extents give the quantity the reactor converted.

Before matching, the capability checks that the `operation_id` and the `conditions` text
of the observation file equal those of the simulation, that each observation time is a
recorded state (before an exchange only at an exchange time), and that each unit is the
unit of its quantity. It refuses observations that fail any check. Observations taken
before and after an exchange describe different states, and a conversion needs its
reference concentration. Reported standard deviations and standard errors are stored as
distinct quantities, unreported errors stay unreported, and the residual summary is
unweighted.

An ensemble comparison matches observations to each member of a set of saved simulations,
for example one simulation per row of source parameters, and reports the pointwise median
and range across members beside the errors of each member. The range describes the
members that were included, and adding members widens it. Every observation must map to a
recorded state of every member with the same units, and a collected-effluent observation
must have the same collection interval and volume in every member.

## Accounting for output and catalyst use

`operation.summarize` builds a ledger from a saved simulation and an assessment (the
input that selects the output and states the recovery and accepted fractions; see
[Assessment](../operation-formats.md#assessment)): initial, fed, formed, consumed,
removed and final amounts; the selected output from the final reactor contents, discrete
exchanges and continuous collections; and the useful output after the stated recovery and
accepted fractions. The recovery and accepted fractions are handling assumptions.

Turnover per supplied active site requires an active-site count above 0 and, for a
catalyst counted in a mass unit, a molecular mass; otherwise, and for any other amount
unit, it is reported as unresolved. Channel extent and net product
differ for reversible reactions and for reaction cascades, and the ledger reports both.

A usefulness threshold is a minimum active catalyst amount (`minimum_active_amount`, in
the `amount_unit` of the catalyst). The capability reports the time at which the active
amount first falls below it. A crossing between recorded states is reported as the
interval between them; for a threshold never reached, the result states that it is not
reached within the recorded time span; denser recording narrows both. A crossing at a
discrete exchange has the time of the exchange. Replenishment can restore activity, so
every transition is stored, not only the first. The threshold applies to the active
catalyst amount only. The continuous collection windows chosen for the selected output
must not overlap. The molecular mass, the amount and the active-site count must refer to
the same protein form or oligomer.

## Comparing the consumables used by alternatives

A consumable is an amount of material, time or energy that has a price per unit in a
comparison file; the file's key `resources` lists the consumables.

`operation.compare_resources` sums the listed consumable amounts (the `resources` field of
the file) multiplied by shared prices, and reports the cost per useful millimole as low,
base and high values over the declared ranges of the prices and of the recovery and
accepted fractions. It also gives the price at which two alternatives cost the same with
all other prices and the recovery and accepted fractions at their base values.

The comparison covers the consumables that were listed. Capital, labor, utilities and
waste treatment enter only when they are entered as consumables. Each alternative lists
every consumable, with a value of 0 for a consumable it does not use. The low, base and
high values are scenarios chosen by the researcher.

An alternative declares the total amount of each consumable and the total product amount
over the cycles it covers. All alternatives use the same prices, so a change in a price
changes the cost of both alternatives in a pairwise comparison. A crossover price outside
the declared range is reported and marked as outside it. When a range includes zero useful
product, the per-product quantities are undefined and the result states this.

## Limits

A trajectory depends on the parameters and conditions it was given and describes the
conditions those parameters came from. The exception is the dependence on temperature and
pH declared from measured series, which applies within the range those series covered.
A result applies where mixing, mass transfer, heat and reactor geometry do not limit the
process.

The intervals of a fit assume independent residuals, which the residuals of progress
curves violate when they are correlated in time. An ensemble range describes the supplied
members. A cost comparison describes the listed consumables.
