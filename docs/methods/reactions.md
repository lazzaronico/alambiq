# Reactions

Four capabilities cover reactions. `reaction.inspect` accounts for a reaction species by
species. `reaction.estimate_thermodynamics` estimates its transformed Gibbs energy.
`reaction.explore_routes` searches for routes from supplied feedstocks to a target with
supplied reaction operators. `reaction.analyze_enzyme_panel` evaluates how well predictors
fitted to measured enzyme activities predict held-out measurements. Species, operators and
conditions are those stated in the request.

## Reaction equations

`reaction.inspect` checks the balance of an equation written with the species, protons,
charges and stereochemistry stated in the request.

Every participant is one explicit species with a role and an exact coefficient. A species is
a SMILES that states its protons, charge and stereochemistry as written. Protons, water,
cofactors and counter-ions are participants. The equation is balanced in elements, isotopes
and charge with exact rational arithmetic. The residual is products minus reactants. An
equation whose species cancel completely is reported as a zero net transformation.

Species identity is the canonical isomeric SMILES. Stereochemistry, charge, isotopes and
disconnected fragments are all part of it, and a neutral acid and its carboxylate are two
species. The request fixes the form. A reference structure or database accession can be
attached to a species; a wrong enantiomer, protonation or isotope then appears as a
mismatch.

Atom maps are optional. Bond changes are reconstructed when the maps cover every heavy atom
one to one with unit coefficients. Otherwise the status gives the reason: absent, partial,
ambiguous through coefficients, or inconsistent. Reaction conditions (pH, temperature and
ionic strength, as supplied) are recorded as context and do not change the balance.

## Thermodynamics

`reaction.estimate_thermodynamics` estimates the transformed Gibbs energy of a reaction with
the component-contribution model of eQuilibrator. The standard energies ΔG′° are at 298.15 K
for the stated pH, pMg and ionic strength. The estimate is corrected to the stated
activities: ΔG′ = ΔG′° + RT ln Q, where Q is the reaction quotient at those activities. For
each reaction and for a weighted route, the capability reports ΔG′°, ΔG′, ln K′ and a model
standard deviation.

Three properties of the estimate:

- An estimate is a model prediction of the equilibrium constant. The reported standard
  deviation is that of the model, conditional on the stated conditions and activities.
- The uncertainty of a route uses the full covariance between the estimates of its
  reactions, so a reaction used twice is not counted as two independent estimates.
- A reaction that cannot be estimated reports the reason: an unresolved accession or
  compound, an unresolved formula, an equation unbalanced in an element other than
  hydrogen, missing dissociation data, or a direction the model leaves unconstrained. A
  route containing it is left unresolved.

The estimate applies to aqueous solution at 298.15 K, at the stated activities and ionic
strength. The sign of ΔG′ gives the direction of the reaction at the stated activities when
its magnitude exceeds the model standard deviation. The rate depends on barriers and
catalysts, which the estimate does not include.

## Route search

`reaction.explore_routes` applies the supplied mapped reaction operators to the supplied
feedstocks, one generation at a time. A generation is one round of expansion. The capability
then traces finite amounts of each species through the resulting reaction graph, in
increments (the extent per step) set in the request. The result depends on the supplied
operators. Every generated equation is checked for balance, and no equation with a residual
is retained.

The request bounds the expansion with `generations`, `max_recipes` (operator applications),
`max_products_per_recipe`, `max_nodes` (molecules), `max_edges` (reactions) and `max_atoms`
(atoms per molecule). The route trace has its own bounds: `max_route_states`, `max_routes`
and `max_route_steps`. The result reports the bounds that were reached.

- If no bound was reached, a target that is not reached is not reachable by these operators
  within these bounds.
- If a bound was reached, the search can be incomplete and the target can be reachable. A
  search that stopped at a bound is incomplete.
- The generation bound is reported whenever the last permitted generation runs, including
  when the graph was fully expanded by then.
- Candidate ranking is capped at the product of the bounds `max_recipes` and `max_nodes`.
  Reaching the cap stops the expansion, retains the completed generations and is reported.

The extent per step sets the resolution of the inventory trace, and a smaller extent needs
more route steps. Catalyst requirements and exact-equation enzyme references are stored with
each route step. A reference attaches to the exact reactant and product sets it was reported
for. When the supplied operators generate two stereoisomers of a product, both are retained
as separate species. A route returned by the capability is a hypothesis for a further
analysis, for example `reaction.estimate_thermodynamics`.

## Activity prediction across enzymes and reactions

`reaction.analyze_enzyme_panel` measures how well models trained on measured enzyme
activities under stated conditions predict held-out measurements. The held-out enzymes,
reactions or both are new.

The capability compares five predictors: the training mean, the nearest measured reference,
and three kernel ridge regressions (a sequence-only model, a reaction-only model and a joint
model). The sequence kernel is the cosine similarity of normalized amino-acid 3-mer counts.
The reaction kernel uses chirality-aware radius-2 Morgan presence fingerprints of the
selected reaction participants (512 bits per reaction side). The joint kernel multiplies both
similarities. The nearest measured reference is the one with the highest joint similarity.
The request supplies the ridge penalty `alpha`, which must be chosen without the held-out
observations.

The split is declared before fitting. A sequence split separates exact sequences, and a
reaction split separates equations. A reaction split treats an equation and a common
positive multiple of it as the same equation. Observations that share a sequence, an enzyme
form, a reaction and a condition (replicates) are assigned to the same side of the split.
Homologs and chemically similar substrates can cross these boundaries, and the result
reports the similarities across the split. A family-disjoint test requires partitioning
outside the capability.

Only measured finite values are used. Inactive, failed, untested and unreported observations
are retained and counted, and are not converted to zeros. Errors are computed only on
observations with finite measured values, under the stated conditions. A predictor shows a
gain over a baseline (the training mean or the nearest measured reference) when its held-out
error is lower than that of the baseline.

Screen conditions can change together with the substrates, so on a held-out screen the
effect of the conditions is confounded with the effect of the substrates. A reaction-only
model gives every enzyme the same prediction on one reaction. The rank correlation of a
predictor is reported as unavailable when its predictions or the observed values are
constant, as for the reaction-only model on a held-out set of one reaction.
