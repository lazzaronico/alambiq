# Reactions

Accounting for a reaction species by species, estimating its transformed Gibbs energy,
searching for routes from supplied feedstocks to a target with supplied reaction
operators, and evaluating how well predictors fitted to measured enzyme activities predict held-out
measurements.
The capabilities are `reaction.inspect`, `reaction.estimate_thermodynamics`,
`reaction.explore_routes` and `reaction.analyze_enzyme_panel`. Species,
operators and conditions are those stated in the request. `reaction.estimate_thermodynamics`
takes standard energies from the component-contribution model of eQuilibrator.

## Reaction equations

`reaction.inspect` uses the species, protons, charges and stereochemistry as stated in the request.
Every participant is one explicit species (a SMILES that states its protons, charge and
stereochemistry as written) with a role and an exact coefficient; protons, water,
cofactors and counter-ions are participants. The equation is balanced in elements,
isotopes and charge with exact rational arithmetic. The residual is products minus
reactants. An equation whose species cancel completely is reported as a zero net
transformation.

Species identity is the canonical isomeric SMILES: stereochemistry, charge, isotopes and
disconnected fragments are all part of it, and a neutral acid and its carboxylate are two
species. The request fixes the form. A reference structure
or database accession can be attached to a species, and a wrong enantiomer, protonation
or isotope then appears as a mismatch.

Atom maps are optional. Bond changes are reconstructed when the maps cover every heavy
atom one to one with unit coefficients; otherwise the status gives the reason (absent,
partial, ambiguous through coefficients, or inconsistent). Reaction conditions (pH,
temperature and ionic strength, as supplied) are recorded as context and do not change
the balance.

## Thermodynamics

`reaction.estimate_thermodynamics` gives standard transformed Gibbs energies at 298.15 K
from the component-contribution model of eQuilibrator, for stated pH, pMg and ionic
strength, and corrects to the stated activities: ΔG′ = ΔG′° + RT ln Q, where Q is the
reaction quotient at those activities. It reports ΔG′°,
ΔG′, ln K′ and a model standard deviation for each reaction and for a weighted route.

Three properties of the estimate:

- An estimate is a model prediction of the equilibrium constant, and the reported
  standard deviation is that of the model, conditional on the stated conditions and
  activities.
- The uncertainty of a route uses the full covariance between the estimates of its
  reactions. A reaction used twice is not counted as two independent estimates.
- A reaction that cannot be estimated reports the reason (an unresolved accession or
  compound, an unresolved formula, an equation unbalanced in an element other than
  hydrogen, missing dissociation data, or a direction the model leaves unconstrained), and
  a route containing it is left unresolved.

The estimate applies to aqueous solution at 298.15 K, at the stated activities and ionic
strength.

The sign of ΔG′ gives the direction of the reaction at the stated activities when its
magnitude exceeds the model standard deviation; the rate depends on barriers and
catalysts, which the estimate does not include.

## Route search

`reaction.explore_routes` applies the supplied mapped reaction operators to the supplied
feedstocks, generation by generation (a generation is one round of expansion), within the
bounds of the request: `generations`, `max_recipes` (operator applications),
`max_products_per_recipe`, `max_nodes` (molecules), `max_edges` (reactions) and `max_atoms`
(atoms per molecule). It then traces finite amounts of each species through the resulting
reaction graph in increments (the extent per step) set in the request. The result depends on
the supplied operators. If no bound was reached, a target that is not reached is not
reachable by these operators within these bounds. If a bound was reached (see below), the
search can be incomplete and the target can be reachable. The route trace has its own
bounds: `max_route_states`, `max_routes` and `max_route_steps`.

Every generated equation is checked for balance, and no equation with a residual is
retained. The bounds that were reached are reported. A search that stopped at a bound is
incomplete. The generation bound is reported whenever the last permitted generation runs,
including when the graph was fully expanded by then. Candidate ranking is capped at the
product of the bounds `max_recipes` and `max_nodes`; reaching the cap stops the expansion,
the completed generations are retained and the stop is reported. The extent per step sets
the resolution of the inventory trace, and a smaller extent needs more route steps. Catalyst
requirements and exact-equation enzyme references are stored with each route step; a
reference attaches to the exact reactant and product sets it was reported for. When the
supplied operators generate two stereoisomers of a product, both are retained as separate
species. A route returned by the capability is a hypothesis for a further analysis, for
example `reaction.estimate_thermodynamics`.

## Activity prediction across enzymes and reactions

`reaction.analyze_enzyme_panel` measures how well models trained on measured enzyme
activities under stated conditions predict held-out measurements when the held-out
enzymes, reactions or both are new. It compares five predictors: the training mean, the
nearest measured reference, and three kernel ridge regressions (a sequence-only model, a
reaction-only model and a joint model). The sequence kernel is the cosine similarity of
normalized amino-acid 3-mer counts. The reaction kernel uses chirality-aware radius-2
Morgan presence fingerprints of the selected reaction participants (512 bits per reaction
side). The joint kernel multiplies both similarities, and the nearest measured reference
is the one with the highest joint similarity. The request supplies the ridge penalty
`alpha`, which must be chosen without the held-out observations.

The split is declared before fitting. A sequence split separates exact sequences, and a
reaction split separates equations; observations that share a sequence, an enzyme form, a
reaction and a condition (replicates) are assigned to the same side of the split. A reaction
split treats an equation and a common positive multiple of it as the same equation. Homologs
and chemically similar substrates can cross these boundaries; the result reports the
similarities across the split, and a family-disjoint test requires partitioning outside the
capability.

Only measured finite values are used; inactive, failed, untested and unreported
observations are retained and counted, and are not converted to zeros. Errors are
computed only on observations with finite measured values, under the stated conditions. A
predictor shows a gain over a baseline (the training mean or the nearest measured
reference) when its held-out error is lower than that of the baseline.

Screen conditions can change together with the substrates, so on a held-out screen the
effect of the conditions is confounded with the effect of the substrates. A reaction-only
model gives every enzyme the same prediction on one reaction. The rank correlation of a
predictor is reported as unavailable when its predictions or the observed values are
constant, as for the reaction-only model on a held-out set of one reaction.
