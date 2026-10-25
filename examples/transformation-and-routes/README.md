# Transformation and routes

**Question.** Is the hydrolysis of methyl acetate thermodynamically favorable under stated
conditions, and which routes do a supplied reaction operator (a SMARTS transformation) and
the supplied feedstocks give?

The campaign has 3 steps for the hydrolysis of methyl acetate:

1. `reaction.inspect` balances the equation and reads the bond changes from the atom
   mapping given in the campaign.
2. `reaction.explore_routes` applies the reaction operator for ester hydrolysis to the
   listed feedstock compounds (methyl acetate and water) and reports the routes that
   produce the target compound (acetic acid).
3. `reaction.estimate_thermodynamics` estimates the standard transformed Gibbs energy
   ΔG°′ and the Gibbs energy ΔG′ for methyl acetate + water -> acetate + methanol at the
   stated activities (1 mM for each solute relative to the 1 M standard state, 1 for
   water), pH, pMg, ionic strength and temperature, with a model uncertainty. Steps 1 and 2
   use neutral acetic acid; step 3 uses the compound acetate (`kegg:C00033`).

```bash
alambiq run examples/transformation-and-routes/campaign.yaml --report report.html
```

**Needs.** The `chemistry`, `routes` and `thermodynamics` extras.

**What to look at.** The equation balances and the atom mapping is complete for heavy atoms
(by atom map number, the bond between atoms 2 and 4 breaks and the bond between atoms 2 and
6 forms). One route produces acetic acid in one step. At pH 7.5, pMg 3, 0.1 M ionic
strength, 298.15 K and the stated activities the estimate is ΔG°′ of -19.9 kJ/mol and ΔG′
of -37.1 kJ/mol with a model standard deviation of 3.7 kJ/mol.

**Limits.** A negative ΔG′ is a necessary condition for the reaction to proceed; the rate is
set by kinetics. The route search returns the routes that the supplied operators generate
from the supplied feedstocks. The energy comes from the eQuilibrator component-contribution
model. Its standard deviation is a statistic of the model, conditional on the stated
conditions and activities.

**Data.** The reaction was written for the example. Thermodynamic data come from the
eQuilibrator packages named in the model section of the report.
