# Cofactor regeneration

**Question.** How does regenerating the cofactor change the amount of product formed?

A 1 mL batch contains 50 mM acetoin, 100 mM formate and 1 mM NADH. A butanediol
dehydrogenase (BDH) reduces acetoin to 2,3-butanediol and converts NADH to NAD; a formate
dehydrogenase (FDH) converts NAD back to NADH; NADH is also lost by decay. The two
operation definitions differ only in the id, the FDH catalyst, the channel `fdh` (the FDH
reaction) and the listing of `fdh` among the active channels. `operation.simulate`
integrates both definitions over 6 h and `operation.summarize` reports the net amount of
butanediol formed.

```bash
alambiq run examples/cofactor-regeneration/campaign.yaml --report report.html
```

**Needs.** The `quantitative` extra.

**What to look at.** Without regeneration the product cannot exceed the NADH supplied: 0.998
µmol of butanediol (0.998 mM) forms and the NADH is consumed. With regeneration 32.7 µmol
(32.7 mM) forms in the same 6 h, 32.8 times as much (32.7 µmol against 0.998 µmol). The same
1 mM of cofactor is present in both runs. The FDH reaction converts NAD back to NADH, so the
net butanediol formed (32.7 mM) exceeds the cofactor supplied (1 mM). The pool
`nicotinamide` (NADH, NAD and the lost fraction NADH_lost) sums to 1 mM at every recorded
time.

**Limits.** Every maximum rate, saturation scale and rate constant is an assumed value of
the example, so the numbers demonstrate the mass balance under these values. The BDH channel
has acetoin and NADH as rate factors; butanediol and NAD do not enter its rate. With
regeneration, the acetoin concentration decreases from 50 mM to 17.3 mM and the formate
concentration from 100 mM to 68.2 mM over 6 h; the rate laws are saturation functions with
scales of 5 mM (acetoin) and 10 mM (formate). Predictions for a real system require measured
kinetics, obtained through `assay.fit` and `operation.define` (see
[connected-investigation](../connected-investigation/README.md)).
