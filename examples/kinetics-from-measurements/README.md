# Kinetics from measurements

**Question.** What Km and Vmax does a Michaelis-Menten fit return for this table of
initial rates?

`assay.inspect` reads a table of initial rates at 10 substrate concentrations. `assay.fit`
fits a Michaelis-Menten curve and reports the standard error of each constant.

```bash
alambiq run examples/kinetics-from-measurements/campaign.yaml --report report.html
```

**Needs.** The `quantitative` extra.

**What to look at.** The report shows the readings, the fitted curve through them and a
table of Vmax and Km with their standard errors and units. The `assay.inspect` result lists
the number of readings by status (measured, failed, missing, censored).

The rates are synthetic. They are generated from Vmax = 180 µM/min and Km = 2.5 mM with no
noise, so the fit returns those values. The [getting-started guide](../../docs/getting-started.md)
runs this example first.

**Limits.** Without noise the standard errors are of the order of 1e-8 mM for Km and
1e-7 µM/min for Vmax and apply to these noise-free rates. A Michaelis-Menten fit describes
initial rates at one enzyme concentration and one set of conditions. kcat requires the
enzyme concentration, which is not an input of this campaign. The model has no inhibition
term. See [the measurements page](../../docs/methods/measurements.md). To fit another
enzyme, replace `rates.csv` with a table of the same layout and describe its conditions in
`campaign.yaml`.
