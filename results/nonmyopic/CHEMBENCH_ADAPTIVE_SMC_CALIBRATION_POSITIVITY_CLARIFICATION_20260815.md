# ChemBench Adaptive-SMC Calibration Positivity Clarification

Date: 2026-08-15 (Europe/London)

This implementation clarification is frozen before any calibration response or
result is generated.

The source's identity-coordinate parameters (`alpha`, `beta`, `n`, `n_inh`,
`n_met`, `pKa`, `pKa1`, and `pKa2`) are all physically positive in v0-v3. The
existing broadened-prior implementation clips expanded identity-coordinate
draws to positive machine epsilon. The adaptive-SMC prior must do the same:

```text
lower = max(observed_low - padding, machine_epsilon)
```

for identity-coordinate parameters. This prevents negative Hill exponents,
modifier powers, and pKa values while preserving every other frozen prior rule.
It is source-derived and does not inspect v4 or any generated observation.

No seed, cohort, assay, query, method, threshold, or gate changes. No source
response, benchmark endpoint, model call, API call, or cost occurred before
this clarification.
