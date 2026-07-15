# Rock Diagnosis Confirmation Amendment: Quarantine Mechanics Smoke

Registered: 2026-07-15, before the amended confirmation execution.

## Event

After the original confirmation registration, a command-line mechanics smoke ran the
held-out `3-6` map for 32 trajectories at trial indices `0..31` with the registered
seed and a reduced bootstrap count. Its small-n decision flag was observed. This is a
protocol deviation from the intended first-read confirmation and those indices are
permanently excluded from evidence.

## Unchanged Elements

The map, start `(0, 3)`, source model, action space, EIG recurrence, policy arms,
candidate-cell accounting, K values `{2, 3, 4}`, horizon 8, seed 2304, primary metric,
bootstrap procedure, pass rule, and no-LLM/no-spend status are unchanged.

## Amended Held-Out Run

The formal confirmation now evaluates exactly 2,000 previously unseen trajectory
indices `32..2031` by setting `trial_offset=32`. Each index has independently keyed
truth and observation randomness, so it has no shared target, sensor uniform, or
candidate-cell schedule with the quarantined smoke trajectories. It will use the
original 10,000 bootstrap resamples and the unchanged decision rule.

The smoke is logged in `EXPERIMENTS.md` and retained only as a mechanics check. This
amendment is written before the `32..2031` result is generated or viewed.
