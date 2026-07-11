# INVALID-ENDPOINT Source Runs

The following run families are quarantined. Their raw directories remain under
`runs/` for diagnosis, but none may be cited as policy evidence:

- every Paprika Step 1 scaffolded run through
  `20260711T014924_paprika-step1-r5-scaffolded-repair5-seed1304`;
- canonical scaffold shards for offsets 0-9, run IDs `20260711T033611` through
  `20260711T042609`, and their combined directory;
- every naive control used by the invalid Step 1 comparison, including
  `20260711T010517` and `20260711T012829`;
- the partial generation-thinking rescue `20260711T052319`;
- generation-thinking rescue shards for offsets 0-9, run IDs
  `20260711T071533` through `20260711T075750`, and their combined directory.

The exact launch history and costs remain in `EXPERIMENTS.md`. The tracked
combined outputs in this directory carry an explicit `INVALID-ENDPOINT` label.
