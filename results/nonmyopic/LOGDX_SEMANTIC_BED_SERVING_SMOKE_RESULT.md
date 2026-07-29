# LogDx-CI Semantic BED Serving Smoke Result

Date: 2026-07-29

Status: **semantic gate failure; exact interface closed**.

Public result:
`results/nonmyopic/logdx_semantic_bed_serving_smoke/logdx-semantic-serving-20260729T012337Z/SERVING.json`.
SHA-256:
`bdbb3e51ffa9573006018830b63701fb3bee165944ca3d2573aea4dc19102b79`.

Private raw-response SHA-256:
`b342474096de97c32f556f5f4be6ea7613bc46f47b452241ef6a430fa97119ea`.

## Execution

The frozen smoke completed exactly five GPT-5.4 Mini planner calls and five
Gemini 2.5 Flash updater calls:

- accepted requests / HTTP attempts: `10 / 10`;
- retries and provider-error retries: `0 / 0`;
- reasoning tokens and forced exits: `0 / 0`;
- prompt / completion tokens: `32,887 / 3,722`; and
- reported cost: `$0.02608480`.

All ten responses parsed exactly. Every planner returned six unique semantic
hypotheses and four unique valid regex actions. All 15 updater follow-up
actions parsed and executed without tool error. No ground truth, diagnosis
evaluator, policy endpoint, or confirmation case was accessed.

## Frozen Gates

Three semantic gates failed:

| Gate | Result | Required |
|---|---:|---:|
| Q1 probe has a raw-log match | `3 / 5` cases | at least `4 / 5` |
| Likelihood range at least 20 with at least 3 values | `3 / 5` cases | at least `4 / 5` |
| Corrected observation-dependent follow-up | `3 / 5` cases | at least `4 / 5` |

Dependency-type diversity passed with four types: file/path, line number,
other literal, and test/symbol.

## Case Pattern

| Case | Q1 match | Likelihood range | Distinct values | Dependent follow-ups |
|---|---:|---:|---:|---:|
| `pip-pytest-network-github-v2-001` | yes | 75 | 5 | 3 |
| `mypy-pandas-001` | yes | 50 | 6 | 3 |
| `pytest-pandas-001` | no | 0 | 1 | 0 |
| `cargo-tokio-001` | no | 0 | 1 | 0 |
| `pnpm-jest-config-v2-001` | yes | 30 | 6 | 6 |

The three successes form the full intersection: when Q1 retrieved evidence,
the updater produced differentiated likelihoods and observation-dependent
follow-ups. In the two failures, the first query returned no evidence, the
updater assigned a flat likelihood vector, and no follow-up used newly learned
raw-log content. This localizes the failure to first-probe recall and its
downstream information bottleneck rather than response transport or tool
execution.

## Decision

The exact LogDx hypothesis/probe/likelihood interface is closed. There is no
prompt repair, model substitution, case removal, threshold change, or rerun.
The preregistered first-link ranking and policy stages are not authorized.

The independent source result remains positive: released multi-step LogDx
agents use observation-dependent tool chains and outperform the corresponding
single-shot debugger. This serving result says only that the tested BED
factorization did not reproduce that capability reliably enough to support a
controlled policy experiment.

Authenticated OpenRouter credits after the run report `$180.00` credited and
`$161.804313206` used, leaving `$18.195686794`.

OpenRouter only. OatML, Slurm, SSH, and cluster use: `0`.
