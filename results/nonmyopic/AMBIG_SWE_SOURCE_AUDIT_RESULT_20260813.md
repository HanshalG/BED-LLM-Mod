# Ambig-SWE Source Audit Result

Date: 2026-08-13

Status: **source-population gate failed before task content; no model call or
endpoint is authorized**.

## Result

The official pinned release is clean and reproducible at commit
`ed58236332ad039b54f968145d7bed9ba988f262`, but its three published CSV views
do not contain the same task population:

| View | Rows | Sorted ID-set SHA-256 |
| --- | ---: | --- |
| fully specified | 500 | `33e18be7a9bd9f674790b63ed4d0b3fb17c176994802e3062b7d5a430a4e7d16` |
| underspecified | 500 | `33e18be7a9bd9f674790b63ed4d0b3fb17c176994802e3062b7d5a430a4e7d16` |
| interaction | 497 | `a48e8b617233ceb2854fa815eb81d407c098f97eb071a8ca011857e3267fe0c3` |

The fully specified and underspecified populations match exactly. The
interaction population omits three IDs. The frozen conjunction required the
same unique task IDs across all three views before mechanics task text could be
opened. That gate fails.

The 497-way intersection would still leave a large retained split, but taking
the intersection after observing the mismatch would amend the prospectively
frozen population rule. The audit therefore stops instead of silently changing
the cohort.

## Privacy Boundary

The audit read only CSV headers and the projected `instance_id` column. It did
not read or serialize task text, gold patches, executable tests, test outcomes,
saved policy responses, or full-versus-hidden missing-information values. All
downstream semantic, simulator, structural-opportunity, and endpoint gates
remain unmeasured.

## Decision

Close this exact release-derived Ambig-SWE route. Do not repair it by
intersecting the three views, changing the source commit, reducing split sizes,
or opening the three missing rows. A future upstream release with an explicitly
versioned aligned population may begin a new source protocol.

This is a source-release null, not evidence that Ambig-SWE lacks useful
non-myopic structure. Its three-turn simulator and executable SWE-Bench endpoint
remain conceptually strong, but they have not passed the dependency-valid
source boundary needed here.

## Provenance And Cost

- protocol SHA-256: `aaa7b29a5fc7bb7124ef7be0b7b6393fce7aa4b7977ed0504dab4fabd1970250`;
- audit script SHA-256: `a21bb3901da8d5577a2b4793287d4fece9490167b34aad34c379b0d529ea7348`;
- audit artifact SHA-256: `11e38b429229e48ffbe23b6b7333645fdfcb8a03514e6e240b0fb4b8fdcfc8b3`;
- focused test SHA-256: `3b14fbe652eac22a6ff63c60cc92cab5927ef093fe7592707fcda7b49fbbecaf`;
- focused tests: `3/3` passed;
- OpenRouter calls/cost: `0 / $0`;
- OATML, Slurm, or SSH use: `0`.
