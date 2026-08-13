# RevengeBench source-opportunity result

Date: 2026-08-13

Status: **execution opportunity required**

OpenRouter calls/cost: **0 / $0**

## Result

The audit opened only the 12 frozen opportunity-target entrypoints after the
source protocol was sealed. All 12 are parseable and 11/12 satisfy the frozen
static nontriviality contract. The 11 targets span all four replayable arenas.

The source prerequisite passes, but the structural opportunity does not pass or
fail statically. RevengeBench probes are executable opponent policies: a probe's
observation is a dynamic full-game trajectory, not a source-local response that
can be derived from branch counts. Consequently, static inspection cannot prove:

- intervention-sensitive answer partitions;
- a changed first probe between exact depth two and compute-matched receding
  myopic selection;
- positive non-myopic information gain; or
- separation on sealed endpoint seeds.

The fail-closed decision is therefore
`freeze_zero_call_execution_opportunity_audit`. It is not a horizon-positive
result, and source complexity is not used as a proxy for information value.

## Frozen successor

Before reading selected probe content or trajectories, the successor protocol
and manifest freeze three hidden hypotheses, three probe policies, three paired
probe seeds, three endpoint seeds, and likelihood temperatures
`{0.5, 1.0, 2.0}` for each of BattleSnake, Halite, HuskyBench, and RoboCode.

The execution gate must use native arena action parsers and exact public
simulation mechanics. It requires robust changed first actions and positive
depth-two information gain against a call-matched receding-myopic controller,
plus endpoint separation. It makes zero model calls and opens no released prior
outcome.

Source machine result: `revengebench_source_opportunity/AUDIT.json`

Source-result SHA-256: `bf89e1ccf3060a0d45ba4df86deb19214cf66f751e255b02665f67eaaeb48c62`

Execution protocol SHA-256: `cec27cf5bb2e697979243a3a4cb1617aac23a4fe6cecad5b537d1926076c3401`

Execution manifest SHA-256: `642e17e0f784ca40fddb13bfd15cadbe781fbff4ccc1e9c6d138865473975696`
