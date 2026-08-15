# ChemBench Factored M-open Oracle Protocol

Date frozen: 2026-08-15 (Europe/London)

## Purpose

This is the zero-call dependency for the factored M-open planner proposed after
the seventh reading of Murphy's Model Discovery Agent. It asks whether the
already verified categorical ChemBench horizon opportunity survives three
MDA-like corrections:

1. support expands only after a calibrated prequential predictive failure;
2. proposals are represented as typed executable edits and cannot be retried
   after rejection or pruning;
3. support is contracted to a 12-model evidence- and family-diverse pool.

The experiment uses the already-opened outside-support v4 cohort. It opens no
new source response, LLM interface, policy endpoint, or v5 data. Its parameters
and gates are frozen before the revised transition is implemented or evaluated.

## Immutable Dependencies

- Official source repository/commit/tree:
  `scientific-discovery/LLM-AutoSciLab` /
  `acf160eb6c96897748dd92b152703b59b74efc05` /
  `e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a`.
- MDA seventh-pass decision:
  `results/nonmyopic/MDA_SEVENTH_PASS_FACTORED_MOPEN_PLANNER_20260815.md`,
  SHA256 `0f478d150d25cb20830e9e86f51bf49e4f5490de770437a3d4a4c2da1db38b5a`.
- Original policy-ladder v4 protocol:
  `results/nonmyopic/CHEMBENCH_POLICY_LADDER_MECHANICS_V4_PROTOCOL_20260814.md`,
  SHA256 `68221f5ec63d3b04c31a1926ee739723ee710058f144ab4ca90707c0ff08e3fa`.
- Original v4 result and transition bank:
  - result SHA256
    `f46e1fac28332b06ac2c62ff91fd68e537849c93ea74199ee7dc5d2d418aaec6`;
  - transition-bank SHA256
    `eb6c51b945c05f697b25681ed02a49763cb50ea234436b691a9d45a7e69b3b59`;
  - independent verification SHA256
    `534a2e2173f1919f3f0d20b4d0a921ca6408a52c844fec4e7c4b60a888a3f6e8`.
- The original v4 aggregate d1/d2/d3 terminal MSE is
  `.0391942507/.0320363611/.0292467959`. These opened values justify this
  architectural successor but cannot alter the frozen thresholds below.

## Cohort And Endpoint

- Nine primitive initial structures use source version `v2`.
- All 48 eligible outside structures and truth worlds use the already-opened
  source version `v4`.
- Slices and query seeds remain:
  - `easy/v4`: `2026081701`;
  - `medium/v4`: `2026081702`;
  - `hard/v4`: `2026081703`.
- The action set remains 18 fixed assays in six groups.
- Observations remain the frozen three-bin categorical likelihoods.
- Execution budget remains four experiments.
- Terminal loss remains expected held-out squared log-rate error on the 1,000
  public proxy queries per slice.
- Planning retains the uniform 48-world speculative oracle prior. The selected
  replay truth is never present in policy state, residual features, trigger,
  edit compiler, proposer, pruning, or action scoring.

## Factored Inference State

The inference state is immutable and contains only public/derived values:

```text
(action/outcome history,
 current live and reserve support,
 all model IDs ever tried,
 accepted typed-edit lineage,
 evidence weights and scalar outside mass,
 latest represented-support predictive probability,
 latest prequential surprise,
 whether expansion fired,
 explore/refine phase).
```

The state exposes a pool-wide residual report derived numerically from history:

- current canonical model name and evidence weight;
- cumulative categorical negative log likelihood;
- signed innovation `observed_bin - predicted_mean_bin`, averaged separately
  over each assay group;
- latest action group, outcome, represented-support predictive probability,
  and prequential surprise;
- tried and currently represented models;
- phase and remaining budget when the future prompt is rendered.

No continuous rate is reconstructed from a category. The report contains no
selected truth ID, v5 value, endpoint loss, policy preference, or oracle rank.

## Prequential Expansion Trigger

Expansion is controlled only by represented-support predictive surprise. For a
parent state and proposed action `a`, before any posterior update define

```text
p_rep(y | a, state) = sum_m w_m p(y | m, a)
surprise(y) = -log(max(p_rep(y | a, state), 1e-300)).
```

For each difficulty slice, calibrate a fixed threshold from the nine primitive
v2 models at the initial state:

1. enumerate all 9 primitive generating models, 18 actions, and 3 outcomes;
2. weight each tuple by a uniform model prior, a uniform action prior, and the
   generator's categorical outcome probability;
3. take the smallest surprise value whose cumulative weighted mass is at least
   `0.90`;
4. trigger expansion only when branch surprise is strictly greater than this
   threshold.

The strict inequality makes the source-calibration false-trigger rate at most
10%, including ties. The threshold uses no compound truth outcome. Outside mass
is retained for forecasting and phase diagnostics but does not independently
trigger expansion.

Every branch updates model likelihoods and support weights. A non-triggering
branch makes no proposer/cache request. A triggering branch requests at most
four edits and sets phase to `explore`; otherwise phase is `refine`.

## Typed Registry-Edit Ceiling

The zero-call proposer remains an oracle ceiling, but it must use the interface
required of the future LLM:

1. rank untried outside models by complete-history likelihood;
2. select at most four;
3. compile each candidate relative to the closest current parent into a typed
   registry patch;
4. validate the patch before the candidate can enter support.

The canonical mechanism signature is derived from frozen source metadata and
contains one core kinetic family plus zero or more modifiers. Legal operations
are:

- `add_factor`;
- `remove_factor`;
- `replace_factor`;
- `replace_core`.

A patch records parent/candidate IDs, operation, added and removed tags, and the
candidate's core family. Applying its tag difference to the parent must exactly
recover the candidate signature. Parent and candidate rows must be finite,
normalized, executable registry entries with unique names. Invalid, duplicate,
already represented, or previously tried candidates are rejected. Every valid
proposal is added to `tried` even if evidence contraction immediately removes
it, so exact rejected/pruned candidates cannot be proposed again.

## Evidence And Diversity Contraction

Total represented support is capped at 12 models after every update.

1. Rank all current and newly accepted candidates by complete-history log
   evidence, breaking ties by model ID.
2. Put the top eight into `live` evidence slots.
3. Fill up to four `reserve` slots from the remaining candidates, first taking
   the best candidate from each core family absent from `live`.
4. Fill any unused reserve slots by evidence rank.

All represented models, live and reserve, contribute to the submitted Bayes
forecast. The speculative 48-world planning prior remains separate and is never
pruned. The runner records every accepted edit, removed model, pool size, and
core-family count.

## Policy Construction

Retain the exact v4 conservative policy ladder. `d1` is the one-step structural
policy. For `k>1`, `d_k` evaluates one exact full-budget policy-improvement step
over `d_(k-1)`. Every state considers the highest speculative-variance assay in
each of the six groups, so the predecessor action remains available.

All policy levels use the same trigger, edit compiler, proposal cache, support
contraction, assays, speculative particles, and terminal loss. Run d3, d2, and
d1 into one immutable cache, then recreate d1 with empty planner memoization.
The replay must add zero proposal misses.

## Required Diagnostics And Controls

- original always-expand v4 oracle result, descriptive only;
- trigger/prune typed-oracle d1/d2/d3, primary;
- call-matched typed-oracle d1;
- fixed-support no-expansion control;
- union-only support diagnostic with the same trigger and proposals but no
  contraction, descriptive only;
- producer-independent replay from the immutable typed proposal/audit bank.

The future real-history-only MDA and LLM controls are not opened by this gate.

## Conjunctive Gate

All conditions must pass:

1. Exact source, predecessor artifact, protocol, privacy, finite/normalization,
   zero-call, and zero-cost checks pass.
2. Source-calibration false-trigger mass is at most `0.10 + 1e-12` on every
   slice.
3. Every accepted proposal has a valid round-tripping typed edit; no model is
   proposed twice along one history; support never exceeds 12.
4. Every slice has at least one triggering and one non-triggering unique
   transition, at least one accepted edit, and at least one contraction event.
5. Every full 12-model support contains at least three distinct core families.
6. Planned terminal risk equals uniform truth-conditional replay within
   `1e-10` for every slice and policy level.
7. Planned risk is non-increasing d1 to d2 to d3 on every slice within `1e-12`.
8. Aggregate d2 terminal MSE is at least 5% below d1.
9. Aggregate d3 terminal MSE is at least 5% below d2.
10. Each successive comparison has more practical paired truth-cell wins than
    losses at tolerance `1e-6`.
11. d2 changes the d1 root on at least one slice, and d3 changes the d2 root on
    at least one slice.
12. Call-matched d1 is exact with zero new producer misses.
13. Fixed support generates zero proposals, and producer-independent replay
    exactly reconstructs transition audits, roots, planned values, truth losses,
    aggregate comparisons, and every gate condition.

Failure closes this exact trigger/pruning formulation. Do not tune the surprise
quantile, inequality, pool size, evidence/diversity split, proposal count,
policy gates, or v4 cohort after reading the result. A pass authorizes only a
prospectively frozen, small LLM proposal-semantics and transition-fidelity gate;
it does not authorize v5 efficacy by itself.
