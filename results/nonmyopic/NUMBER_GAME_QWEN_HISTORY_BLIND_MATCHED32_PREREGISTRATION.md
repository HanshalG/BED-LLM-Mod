# Number Game Qwen History-Blind Matched-32 Preregistration

Date frozen: 2026-07-30

## Question

Does conditioning Qwen's hypothesis-generation prompt on simulated answer
history improve the belief states used by non-myopic Number Game planning?

The completed zero-call support-quality audit could not identify this effect:
pooling stored branch-conditioned outputs and filtering them afterward exactly
reconstructed routed support, because parsing had already enforced each
branch's observations. This fresh control removes history from the generation
prompt itself while matching the model, draw count, branch count, support
update, source trees, and canonical evaluation.

This is a mechanism study, not a new policy-efficacy trial. It cannot rescue,
relabel, or replace the source study's mechanics-qualified `gated_null`.

## Frozen Source Cohort

- source `RESULT.json` SHA256:
  `04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e`
- source `TREES.json` SHA256:
  `8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82`
- source `TARGETS.json` SHA256:
  `2fd09b75bcce734e17f237ced9a49e97e13e290e2f5229b9354ad7a8a1bdafd6`
- source tree indices: `0..31`
- source tree seeds: `80000..80031`
- candidate roots per tree: `8`
- canonical targets: all `33` unique `TARGETS.json` extensions, uniformly
  weighted

The runner must fail before requests if any source hash, tree count, tree
index, tree seed, candidate-root set, selected-root record, stored dynamic
branch, or canonical target differs.

## Fresh Control Generation

For each source tree, enumerate exactly:

- `16` first-stage branches: eight stored candidate roots times both answers;
- `32` second-stage branches: the stored dynamic second query under each
  first-stage branch times both second answers.

For each of these `48` branch slots, generate one pooled support from two
independent Qwen draws. Every draw uses:

- model: `qwen/qwen3.7-plus`;
- reasoning: disabled;
- temperature and strict 24-item schema identical to the source planner;
- prompt: the exact initial Number Game prompt with **no observations**;
- semantic repair: disabled.

Control seed schedule:

`9000000 + 1000 * tree_index + 2 * history_index + draw_index`

where `history_index` follows source root order, first stages before second
stages, labels in `False, True` order, and `draw_index` is `0` or `1`.

The conditional arm is the already stored source dynamic support and makes no
new calls.

## Matched Support Update

Every support is deduplicated by complete extension.

For branch history `h1 = (r, y1)`:

- conditional: stored dynamic retained-rejuvenation first support;
- history-blind: filtered source initial support union the fresh pooled
  no-observation support assigned to that branch, then filtered by `h1`.

For `h2 = (r, y1, q2, y2)`:

- conditional: stored dynamic retained-rejuvenation second support;
- history-blind: the history-blind first support filtered by `(q2, y2)` union
  the fresh pooled no-observation support assigned to the second-stage branch,
  then deduplicated.

Thus both arms receive two fresh candidate draws at each simulated refresh.
The control differs only in whether those draws saw the branch history in
their prompt. Empty or thin post-filtered control supports are scientific
outcomes, not parser failures.

## Frozen Endpoints

Replay all 33 canonical targets through all eight roots on every tree, holding
the stored dynamic-policy query history fixed.

For first and second stages, report:

- posterior-predictive MSE against the exact canonical posterior;
- exact truth-extension coverage;
- support size;
- conditional-minus-history-blind paired tree-bootstrap intervals.

At each root, define prompt-conditioning benefit as:

`history-blind predictive MSE - conditional predictive MSE`.

On source trees where dynamic and fixed support selected different roots,
report:

- prompt benefit at the dynamic-selected root minus prompt benefit at the
  fixed-selected root;
- its tree-bootstrap interval;
- Spearman correlation between that contrast and the source tree's
  exact-canonical realized Brier advantage;
- a bootstrap interval for that correlation.

The correlation is explanatory and cannot override the frozen gates.

## Frozen Scientific Gates

Prompt conditioning is called directionally coherent only if all conditions
hold:

1. at least `20` of the 32 source trees have different dynamic and fixed roots;
2. second-stage conditional-minus-history-blind predictive MSE has a 95%
   interval below zero;
3. second-stage conditional-minus-history-blind truth coverage has a 95%
   interval whose lower endpoint is at least zero; and
4. on changed-root trees, mean dynamic-root-minus-fixed-root prompt benefit has
   a 95% interval above zero.

First-stage metrics, support size, and benefit-to-realized correlation are
secondary diagnostics.

## Serving Smoke

Before scientific calls, run one fresh exact-10 smoke:

- five pooled no-observation supports;
- seeds `8900000..8900009`;
- exact 10 accepted and HTTP requests;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all 10 draws strict JSON with at least 16 valid unique extensions;
- every pooled support has at least 24 unique extensions;
- every second draw adds at least two extensions;
- cost at most `$0.12`.

The smoke has no efficacy endpoint. Any failed gate closes this route and the
formal seeds remain unopened.

## Formal Mechanics And Budget

- fresh trees: `32`;
- branch slots per tree: `48`;
- pooled draws per slot: `2`;
- exact accepted requests: `3072`;
- target or validation generation requests: `0`;
- concurrency: `256`;
- provider retries: at most `32`;
- all accepted draws must parse as strict JSON;
- every raw draw must contain at least `16` valid unique extensions;
- every pooled draw must contain at least `24` unique extensions;
- every second draw must add at least `2` extensions;
- reasoning tokens and forced exits: `0`;
- run cost cap: `$4.25`;
- minimum starting balance: `$5.00`;
- bootstrap seed/samples: `9100000` / `20000`;
- resampling unit: one complete tree;
- no partial scoring or reuse after failure.

The runner checks live authenticated credit before opening formal seeds and
reconciles accepted requests, HTTP attempts, retries, parser diagnostics, and
cost after completion.

## Outputs

Public:

- source bindings, protocol, mechanics, usage, gates, aggregate endpoints, and
  per-tree/per-root metrics;
- parsed supports as extension hashes only where needed for reproducibility.

Private and uncommitted:

- raw provider responses.

Formal outputs will be written under:

`results/nonmyopic/number_game_qwen_history_blind_matched32/`
