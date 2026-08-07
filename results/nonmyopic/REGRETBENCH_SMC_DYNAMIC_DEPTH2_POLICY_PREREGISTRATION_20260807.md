# RegretBench SMC Dynamic Depth-Two Policy Preregistration

Date frozen: 2026-08-07, before any RegretBench model response.

Status: sealed conditional policy protocol; no policy endpoint is open.

## Claim

Test whether depth-two planning over an LLM's own path-dependent semantic
particle updates improves two-question intent identification. The headline
comparison is `smc_dynamic_depth2` versus `smc_myopic_refresh_brier`: both use
the exact same generated one-step branches and terminal Brier utility, but the
first chooses the root by expected two-question risk while the second chooses
it by immediate post-refresh risk.

The LLM is load-bearing. It predicts semantic reply likelihoods, retains or
revises natural-language hypotheses after each answer, proposes future
questions, and predicts their replies. The hidden RegretBench CIG is used only
by the environment and evaluator and is never supplied to the planner.

## Authorization Boundary

This policy is authorized only after all of the following:

1. the primary Aug 8 support-recovery development result is a literal,
   independently verified `gated_null` with every mechanics gate passing;
2. the sealed SMC support-recovery contingency then produces a literal
   `passed` result with exact independent replay;
3. no primary policy, primary confirmation, SMC policy, or SMC confirmation
   artifact has been opened; and
4. the policy runs on a later Europe/London budget day through a separately
   frozen dated executor.

A primary support pass, either mechanics failure, partial artifact, missing or
failed verification, SMC support null, or any opened descendant authorizes
nothing. A support pass authorizes only this development policy, not a paper
claim or confirmation.

Bound predecessors:

- SMC support protocol:
  `b8eafe438c21e4793a59bd24a0991d20ecf4f53efa5a19fe21dc4eed7c22f807`;
- SMC support core:
  `91be9699391aa67070174ca3be2fdb7b6cd9e1ae210fcbb0f5c7e35ba601f280`;
- SMC support verifier:
  `fab0b932c1c00ccb2a7333f63394e16b930cd238c3d5f1da73c8692992fa1e2a`;
- SMC support daily executor:
  `18805dbd9f79e002fd6eaa6df9aa2c8da1afdc76cd360b0f9f26e7b2f64d7c84`;
- development split:
  `29d33b2fda0be7cc4eea6f4d9d4fe74fe200b5c580632dcb7ba844c3b825af69`.

## Model And Information Boundary

- Semantic particle model: `deepseek/deepseek-v4-flash-0731`.
- Reasoning: disabled and excluded for every BED and environment role.
- Temperature: `0.7`; maximum output: `2,400` tokens.
- Large planning concurrency: at most `128`.
- No scientific retry, repair, continuation, task replacement, or schema
  coercion.

A separately labelled `naive_thinking` baseline uses `openai/gpt-5.6-luna`
with medium reasoning. It sees only prompt and realized dialogue, never
particles, candidates, scores, hidden CIG fields, or endpoints. Its reasoning
is excluded from all BED policies and its result is descriptive only.

Public model payloads contain only opaque task ID, ambiguous prompt, visible
dialogue, model-generated parent particles, four model-generated questions,
and parent-provenance hashes. Hidden intents, aliases, slots, facets,
reference questions, truth indexes, benchmark probabilities, scores, and
policy labels are forbidden.

## Initial Parent Annotation

For each task, reuse the exact eight raw initial particle slots and four root
questions from the verified primary support run. Repeated raw slots remain
distinct indexed particles. No initial hypothesis or question is regenerated.

One strict annotation call receives the indexed particles and fixed questions.
It returns exactly one record for every `parent_index` in the permutation
`0..7`, each containing exactly four nonempty predicted replies aligned to the
fixed questions. Interpretation, final answer, and weight are copied locally
from the verified parent and cannot be changed by this call. This constructs
the initial enriched support while preserving the banked parent population
exactly.

The four reply partitions define immediate EIG and fixed-support controls.
Duplicates remain valid separate particles; probabilities are normalized over
all eight slots.

## SMC Transition Interface

Every simulated or realized answer transition receives the complete current
eight-particle support and visible history. It returns exactly:

- eight unique child `(interpretation, final_answer)` pairs;
- one unique `parent_index` for every index `0..7`;
- `revision_type` equal to `retained` or `revised`;
- a nonnegative child weight and four predicted replies for each child; and
- four distinct single-dimension clarification questions aligned to those
  reply vectors.

`retained` requires the normalized interpretation and final answer to equal
the indexed parent exactly. `revised` requires at least one to differ. Every
transition must retain between two and six parents inclusive. Predicted replies
may change for retained particles because the child questions are newly
generated. Child weights are renormalized after strict parsing. Duplicate
children, invalid lineage, false labels, unchanged revisions, malformed
questions or replies, or extra fields fail mechanics rather than being
repaired.

For the realized second answer, the exact generated reply partition updates
the first-transition particle weights before they become parents of the final
SMC transition. If no generated reply matches, the primary terminal mass is
zero and the unchanged first-transition weights are passed to the final
descriptive transition with the full visible dialogue. This prevents an
undefined posterior from silently dropping the trajectory.

## Dynamic Tree

For each of 64 development tasks, every root, every one of the eight initial
particle slots as simulated truth, and two independent draws receives one
conditioned and one history-blind SMC transition:

```text
4 roots * 8 particle slots * 2 draws * 2 arms = 128 branches per task
64 initial annotations + 64 * 128 branches = 8,256 planning calls
```

The conditioned arm appends the root question and that particle's aligned
predicted reply. The history-blind arm receives the identical parent support
without the simulated dialogue. Each conditioned/blind pair shares a seed;
all four roots share the seed for a fixed task, particle, and draw. Calls are
adjacent within each pair.

For each child support and simulated truth answer:

1. match the truth answer to child final answers with the frozen conservative
   lexical matcher;
2. choose the child question with maximum EIG;
3. marginalize over its generated reply partition within the matched truth
   group; and
4. calculate expected terminal truth-group Brier and log loss.

No child match receives Brier `1` and log loss at floor `1e-12`. A root's risk
is the initial-weighted mean over all eight particle slots and both draws.

## Policies And Controls

All policies share the exact parent population, annotations, branch tree,
hidden truth, and realized-history transitions.

- `smc_dynamic_depth2`: minimum conditioned expected terminal Brier.
- `smc_myopic_refresh_brier`: minimum immediate post-transition truth-group
  Brier on the same conditioned branches.
- `smc_myopic_brier`: minimum immediate Brier on the fixed annotated parents.
- `smc_history_blind_depth2`: the dynamic scorer on paired no-history SMC
  transitions.
- `smc_myopic_width`: maximum immediate EIG on annotated parents.
- `smc_fixed_depth2`: exact two-step Brier on annotated parents without SMC
  transitions.
- `random`: uniform root from a frozen local seed.
- `naive_thinking`: Luna medium reasoning directly asks two questions from
  visible dialogue; descriptive, unmatched-compute, and non-gating.

Ties use the lowest original root index. The headline comparator is
`smc_myopic_refresh_brier`, which matches both generated transition quality and
one-step utility. Fixed-parent Brier and entropy-EIG controls remain mandatory
to expose alternate notions of myopia. History-blind isolates sampling and
parent-continuity effects from answer conditioning.

## Realized Execution And Endpoints

The hidden intent is sampled uniformly by frozen local seed only after all
planning responses and root selections are frozen. Each distinct selected root
is executed once and shared across policies selecting it:

1. map the generated root with the official RegretBench mapper;
2. obtain its exact true slot value or the fixed unsupported reply;
3. make one conditioned SMC transition from annotated parents;
4. select its maximum-EIG second question without hidden information;
5. map and answer that question exactly;
6. compute aligned terminal truth mass from its generated reply partition; and
7. update parent weights and make one final SMC transition from the complete
   visible history as a secondary robustness endpoint.

The primary endpoint is aligned terminal Brier after question two. Hypotheses
whose predicted second reply exactly matches the normalized environment reply
receive likelihood one and all others zero. Truth mass is the normalized mass
of matching answer aliases within that outcome group. No represented reply or
no truth match gives mass zero. Report aligned log loss, first-step mass,
supported-action rates, and fresh final-transition mass/Brier/log/coverage as
secondary endpoints.

Report paired task means, sample standard deviations, wins/ties/losses, 20,000
paired-bootstrap intervals and improvement probabilities, root disagreement,
and predicted-to-realized Spearman diagnostics. Two branch draws are replayed
separately as a non-gating draw-stability diagnostic and cannot rescue or
reclassify the result.

## Serving Smokes

Before development, run an exact-10 SMC enriched smoke on the four primary
mechanics parents: four annotation calls plus one conditioned/blind transition
pair for each of the first three tasks. It uses exact official replies. Passage
requires exact accepted requests and attempts, zero retries/provider retries/
reasoning/forced exits, strict annotation and child lineages, two through six
retained children, four aligned questions/replies, at least two informative
initial roots per task, an informative follow-up in every child, all three
first and second actions officially supported, every exact second reply
represented, privacy/provenance audits passing, and cost at most `$0.20`.
No truth-mass or policy efficacy enters smoke passage.

After that, run a separate exact-10 Luna medium-reasoning smoke matching the
existing RegretBench naive protocol. Failure disables only the descriptive
baseline and cannot stop, pass, rescue, or veto the DeepSeek policy.

## Frozen Seeds And Request Counts

- enriched smoke annotations: `202608290000 + task`;
- enriched smoke branches: `202608291000 + task`;
- naive smoke: ranges beginning `202608292000`;
- development annotations: `202608300000 + task`;
- matched branch seed:
  `202608310000 + task*16 + particle*2 + draw`, reused across roots;
- hidden truth: `202608320000 + task`;
- random root: `202608330000 + task`;
- realized first transition: `202608340000 + task`, reused across roots;
- realized final transition: `202608350000 + task`, reused across roots;
- bootstrap: `202608360000`;
- naive first/second: `202608370000 + task`, `202608380000 + task`;
- naive first/final endpoint support: `202608390000 + task`,
  `202608400000 + task`.

Primary planning is exact `8,256` calls. Primary realized execution is one
first and one final transition per distinct selected root, at most `512` calls.
An available naive baseline adds `128` Luna and `128` DeepSeek endpoint calls.
Maximum development total is `9,024`: at most `8,896` DeepSeek and `128` Luna.
All component counts and costs are reported separately.

## Mechanics Gates

All must pass:

- exact verified predecessor, source, split, protocol, seed, and code bindings;
- exact request/attempt accounting and zero retries, provider retries,
  DeepSeek reasoning tokens, and forced exits;
- every annotation has exact parent permutation and four aligned replies;
- every SMC child has exact lineage, eight unique particles, four aligned reply
  vectors, and two through six retained parents;
- every task has at least two informative initial roots and at least 90% of
  simulated children have an informative follow-up;
- each primary policy has at least 48 supported first and 40 supported second
  actions, with at least 40 exact second replies represented;
- every model payload privacy and parent-provenance audit passes;
- conditioned/blind adjacency and all common-random-number schedules are exact;
- public artifacts exclude raw questions, replies, particles, aliases, facets,
  truth indexes, and raw model responses; and
- combined development spend is at most `$3.50`.

Naive availability and diagnostics do not enter primary mechanics or status.

## Scientific Gates

All are conjunctive and use aligned terminal Brier unless stated otherwise:

1. dynamic differs from each refresh-myopic, fixed-parent myopic, and EIG
   myopic on at least `16/64` tasks, and from blind and fixed-depth-two on at
   least `12/64`;
2. conditioned predicted gain over each myopic root is at least `0.01`;
3. dynamic minus each of refresh-myopic, fixed-parent myopic, and EIG myopic is
   at most `-0.02`, bootstrap improvement probability at least `0.90`, and
   wins exceed losses;
4. dynamic minus history-blind is at most `-0.015`, probability at least
   `0.80`, and wins exceed losses;
5. dynamic minus fixed-depth-two is at most `-0.01`, probability at least
   `0.80`, and wins exceed losses;
6. dynamic mean log loss is no worse than every comparator; and
7. on each changed-root myopic comparison, predicted advantage has Spearman
   correlation at least `0.15` with realized advantage and bootstrap
   probability of positive correlation at least `0.80`.

Failure closes this interface once. A pass authorizes only a separately frozen
confirmation on the untouched confirmation split. No threshold, draw count,
matcher, seed, model, policy, endpoint, prompt, or favorable subset may change.

## Daily Budget

The enriched smoke cap is `$0.20`, naive smoke cap `$0.20`, and development cap
`$3.50`, for a same-day worst case of `$3.90` under the hard account-wide
`$5.00` Europe/London cap. Each stage reserves its full cap immediately before
dispatch and spend is reconciled as the maximum of posted account use and
locally measured accepted-request cost. Unspent allowance never rolls over.

Freezing and implementing this protocol makes zero model calls and opens no
support, policy, confirmation, or reporting endpoint.
