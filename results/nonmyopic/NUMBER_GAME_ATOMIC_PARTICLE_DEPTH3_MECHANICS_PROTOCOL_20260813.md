# Number Game Atomic-Particle Depth-Three Mechanics Protocol

Date frozen: 2026-08-13, after terminal closure of the overgenerated
factorized interface and before any response under this distinct interface.

## Scientific Question

Can independent one-hypothesis LLM samples form a stable, path-dependent
particle belief whose non-myopic depth-three ranking predicts realized
canonical posterior quality better than call-matched myopic, history-blind,
fixed-support, and random controls?

This is a mechanics stage, not a policy headline. A pass authorizes only a
separately frozen development protocol with fresh trees, sealed independently
generated endpoints, paired common random numbers, and powered uncertainty.
Failure closes this exact model, interface, seeds, and mechanics cohort.

The design responds to the observed generation--evaluation gap in Number Game
LLMs and to BED-LLM's evidence that targeted proposal generation is important.
It also addresses this project's direct failures: large responses coupled
diversity, syntax, observation obedience, and serialization into one fragile
event. Here every request is one particle. Code evaluates executable semantics;
the LLM alone proposes the open-ended rule distribution.

## Irreducible LLM Role

The LLM samples one human-plausible executable concept conditioned on the
current query/answer history. Code may parse, execute, filter, resample, and
score these particles, but may not generate, repair, translate, mutate, or
complete a rule. Duplicate extensions remain duplicate particles and retain
their empirical frequency; they are not deduplicated into a uniform support.

The frozen executable DSL is the existing Number Game DSL: boolean expressions
over `n` in `0..100`, bounded integer arithmetic and comparisons, and only
`divisible`, `is_square`, `is_power_of_two`, `is_prime`, `digit_sum`, and
`ends_with`. Explicit member/exception lists and constant concepts are invalid.
Every accepted particle must execute on all 101 integers and obey its supplied
history exactly.

Novelty is evaluated against the already frozen 416,366-extension classical
grammar bank (SHA-256
`6e2a523d7d7c0c59b5df0494de37087525c7009d43a5eaba01390e005d85e0e0`).
This bank is a yardstick only: it never enters prompts, selection, filtering,
weights, or endpoint scoring.

## Frozen Atomic Calls

- Exact model: `qwen/qwen3.7-plus`, nonreasoning.
- Temperature `.85`; strict JSON object with exactly `name` and `expression`.
- Maximum output 220 tokens/request; no forced finalization or format repair.
- Seeds `202608136000..202608142399` in canonical request-slot order.
- Four fresh mechanics trees, local seeds `202608143000..202608143003`.
- Eight deterministic diversity cues cycle by request slot: divisibility,
  digit structure, bounded interval, prime/square, affine transform, modular
  structure, boolean composition, and sequence structure.
- Exact calls:
  - 256 initial particles: 64/tree;
  - 1,024 first-refresh conditioned particles: 32 for each of four roots,
    two simulated answers, and four trees;
  - 1,024 first-refresh no-observation particles: 256/tree;
  - 2,048 second-refresh conditioned particles: 32 for each tree/root/first
    answer/second answer;
  - 2,048 second-refresh no-observation particles: 512/tree.
- Total: exactly 6,400 accepted requests. Zero retries. Calls are checkpointed
  in fixed blocks no larger than 256; every block is durably written before the
  next is authorized.

The first- and second-refresh blind pools receive the exact initial prompt and
no observations. They use the same call counts, model, schema, temperature,
cue schedule, and fresh seed allocation as their conditioned counterparts.
They are generated independently and are not pooled counterfactual conditioned
responses.

## Particle Construction

The initial 64 slots must retain at least 48 valid particles and 24 unique
extensions per tree. Deterministic systematic resampling with the tree seed
restores exactly 64 particles; multiplicities are preserved.

For each conditioned 32-slot refresh, at least 24 particles and 16 unique
extensions must be valid and history-consistent. Resample its valid multiset to
32. For each blind branch/history, filter the stage-specific blind pool by the
history, require at least 24 valid particles and 16 unique extensions, then
resample to 32. A refreshed belief is exactly:

1. 32 deterministic resamples from the prior particle belief consistent with
   the new observation; and
2. 32 deterministic resamples from the corresponding generated multiset.

Thus dynamic and blind beliefs have exactly 64 particles and identical
retention width. No simulated truth particle is injected. Empty/constant,
invalid, inconsistent, or unavailable particles are never repaired.

## Candidate Roots And Depth-Three Value

Each tree has exactly four target-blind root candidates derived from its
initial particle multiset: the myopic-EIG root, fixed-support depth-three root,
and then the highest-immediate-EIG roots with new particle answer signatures,
breaking all ties by lower integer. No canonical target or realized endpoint
enters this set.

For each root, the dynamic score is immediate binary predictive entropy plus
the expected best second-query entropy under the first refreshed belief plus
the expected best third-query entropy under the second refreshed belief.
Probabilities at every branch are empirical particle frequencies. The second
and third queries maximize one-step EIG in their current belief and exclude
earlier queries. The history-blind score uses the identically sized blind
beliefs. Fixed depth three only filters/resamples the initial particles.

The policies are:

- `dynamic_depth3`: maximize the path-conditioned score;
- `call_matched_myopic`: maximize initial immediate EIG while consuming the
  same immutable response bank, then replan myopically after each answer using
  the dynamic refreshed belief;
- `history_blind_depth3`: maximize the blind score and update with blind pools;
- `fixed_support_depth3`: maximize fixed-support depth-three EIG;
- `deterministic_random`: choose from the same four roots using the tree seed,
  then use dynamic refreshed beliefs for later myopic queries.

## Response-Before-Outcome Ordering

Before any canonical concept, classical grammar extension, realized answer,
posterior target, or policy endpoint is loaded, the producer must bank all
6,400 raw responses, exact prompt/payload hashes, seeds, model routes, finish
reasons, token usage, and costs. An independent replay must reconstruct every
prompt from public constants and verify structural privacy.

Only after a complete, transport-valid bank exists may a separate scorer load:

- the equal-weight canonical 33-concept bank used by prior Number Game
  calibration audits; and
- the frozen classical grammar bank for aggregate novelty only.

No canonical expression/name, grammar membership, target answer, Brier score,
preferred root, or endpoint value appears in a model prompt.

## Frozen Mechanics Gates

All gates are conjunctive.

### Transport and particle validity

1. Exactly 6,400 accepted requests and HTTP attempts; zero retries, provider
   errors, reasoning tokens, forced exits, format repairs, or non-`stop`
   finishes; exact model/seeds/payloads; stage cost at most `$3.50`.
2. Every initial and refresh group passes its frozen valid/unique floors; every
   dynamic and blind belief has exactly 64 particles; all rows normalize and
   all values are finite.
3. Prompt privacy, response-before-outcome ordering, source/artifact hashes,
   account-wide budget accounting, and unopened development/confirmation
   paths replay exactly.

### LLM ownership and stability

4. At least 10% of unique valid second-refresh conditioned extensions are
   outside the frozen 416,366-extension grammar bank, and every tree has at
   least one grammar-novel second-refresh extension.
5. Recompute dynamic root choice on two disjoint request-slot halves, with
   all particle widths halved identically. The two half estimators agree with
   the full dynamic root on at least three of four trees.
6. Dynamic and call-matched-myopic roots differ on at least three of four
   trees; dynamic and history-blind roots differ on at least two.

### Belief and first-link calibration

7. Across all reachable second-refresh histories, path-conditioned canonical
   posterior-predictive MSE is at least 5% below history-blind MSE, and exact
   canonical truth-extension coverage is at least five percentage points
   higher. The same histories and retained particles are used in both arms.
8. Across the 16 tree/root cells, Spearman correlation between dynamic
   depth-three score and negative realized canonical posterior-predictive
   Brier is at least `.40`. The selected dynamic root is among the two best
   realized roots on at least three trees.

### Mechanics endpoint

9. On the equal-weight canonical 33 concepts, dynamic depth three reduces mean
   terminal posterior-predictive Brier by at least 3% versus call-matched
   myopic, wins on at least three of four tree means, and is nonworse than
   deterministic random. Queries are replayed with common target answers and
   Brier excludes the three queried integers.
10. Dynamic depth three is nonworse than history-blind depth three on mean
    Brier and truth-extension coverage. Fixed-support depth three is reported
    without a directional gate because this stage isolates non-myopia and
    conditioning before claiming support expansion superiority.

A pass is `atomic_particle_mechanics_pass` and authorizes only freezing a
fresh powered development. Any failed conjunction is
`atomic_particle_mechanics_null`; malformed/partial/budget-invalid execution is
`failed_closed`. Neither status opens a paper headline or confirmation.

## Budget And Failure Semantics

The immutable Europe/London account-wide daily cap remains `$5.00`; unrelated
usage counts from the authenticated prior-day/current-day boundary. Before
each block, reread cumulative credits/usage and the exact Qwen catalog row,
reserve the full byte-as-token worst-case exposure of that block, and require
accepted prior cost plus the reservation to fit both the `$3.50` stage cap and
remaining daily/account balance. Reauthorize immediately before every HTTP
attempt. Reconciled spend is the maximum of posted usage since the frozen daily
boundary and locally recorded accepted-request cost.

Any failure banks its exact prefix once, authorizes nothing, and forbids retry,
seed substitution, threshold repair, favorable-subset scoring, or response
reuse. Synthetic/adversarial tests and an adapter-construction rehearsal must
pass before an immutable execution binding may be pushed.
