# Zendo Particle-Multiset Confirmation Preregistration

Date frozen: 2026-07-25, before any response on these tasks.

## Claim

The one-task `mu` smoke passed every frozen gate: a blinded scorer selected a
lower-immediate-EIG root whose realized outcome-conditioned regenerated belief
was behaviorally closer to hidden truth than the beliefs induced by myopic and
fixed-support depth-two roots.

This confirmation tests whether that first-link result transfers across all seven
remaining fresh public Zendo rules and whether **aligned future regenerated
beliefs**, rather than generic semantic experiment scoring or extra model compute,
are load-bearing.

The benchmark's rules are public development rules, not a sealed test set. No
population-level or paper claim will be made unless the prospective aggregate
gates pass.

## Frozen source, tasks, and seeds

- Official source commit:
  `af07590c4f4f617a79791e173460e5a4322b727f`
- Case-file SHA-256:
  `6440c543ff491af13606b79c57384281fae4e8bc205366e67be06d1c81fbacbc`
- Tasks, in order:
  `upsilon`, `iota`, `kappa`, `omega`, `nu`, `xi`, `psi`.
- None has received a prior paid Zendo response. The earlier failed `zeta` smoke
  stopped before its planned `xi` task.
- Base seed `24371`; per-task scene/audit seed is
  `24371 + 7919 * RULE_ORDER.index(task)`.
- Initial observation is the first official positive scene that validates under
  the frozen one-to-six-block DSL. This is index 0 for every task except `xi`,
  where the first representable positive is index 1.
- Every endpoint uses a separate deterministic 512-scene random legal audit bank.
- All initial-scene, 256-scene pool, 512-scene audit, and random-control values
  were materialized before responses in
  `results/nonmyopic/zendo_particle_multiset_confirmation/SCENE_BANK_MANIFEST.json`.

## Frozen calls and policies

Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.

Exactly 84 physical requests:

- 7 initial 12-particle executable belief populations;
- 56 outcome-isolated branch refreshes: 7 tasks by 4 roots by 2 labels;
- 21 independent score calls: aligned, root-only, and shuffled for every task.

Concurrency is 8. Projected cost is `$1.10`; hard cap is `$1.50`. This remains
inside the through-Monday allowance and protects the `$25` reserve. OatML is not
used.

Each task uses the preregistered multiset mechanics:

- target-blind 256-scene bank;
- four informative roots spanning `1.00`, `0.75`, `0.50`, and `0.25` of maximum
  initial-support EIG;
- repeated ASTs retain particle multiplicity;
- exact best continuation under each regenerated branch population;
- hidden official predicate evaluated only after all populations and all three
  score vectors freeze.

Policies share the same generated populations, roots, continuations, and hidden
outcomes:

- **Aligned model-aware:** blinded scorer sees each root with its own two
  outcome-conditioned regenerated beliefs and exact continuations.
- **Root-only compute match:** one independent LLM response sees the current
  particles, roots, and predicted label probabilities, but no future regenerated
  beliefs, continuations, or exact EIG values.
- **Shuffled future-belief control:** one independent LLM response sees each root
  paired with another root's complete future pathways under a deterministic
  cyclic shift `1 + task_index % 3`.
- **Myopic:** exact one-step initial-support EIG.
- **Fixed support:** exact depth-two EIG retaining the initial particles.
- **Random:** deterministic uniform root from `Random(task_seed + 17)`.

The primary endpoint is posterior-weighted behavioral agreement with the hidden
official predicate after the selected root and that realized branch's exact
continuation. Root endpoints are computed once and reused by every policy, giving
paired common-random-number comparisons.

## Frozen mechanics gates

All must pass:

- exactly 84 adapter requests and HTTP attempts;
- zero retries, reasoning tokens, forced exits, repairs, and parse failures;
- every 12-particle population has at least eight unique ASTs;
- every initial population has at least six behavioral signatures;
- every task has four distinct informative root signatures;
- every task has at least four behaviorally distinct refreshed populations;
- all exact continuations have positive finite EIG;
- every aligned, root-only, and shuffled score vector varies;
- aligned scores have a unique maximum on at least five tasks;
- cost at most `$1.50`.

## Frozen scientific gates

All must pass:

- endpoint range at least `0.10` on at least 4/7 tasks;
- aligned root differs from exact myopic on at least 4/7;
- aligned sacrifices at least `0.01` immediate-EIG nats on at least 3/7;
- aligned versus myopic: mean endpoint gain at least `0.04`, at least four wins,
  at most two losses, and exact task-level one-sided sign-flip `p <= 0.10`;
- aligned versus fixed: mean gain at least `0.03`, at least four wins, at most two
  losses;
- aligned versus root-only: mean gain at least `0.03`, at least four wins, at most
  two losses;
- aligned versus shuffled: mean gain at least `0.03`, at least four wins, at most
  two losses;
- aligned versus random: mean gain at least `0.04` and at least four wins;
- mean aligned score-endpoint Spearman at least `0.25`;
- mean aligned Spearman exceeds root-only and shuffled by at least `0.15` each;
- mean within-task aligned pairwise endpoint-ranking accuracy exceeds root-only
  and shuffled by at least `0.05` each.

No task, seed, threshold, parser, prompt, particle interpretation, scorer mapping,
or endpoint will change after responses. A serving or scientific failure closes
this exact confirmation. No partial favorable subset will be reported as a pass.
