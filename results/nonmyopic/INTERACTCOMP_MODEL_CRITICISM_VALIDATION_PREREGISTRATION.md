# InteractComp Model-Criticism Validation Preregistration

Date: 2026-07-25

## Question

On prospectively selected, untouched InteractComp tasks where the LLM's current
answer support has collapsed, does an LLM-generated auxiliary support identify
clarification questions that expose model misspecification and induce better
exact-answer recovery than ordinary EIG, compute-matched augmented-support EIG,
and random selection?

This is a fresh first-link ranking validation. It is not yet a depth-two policy
test. Passage authorizes only a separately preregistered path-dependent
depth-two comparison.

## Frozen Source And Screen

- InteractComp repository:
  `https://github.com/FoundationAgents/InteractComp`.
- Repository commit: `9cdf7f804f527ad32a405efaa6c86aae03692556`.
- Encrypted data SHA-256:
  `0bd0ccc4b69c228c04c15b1147211adc6c6483b852d74e5ac7e5a34c9db80496`.
- Structural eligibility and manifest are unchanged from the first-link gate:
  156 eligible tasks, seed `24377`, manifest SHA-256
  `9375f5cb3e52590c5eafca25c96f6378c6f44da674162ffdf63c5fe949eeb6c6`.
- Previously opened manifest entries at indices `75` and `37` are excluded.
- The next 16 untouched screen indices are:
  `109, 141, 97, 80, 84, 21, 34, 123, 106, 170, 115, 45, 59, 29, 25, 94`.
- Corresponding benchmark IDs:
  `110, 142, 98, 81, 85, 22, 35, 124, 107, 171, 116, 46, 60, 30, 26, 95`.
- Validation seed: `24379`; random-control seed: `24380`.

No content from these 16 tasks has been inspected. All hidden contexts and
answers remain encrypted during screening.

## Prospective Enrollment

1. Generate eight independent answer/profile particles for every screen task
   using `openai/gpt-5.4-mini`, non-thinking, temperature `.7`.
2. Count exact normalized unique answer entities.
3. Enroll the first six manifest-ordered tasks with at most four unique entities
   among eight particles.
4. If fewer than six qualify, stop after the 128 screening calls without
   decrypting a context, answer, or endpoint.

Enrollment depends only on target-blind model-state collapse. There is no
replacement based on response quality, endpoint, domain, or target recovery.

## Frozen Belief And Roots

For each enrolled task:

- generate four entity-name-free yes/no clarification questions;
- classify all eight current particles Y/N/U on all roots;
- generate 16 proposed outside-support particles;
- classify every proposal as semantically distinct (`D`) or the same
  answer/alias/spelling/broader-narrower label (`S`) relative to current support;
- retain the first eight `D` proposals, failing closed if fewer than eight
  survive; and
- classify those eight auxiliary particles Y/N/U on all roots.

The generator, semantic validator, and classifier are all
`openai/gpt-5.4-mini`, non-thinking. Semantic validation is an individual
target-blind call per proposal. Repeated auxiliary entities are retained as
model-induced particle mass after semantic distinction from current support.

## Frozen Scores And Controls

For root `q`, let `p_0(y|q)` and `p_1(y|q)` be empirical Y/N/U distributions
under current and auxiliary particles.

Primary model-criticism score:

```text
I(Z;Y|q) =
  H(0.5 p_0(.|q) + 0.5 p_1(.|q))
  - 0.5 H(p_0(.|q))
  - 0.5 H(p_1(.|q))
```

`Z` is the balanced current-versus-auxiliary model identity. The primary policy
maximizes this score because enrollment has already established current-support
collapse.

Paired controls:

- current-support EIG: `H(p_0(.|q))`;
- compute-matched augmented-support EIG:
  `H(0.5 p_0(.|q) + 0.5 p_1(.|q))`; and
- seeded random root.

All methods share identical particles, roots, classifications, true responses,
and refreshed endpoints. Ties use frozen root order.

## Realized Transition And Endpoint

After all scores freeze:

1. decrypt hidden contexts;
2. obtain the true Y/N/U answer to every root from `openai/gpt-5.4`,
   non-thinking, using only the benchmark context;
3. regenerate eight answer/profile particles after every realized root answer;
4. require at least six valid refreshed particles per root;
5. checkpoint all target-blind responses and scores; and
6. only then decrypt exact benchmark answers.

The endpoint is exact normalized target-answer mass in each refreshed
population. Initial target mass, root endpoint range, oracle root, score-endpoint
Spearman correlation, and each policy's selected endpoint are reported.

## Exact Calls

| Stage | Mini | GPT-5.4 |
|---|---:|---:|
| Sixteen-task initial screen | 128 | 0 |
| Four roots on six enrolled tasks | 24 | 0 |
| Current-particle classifications | 48 | 0 |
| Sixteen auxiliary proposals/task | 96 | 0 |
| Semantic distinctness validation | 96 | 0 |
| Retained auxiliary classifications | 48 | 0 |
| True closed-mode responses | 0 | 24 |
| Realized-root support refreshes | 192 | 0 |
| **Total** | **632** | **24** |

The full run must make exactly 656 physical requests and 656 HTTP attempts.
There is no scientific repair, replacement, or response reissue.

## Exact Gates

Every gate must pass:

1. exactly 656 physical requests and 656 HTTP attempts;
2. zero transport retries, reasoning tokens, and forced exits;
3. exactly six prospectively enrolled supports, all with at most four unique
   initial entities;
4. at least three unique roots and two non-unknown true responses per task;
5. eight semantically distinct auxiliary particles per task;
6. at least six valid particles in every refreshed root;
7. at least four tasks have zero initial target mass;
8. at least four tasks recover target mass of at least `1/8` under some root;
9. at least four tasks have endpoint range at least `1/8`;
10. at least four tasks are rankable;
11. mean model-criticism score-endpoint Spearman is at least `.20`;
12. mean selected endpoint advantage is at least `.02` over current EIG,
    compute-matched augmented EIG, and seeded random separately;
13. model criticism beats current EIG on at least two tasks and loses on at most
    one;
14. model criticism selects an oracle root on at least three tasks; and
15. adapter cost is at most `$1.50`.

Failure closes this exact screen, prompt, semantic filter, and state-switch
rule. There is no favorable task subset, threshold relaxation, model swap, or
same-interface rerun. Passage authorizes only a fresh depth-two preregistration.

## Budget

- Projected cost: `$0.50`.
- Hard run cap: `$1.50`.
- Project-ledger spend before the run: `$86.27320081920747`.
- Monday local allowance remaining: `$14.870103999999941`.
- Last authenticated OpenRouter remaining: `$44.111601884`, or `$19.111601884`
  above the protected `$25` reserve.
- OatML resources: prohibited.

The stricter live, local, and run-cap budget is rechecked before launch.

## Deterministic Verification

The full fixture completed exactly 656 simulated requests: 632 generator and 24
responder calls, with zero retries, reasoning, or forced exits. It enrolled six
collapsed supports, completed semantic filtering and all branch refreshes,
froze scores before answer loading, and failed scientific gates on target-free
fixtures as intended. Focused tests:

```text
pytest -q tests/test_interactcomp_model_criticism_validation.py \
  tests/test_interactcomp_robust_support_development.py \
  tests/test_interactcomp_first_link_opportunity.py \
  tests/test_helpers_load_config.py tests/test_core_config.py
111 passed
```
