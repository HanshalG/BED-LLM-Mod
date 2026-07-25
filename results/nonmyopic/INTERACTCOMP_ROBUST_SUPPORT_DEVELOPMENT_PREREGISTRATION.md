# InteractComp Robust-Support Development Preregistration

Date: 2026-07-25

## Purpose

Develop one prospectively specified repair for the open-support failure observed
in the completed InteractComp first-link gate. This is an open-task development
diagnostic, not a fresh-task result and not a non-myopic efficacy test.

The motivating failure is precise: on task 38, all current particles predicted
the same answer to the only truth-recovering clarification. Its ordinary
fixed-support EIG was therefore zero even though the real response contradicted
the entire support and caused regeneration to recover the target.

## Frozen Inputs

- Reuse only completed tasks 76 and 38 from run
  `interactcomp-first-link-opportunity-20260725T102000Z`.
- Prior public artifact SHA-256:
  `86c741d9cf38f83b282b8f5048d50d10b54a563ca05bdec897ae7ec72b9a0e91`.
- Prior private target-blind artifact SHA-256:
  `fb90c37a65d004a79b1b0aff8dce986aaf0de3ff0abbd633b68aff01b83d3f95`.
- Reuse the frozen ambiguous questions, eight current particles, four
  clarification roots, current-particle Y/N/U classifications, and external
  root endpoints.
- No hidden context, target answer, true response, or prior refreshed particle
  is included in a new model prompt.

The prior endpoint values are known because these are development tasks. The
harness nevertheless freezes every new score and checkpoints all responses
before loading the public endpoint artifact.

## Auxiliary Support

- Model: `openai/gpt-5.4-mini` through OpenRouter.
- Reasoning disabled; generation temperature `.7`; classification temperature
  `0`.
- Generate eight independent plausible answer/profile particles per task that
  must be outside the entire current support.
- The prompt asks for materially different interpretations and shared
  assumptions the current population may have missed.
- Exact current-support entity repetitions fail closed.
- At least four unique normalized auxiliary entities are required per task.
- Classify each auxiliary particle as Y/N/U on the four already-frozen roots.
- Strict two-line particle and four-character classification parsers are
  unchanged from the completed first-link gate.
- No response repair, replacement, or scientific reissue is allowed.

## Frozen Score

For root `q`, let `p_0(y|q)` be the empirical Y/N/U distribution under the
eight current particles and `p_1(y|q)` the corresponding distribution under
the eight auxiliary particles. Introduce a balanced diagnostic model-identity
variable `Z`, where `Z=0` denotes current support and `Z=1` auxiliary support.

The misspecification information is:

```text
I(Z;Y|q) =
  H(0.5 p_0(.|q) + 0.5 p_1(.|q))
  - 0.5 H(p_0(.|q))
  - 0.5 H(p_1(.|q))
```

The only tested robust score is:

```text
robust_score(q) = current_EIG(q) + I(Z;Y|q)
```

where `current_EIG(q) = H(p_0(.|q))` under the existing deterministic particle
classifications. Natural logarithms are used. The `0.5/0.5` mixture, unit
coefficient, formula, and frozen-order tie breaking may not be tuned after
auxiliary responses.

This follows the model-misspecification BED principle of adding an auxiliary
model and a Bernoulli indicator of whether the original model is correct. It
tests whether LLM-generated outside support makes consensus-check questions
prospectively valuable.

## Exact Calls

| Stage | Mini calls |
|---|---:|
| Outside-support particles | 16 |
| Auxiliary-particle classifications | 16 |
| **Total** | **32** |

The run must make exactly 32 physical requests and 32 HTTP attempts.

## Exact Gates

Every gate must pass:

1. exactly 32 physical requests and 32 HTTP attempts;
2. zero transport retries, reasoning tokens, and forced exits;
3. at least four unique auxiliary entities per task, with no current-support
   entity repeated;
4. robust score selects the externally best root on both open tasks;
5. robust-score endpoint Spearman correlation is finite and positive on both
   tasks;
6. mean robust-score endpoint Spearman is at least `.50`;
7. mean selected endpoint gain over ordinary EIG is at least `.0625`; and
8. adapter cost is at most `$0.25`.

Passage authorizes only a separately preregistered ranking gate on fresh
encrypted tasks, with this exact score frozen. Failure closes this exact
auxiliary prompt and score. No coefficient, mixture weight, task, or formula
selection from the returned particles is permitted.

## Budget

- Projected cost: `$0.05`.
- Hard run cap: `$0.25`.
- Project-ledger spend before the run: `$86.25126556920749`.
- Monday local allowance remaining: `$14.892039249999925`.
- Last authenticated OpenRouter remaining: `$44.177721634`, or `$19.177721634`
  above the protected `$25` reserve.
- OatML resources: prohibited.

The stricter live, local, and run-cap budget applies and is rechecked before the
paid command.

## Deterministic Verification

The fixture completed exactly 32 simulated requests with zero retries,
reasoning, or forced exits; froze all robust scores before endpoint loading; and
failed scientific gates on deliberately generic auxiliary particles. Focused
tests:

```text
pytest -q tests/test_interactcomp_robust_support_development.py \
  tests/test_interactcomp_first_link_opportunity.py \
  tests/test_helpers_load_config.py tests/test_core_config.py
108 passed
```
