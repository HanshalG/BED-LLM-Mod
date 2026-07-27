# WorldValues Persona Non-Myopic Opportunity Result

## Decision

The exact structural gate **fails decisively**. The released fixed-persona
model is effectively myopic for the frozen target-entropy task, so no dynamic
LLM persona-support smoke is authorized.

No seed, question set, prior, temperature, objective, or threshold is changed
after the result.

## Protocol

- Official repository commit:
  `fbd8e19eed6af960b64e3afae13e3bcf4020b73f`
- Direct-distribution matrix SHA-256:
  `24d5d7b3a9bdf94894952dc7bfff39d409f8f78c8131ccd7de87fb292068638e`
- Frozen task-spec SHA-256:
  `da1ae2fdc0d1f0495e8e6c01c78a1c656dab0416076419c65f2e3d5e52d5602d`
- 20 content-blind tasks, each with 8 target and 24 candidate questions.
- Uniform prior over all 2,058 released semantic personas.
- Exact four-category response likelihoods from the released GPT-5-mini
  persona-question matrix.
- Objective: mean posterior-predictive target entropy in nats.
- Myopic and adaptive-d2 roots receive the same candidates and each receives
  its own answer-conditioned exact best followup.

## Result

| Metric | Observed | Frozen gate |
|---|---:|---:|
| Complete tasks | 20/20 | 20/20 |
| Mean initial target entropy | `1.183241` | `>= .5` |
| Dynamic immediate/final score tasks | 20/20, 20/20 | 20/20, 20/20 |
| Adaptive-d2 root changes | 1/20 | `>= 5/20` |
| Strict immediate-sacrifice tradeoffs | 1/20 | `>= 4/20` |
| Mean d2 final advantage | `0.0000201` nats | `>= .001` |
| Strict total final advantage | `0.0004023` nats | `>= .01` |
| Mean strict immediate sacrifice | `0.0013406` nats | `>= .002` |

The sole changed task was task 11:

- myopic root `Q29`;
- adaptive-d2 root `Q197`;
- immediate sacrifice `0.0013406` nats; and
- final advantage `0.0004023` nats.

All other 19 tasks selected exactly the myopic root at depth two. Every frozen
mechanics and non-saturation gate passed; every substantive depth gate failed.

## Interpretation

The fixed persona dictionary is expressive, but the released model assumes
question responses are conditionally independent given one persona. Under that
model, target-entropy reduction is nearly adaptively submodular: the best
one-step question is also the best first step of the two-question policy.

Path-dependent LLM regeneration could break this structure, but adding it after
the exact substrate shows no meaningful first-action conflict would make the
depth effect depend on an unvalidated prompt artifact. That is the wrong order
of evidence. This construction closes before any model call.

The source remains useful as a fixed-support semantic BED baseline. A future
dynamic-persona experiment needs either a native delayed-reveal structure or
real respondent trajectories with a prospectively demonstrated root conflict.

## Artifacts and Cost

- Public audit:
  `results/nonmyopic/worldvalues_persona_nonmyopic_opportunity/AUDIT.json`
- Audit SHA-256:
  `a296701273c3c657ec059d8800330736ac8bbf1d45fc9e4f884a58e71827519d`
- Preregistration:
  `results/nonmyopic/WORLDVALUES_PERSONA_NONMYOPIC_OPPORTUNITY_PREREGISTRATION.md`

OpenRouter calls: `0`; cost: `$0`; OatML/cluster use: none. The `$25` reserve
and `$8.3292075` pre-Monday allowance are unchanged.
