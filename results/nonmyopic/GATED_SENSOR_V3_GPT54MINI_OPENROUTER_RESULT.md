# Gated Sensor v3 GPT-5.4 Mini OpenRouter Result

## Decision

The frozen S1 proposal-quality gate **failed**. The indexed-v3 representation
supplied exact branch-conditioned predicate marginals, but GPT-5.4 Mini's proposed
continuations recovered only `0.6938` of the exact same-root value, below the
registered `0.90` threshold and `0.0896` below matched-random continuations on the
same StrategyEIG-reached states. The separately preregistered 30-pair confirmation
is therefore not authorized.

## Protocol And Mechanics

- Model: non-thinking `openai/gpt-5.4-mini`, temperature zero.
- S1: four paired trials, eight rounds, K4, seed `24112`, 2,000 bootstraps.
- Serving: 52/52 accepted physical cells, zero invalid responses, zero reasoning
  tokens, zero forced exits, and zero rollout-scoring LLM calls.
- Cost: `$0.11297670` for 201,818 prompt and 1,300 completion tokens.
- All paired-truth, legal-action, shared-root, and terminal-call mechanics passed.

## Frozen Proposal Gate

| Arm | h2 states | Continuation efficiency | Root coverage | Optimal continuation |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.4 Mini StrategyEIG | 28 | 0.6938 | 1.0000 | 0.2500 |
| Matched random on GPT states | 28 | 0.7834 | 1.0000 | 0.2857 |
| Reached random-policy states | 28 | 0.7982 | 1.0000 | 0.2857 |

Both registered requirements failed: continuation efficiency was below `0.90`,
and GPT-minus-matched-random was `-0.0896` rather than at least `+0.05`.

The policy endpoints were secondary and could not rescue the proposal gate.
StrategyEIG beat both greedy controls by `+1.1571` entropy-AUC nats, but lost to
matched random by `-0.1219` (95% interval `[-0.2683, -0.0078]`, 0/1/3
wins/ties/losses). Truth-log AUC agreed: `-0.1768`
`[-0.3511, -0.0025]`.

## Failure Localization

A zero-call audit reconstructed every branch belief and ranked each selected
follow-up by exact immediate EIG. The table compares the frozen v3 S1 with the
previous banked v2 probe on their respective StrategyEIG-reached states.

| Choice diagnostic | Indexed v2 | Indexed v3 |
| --- | ---: | ---: |
| Requests / branch choices | 56 / 328 | 28 / 164 |
| Overall branch-choice EIG efficiency | 0.4395 | 0.3654 |
| Activation-root EIG efficiency | 0.8007 | 0.8628 |
| Measurement-root EIG efficiency | 0.2311 | 0.0784 |
| Measurement-root optimal-choice rate | 0.0721 | 0.0096 |
| Measurement branches choosing activation | 0.3462 | 0.6731 |
| Measurement branches with zero EIG | 0.3462 | 0.6731 |
| All branches selecting index zero | 0.5030 | 0.6524 |

The v3 model handled activation roots reasonably: after a panel was activated, it
usually selected an informative precise test. The failure was the converse. After
a measurement root, 70/104 branch choices selected another panel activation even
though that follow-up was terminal in the two-step strategy and therefore had
exactly zero information gain. Only 1/104 measurement-root branches selected an
EIG-optimal continuation.

This exposes a fixed-shape interface failure. Activation actions occupy the first
three entries in measurement follow-up menus, while the schema example and many
responses favor small or zero indices. The model also appears to reason about an
activation's future usefulness beyond the represented horizon. Exact verification
can reject or rank complete proposed strategies, but it cannot replace a bad
follow-up inside every candidate. Branch-conditioned marginals therefore did not
repair the missing proposal link.

The v2/v3 numerical contrast is diagnostic rather than a causal interface estimate:
the probes use different seeds and sample sizes. The frozen v3 failure itself is
unambiguous because its registered thresholds and matched-random comparison are
evaluated on the same exact reached states.

## Artifacts

- S1 run: `gated_sensor_v3_gpt54mini_openrouter_s1_20260722/RESULT.json`
- Continuation audit: `gated_sensor_v3_gpt54mini_openrouter_s1_audit_20260722/`
- Indexed-choice audit: `gated_sensor_v3_gpt54mini_openrouter_s1_choice_mechanics_20260722/`
- v2 diagnostic comparator: `gated_sensor_v2_gpt54mini_choice_mechanics_20260722/`

