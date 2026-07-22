# Gated Sensor Indexed v2 Exploratory Result

## Decision

The indexed v2 interface fixed the literal-action legality failure, but it did not
pass the preregistered matched-random gate. The strongest exploratory probe used
GPT-5.4 Mini, four fixed root slots, the full exact 32-state posterior, eight paired
trials, and eight rounds.

| Entropy-AUC comparison | Mean gain | Paired 95% CI | Wins/ties/losses |
| --- | ---: | ---: | ---: |
| StrategyEIG - shared d1 | `+1.1200` | `[+1.0032, +1.2235]` | `8/0/0` |
| StrategyEIG - exhaustive d1 | `+1.1200` | `[+1.0099, +1.2229]` | `8/0/0` |
| StrategyEIG - matched random | `-0.0413` | `[-0.1854, +0.0795]` | `5/0/3` |
| StrategyEIG - exhaustive d2 | `-0.0790` | `[-0.2203, +0.0619]` | `3/0/5` |

The run accepted 104 requested cells, repaired two invalid responses on the allowed
single retry, and spent `$0.14869995`. All selected actions were legal, trials and
truths were paired, activation roots were shared with the random control, and exact
rollout scoring made no LLM calls.

## Interface Audit

The exploratory sequence isolated representation and information problems:

1. Literal action strings failed closed in v1 after 662 rejected attempts.
2. Indexed objects still induced inconsistent branch-key layouts.
3. A fixed `K x 2` integer matrix eliminated legality errors.
4. With six roots, a four-trial GPT-5.4 Mini probe beat random by only `+0.0166`
   nats and tied on three trials.
5. Reducing to four roots without the full posterior hurt the random comparison:
   `-0.2153` `[-0.3707, -0.0594]` over eight trials.
6. Showing the complete 32-state posterior recovered most of that deficit, yielding
   the strongest result above, but its confidence interval still crossed zero.

## Diagnosis

The exact scorer can value an activation root only through the continuation supplied
by the LLM. In losing trials, the proposal stays in a panel after its independent
predicate rank is exhausted or selects a precise test redundant with the branch
history. A zero-call exact closure audit separates root and continuation quality:

| Proposal source | h2 states | Root coverage | Continuation efficiency | Optimal continuation rate |
| --- | ---: | ---: | ---: | ---: |
| GPT-5.4 Mini | 56 | `1.0000` | `0.7591` | `0.3214` |
| Matched random on the same beliefs | 56 | `1.0000` | `0.8020` | `0.3571` |

Root coverage is exact same-root closure divided by exhaustive d2; continuation
efficiency is proposed value divided by same-root closure. Thus every audited fixed
root set contained an exhaustive-optimal root, but the LLM continuations recovered
`0.0429` less same-root value than random continuations rescored on the exact same
LLM-reached beliefs. The separately reached random arm gives `0.8182`; it is
descriptive because trajectory drift changes its beliefs. Full-posterior prompting
helps, but exact verification cannot repair a missing useful continuation policy.

This is therefore an exploratory interface audit, not a formal v2 policy result. It
strengthens the paper's mechanism claim: the first link in the non-myopic chain is
proposal quality under the simulated branch, and a correct exact scorer is
insufficient when that proposal is redundant.

## Artifacts

- Final full-posterior probe:
  `results/nonmyopic/gated_sensor_v2_gpt54mini_fullposterior_k4_probe_20260722/RESULT.json`
- Four-root prompt ablation:
  `results/nonmyopic/gated_sensor_v2_gpt54mini_k4_probe_20260722/RESULT.json`
- Six-root fixed-matrix probe:
  `results/nonmyopic/gated_sensor_v2_gpt54mini_fixed_probe_20260722/RESULT.json`
- Shared-state continuation audit:
  `results/nonmyopic/gated_sensor_v2_gpt54mini_continuation_audit_20260722/AUDIT.json`
- Interface implementation: `scripts/nonmyopic_gated_sensor_strategy_prior_v2.py`

Indexed-v2 exploration spent approximately `$0.52`, including an aborted thinking
probe with no policy endpoint. The project ledger stands at `$30.22841658` of the
`$40` budget, leaving approximately `$9.77`.
