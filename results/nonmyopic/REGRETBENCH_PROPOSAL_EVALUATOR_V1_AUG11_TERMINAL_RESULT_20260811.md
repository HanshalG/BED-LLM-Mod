# RegretBench Proposal-Evaluator V1 Aug 11 Terminal Result

## Disposition

The prospectively frozen proposal-evaluator-v1 exact20 mechanics smoke is **failed closed**. It authorizes nothing. No answer-conditioned proposal, answer-free proposal, branch evaluator, exact branch update, second action, task endpoint, development cohort, confirmation cohort, efficacy analysis, or paper headline opened.

The wrapper was invoked once from pushed commit `39d837a3` after an authenticated read-only preflight verified exact hashes, pristine paths, live DeepSeek V4 Flash 0731 text/schema/seed support, unchanged $0.08/$0.18 per-million-token prices, and the full $0.20 account-wide reservation.

## Observed Failure

Four clean nonreasoning requests completed: two root proposals and their two answer-blind evaluators. Both selected root questions had strongly nondegenerate partition EIG, `1.913548` and `2.079442` nats. Execution stopped before all 16 branch requests at the frozen gate:

```text
selected root branch reply is not a real facet value
```

For both tasks, neither of the two highest-mass predicted reply strings exactly equaled a canonical source-slot string.

## Codec Diagnosis

This is a mixed reply-codec failure, not evidence about non-myopic policy efficacy or the answer-conditioned proposal mechanism, which never ran.

The diagnostic has two components:

- On the township task, the selected replies were semantically valid county names written with a natural `County` suffix, while the source codec stores bare county names.
- On the Lord Norton task, the selected replies expressed the intended person-versus-title distinction in natural prose, while the source codec stores terse category labels.

However, lower-mass groups also introduced unsupported interpretations and locations. Therefore removing the source-validity gate after seeing these responses would be invalid. The exact-string interface and seed remain closed.

A genuinely new successor needs a frozen executable reply codec, not a relaxed threshold. Plausible routes are prospectively generated answer-option IDs with an independently tested environment mapper, or a separately gated semantic reply canonicalizer. Either route must be tested on untouched tasks before branch planning and must preserve answer-free compute matching and one-time conditioning.

## Cost And Provenance

- Locally measured stage cost: `$0.000700813708`.
- Conservative Aug 11 account-wide recorded spend after both closed smokes: `$0.002769002708`.
- Daily failure SHA-256: `a66a45339b887b70ca082309c2cd595a16259414af4fb36e0e77d33b043711af`.
- Stage ledger SHA-256: `307cbb01ec7602e5b43cc999ce897878bbb2a5be227de5b173249680e0140f5a`.
- Private raw-response SHA-256: `f38dc6d759f4a593105c7eeefe35c1fc5286b3da76d85743dc121e3c1e379f9d`.
- Public diagnostic: `results/nonmyopic/regretbench_proposal_evaluator_v1_smoke/TERMINAL_AUDIT.json`.

The private raw responses and run log remain untracked. The public terminal artifacts expose no model prompt, raw response, source-slot value, task endpoint, or hidden intent.
