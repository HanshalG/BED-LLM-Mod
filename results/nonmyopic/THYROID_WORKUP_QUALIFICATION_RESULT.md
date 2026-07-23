# UCI Thyroid Blood-Workup Qualification Result

The preregistered exact depth qualification **passed every gate** on a fresh
1,000-patient sample from the complete 7,200-row UCI ann-thyroid cohort. This is a
natural delayed-acquisition result grounded in the dataset's released delay,
group, and expense metadata. It does not yet establish an LLM-policy advantage.

| Endpoint (d2 - d1) | Mean gain | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Entropy AUC | +0.165898 | [+0.152285, +0.179069] | 810/0/190 |
| Truth-log AUC | +0.166883 | [+0.135923, +0.199760] | 858/0/142 |
| Final entropy reduction | +0.242335 | descriptive | 798/165/37 |
| Final class accuracy | +0.059 | descriptive | 69/921/10 |

Depth one queried `on-thyroxine` first in all 1,000 trials. Depth two instead
selected `collect:blood-sample`, which has exactly zero immediate information gain,
in all 1,000 trials and then queried TSH in round two. Mean target-entropy traces
after rounds 1--8 were:

- Depth one: `0.299972, 0.293617, 0.291258, 0.282328, 0.278353, 0.274203, 0.271471, 0.269910`.
- Depth two: `0.310253, 0.166421, 0.136011, 0.112901, 0.090838, 0.054966, 0.034965, 0.027574`.

Final class accuracy was 92.4% for depth one and 98.3% for depth two. The temporary
round-one entropy cost is therefore the setup action the task was designed to test,
not a post hoc artifact: depth two accepts no immediate information to unlock the
assay that drives the subsequent improvement.

An independent audit replayed all 16,000 recorded decisions from the official raw
cohort and re-solved 9,497 unique planning subtrees with a separate recursion. Every
recorded action was independently optimal, every state, observation, posterior,
planning cost, and aggregate matched, and fresh-bootstrap intervals remained
strictly positive: `[+0.152230, +0.179057]` for entropy AUC and
`[+0.135525, +0.199662]` for truth-log AUC.

All qualification and audit computations were exact and made zero LLM calls.

Artifacts:

- `results/nonmyopic/THYROID_WORKUP_PREREGISTRATION.md`
- `results/nonmyopic/thyroid_workup_qualification_20260723/REPORT.json`
- `results/nonmyopic/thyroid_workup_qualification_20260723/REPORT.md`
- `results/nonmyopic/thyroid_workup_qualification_20260723/AUDIT.json`
- `results/nonmyopic/thyroid_workup_qualification_20260723/AUDIT.md`

Data source: UCI Thyroid Disease, https://doi.org/10.24432/C5D010, CC BY 4.0.
