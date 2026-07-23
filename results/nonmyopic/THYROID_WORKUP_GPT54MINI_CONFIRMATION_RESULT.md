# UCI Thyroid GPT-5.4 Mini Trajectory Confirmation Result

The preregistered late-state S0 serving smoke passed, but the fresh 50-patient S1
trajectory confirmation **failed four of six scientific gates**. The result is a
useful partial positive: non-thinking GPT-5.4 Mini improves realized entropy and
truth-log AUC over exact depth one, but does not beat matched-random continuations
and captures only 30.8% of the exhaustive depth-two gain.

| S1 endpoint | Mean gain | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Entropy AUC vs exact d1 | +0.064092 | [+0.020485, +0.104630] | 33/8/9 |
| Truth-log AUC vs exact d1 | +0.088291 | [+0.019775, +0.177269] | 36/8/6 |
| Entropy AUC vs matched random | -0.033888 | [-0.073944, +0.007266] | 18/5/27 |
| Truth-log AUC vs matched random | +0.009711 | [-0.047183, +0.087825] | 19/5/26 |

Exhaustive depth two improves entropy AUC over depth one by `+0.208138`; GPT
recovers `30.793%` of that gain, below the frozen 60% gate. Its first action is blood
collection in only 2/50 trajectories (4%), versus 8/50 for matched random and 50/50
for exhaustive depth two.

The initial-state mechanism explains the gap. The initial prompt and belief are
identical across all 50 trials, but GPT proposes `query:age` after blood collection
in 48 responses and `query:tsh` in two. Exact scoring therefore selects
`query:on-thyroxine` as the root in those same 48 trials and selects collection only
for the two TSH continuations. Matched random occasionally samples informative assay
continuations and consequently beats GPT on mean entropy AUC.

Mean post-action entropy traces were:

- Exact d1: `0.315109, 0.329436, 0.317024, 0.327632, 0.323924, 0.324710, 0.303867, 0.290006`.
- Exact d2: `0.310253, 0.135459, 0.137250, 0.102688, 0.081818, 0.047285, 0.026388, 0.025461`.
- GPT named h2: `0.314134, 0.336306, 0.326669, 0.308961, 0.269528, 0.234033, 0.157714, 0.071625`.
- Matched random h2: `0.311207, 0.325276, 0.270503, 0.223692, 0.210437, 0.183467, 0.122955, 0.100331`.

S0 accepted 12 cells after one corrected response and cost `$0.02180625`. S1
accepted all 350 logical cells after 10 corrected responses and cost `$0.44512275`.
Together they used 373 physical requests, 685,236 prompt tokens, 41,588 completion
tokens, zero reasoning tokens, zero forced exits, and `$0.46692900`. The cumulative
project spend is `$34.92378331`, leaving `$75.07621669` under the `$110` ceiling.

An independent audit replayed every one of the 1,600 arm decisions, regenerated all
matched-random policies, independently re-solved 2,519 exact subtrees, and verified
every posterior and aggregate. Fresh intervals preserve the result: entropy versus
d1 `[+0.019582,+0.104917]`, entropy versus random `[-0.074903,+0.005707]`, and the
registered scientific gate remains false.

The evidence supports a specific boundary: exact verification can exploit a good
LLM continuation, and GPT is mechanically robust, but semantic test names and class
probabilities do not ground the model in empirical feature-target dependence. A
future method must expose calibrated continuation utility or learned likelihood
summaries; increasing model strength alone is insufficient.

Artifacts:

- `results/nonmyopic/THYROID_WORKUP_GPT54MINI_CONFIRMATION_PREREGISTRATION.md`
- `results/nonmyopic/thyroid_workup_gpt54mini_robust_smoke_20260723/SMOKE.json`
- `results/nonmyopic/thyroid_workup_gpt54mini_confirmation_20260723/CONFIRMATION.json`
- `results/nonmyopic/thyroid_workup_gpt54mini_confirmation_20260723/AUDIT.json`
