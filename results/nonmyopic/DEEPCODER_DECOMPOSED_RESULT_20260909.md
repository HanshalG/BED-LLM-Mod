# Luna-medium execution decomposition: complete prospective null

Frozen launch commit ffda95b5, protocol SHA256
180e43d5a95e71990ae20c4abafaaf4213f53c10de89462304687899168eca8f.
All 96 requests completed without retries, transport/schema errors, or new uncertain
charges. Forecasts for all eight cases were saved before the evaluation outputs.
The paid process and independent replay both exited successfully.

Replay verifies all requests, source histories, intermediate execution, support,
restricted syntax weights, predictions, outcomes, gates and request receipts.
Terminal result SHA256:
57c0c4398bf7cbfbb6a5a5ec9872883770778a242ae5da44fb0ecb4e09f4ff11.
Forecast SHA256:
a49d86684e8417651da86e9b7e02f33786fa751550abaffb1465f98be2896c76.
Artifacts: deepcoder_decomposed_20260909/. No experiment implementation changed.

## Aggregate results

Lower half-Brier is better. All LLM arms include the complete compatible short
component. Coverage below counts cases with nonempty support, not calibrated beliefs.

| Arm | Mean Brier | Zero-mass targets / 256 | Coverage / 8 | Compatible proposals / proposed |
|---|---:|---:|---:|---:|
| Subgoal + execution | 0.369059794 | 85 | 6 | 4 / 192 |
| Execution only | 0.369082453 | 85 | 6 | 33 / 192 |
| Whole program | 0.368689502 | 85 | 6 | 72 / 121 |
| Exact short component | 0.412037037 | 96 | 5 | Not an LLM arm |

Both candidate gates fail. Neither stepwise arm beats whole-program generation;
each has only one case win greater than .01 against short support (case4), not the
required three. Both also fail all-eight nonempty support. Their roughly 10.4%
aggregate reduction versus short support alone cannot rescue the other criteria.
All arms have infinite aggregate NLL because of zero-mass targets; JSON null NLL
denotes this, not missing observations or zero log loss.

## Paired cases and source-length strata

| Case | True source length | Subgoal | Execution | Whole | Short |
|---|---:|---:|---:|---:|---:|
| 0 | 4 | 1 | 1 | 1 | 1 |
| 1 | 2 | 0 | 0 | .000114784 | 0 |
| 2 | 4 | 1 | 1 | 1 | 1 |
| 3 | 3 | 0 | 0 | .001264143 | 0 |
| 4 | 3 | .656250000 | .656250000 | .651246765 | 1 |
| 5 | 2 | .296228355 | .296409628 | .296890298 | .296296296 |
| 6 | 2 | 0 | 0 | 0 | 0 |
| 7 | 2 | 0 | 0 | .000000023 | 0 |

The four length-two cases average approximately .07406/.07410/.07425/.07407
(subgoal/execution/whole/short). The two length-three cases average
.328125/.328125/.326255/.5. Both length-four cases have empty support and loss1
in every arm. These tiny descriptive strata are not independent claims or selected
subsets for a future rerun. Case3 demonstrates that a shorter consistent program
can predict the target panel well; exact syntax recovery is not required.

## Failure mechanism

Actual intermediate feedback and valid typed syntax did not translate into solving
the observed task. Subgoal proposals fit only 2.08% of completed prefixes. Predicted
intermediate outputs disagree with actual execution in 89 of 768 branch-example
instances (11.59%); even mostly correct local execution predictions are insufficient
to construct the right final behavior. This mismatch is diagnostic, not a rejected
response or an altered subgoal gate.

Subgoal prefixes remain eight syntactically distinct branches at every step in every
case. Execution-only also usually preserves eight distinct paths (exceptions cases5
and7). Thus simple duplicate-syntax collapse does not explain the null. This does
not prove behavioral diversity or good coverage; many different programs can still
make the same wrong predictions.

Case4 supplies essentially the entire short-control improvement in every LLM arm.
Its true Count/Take/Map composition is not recovered as a sufficiently predictive
belief: all LLM arms still assign zero mass to 21 of 32 outputs. The two four-step
Reverse/Scanl/ZipWith and Reverse/ZipWith compositions supply the other 64 unsupported
targets through empty support. Adding intermediate reasoning did not resolve them.

## Actual compute and accounting

| Arm | Calls | Prompt tokens | Completion tokens | Reasoning tokens | Cost USD |
|---|---:|---:|---:|---:|---:|
| Subgoal | 32 | 208823 | 86685 | 79884 | .15087890 |
| Execution | 32 | 189997 | 117503 | 115583 | .18313307 |
| Whole | 32 | 300504 | 143376 | 137736 | .18292282 |

Equal request and token ceilings did not produce equal actual token use. The observed
whole-program advantage is not a precisely compute-matched causal estimate.
Total accepted cost .51693479, below the $3.84 worst-case allocation. Latest
authenticated credits/usage/balance: 245 / 221.179955339 / 23.820044661.
London Sept9 posted spend .72167686; conservative spend .76167686 includes the
unchanged earlier .04 uncertainty; remaining allowance 4.23832314. No credit top-up
inferred or reserve released. No cluster or historical runtime touched.

## Decision

Close this exact prospective stepwise interface. No threshold rescue, retries,
post-hoc case removal, wider decoding, joint screen, or depth sweep on this result.
It is a proposal-construction null, not evidence that non-myopic BED cannot work or
that the methods are equivalent. ExeDec-inspired local decomposition is not the next
default architecture on this evidence; whole-program synthesis is the stronger
observed starting point, although it too lacks adequate support here.

The next dependency should be a zero-call source/opportunity decision about a task
where semantic context plausibly helps LLM hypotheses and real sequential information
structure exists, with a strong symbolic control. Do not spend the remainder on
another undirected random-list-program prompt variant. The full LLM-native non-myopic
research goal remains active and incomplete. Previous turn qualified the instrument;
this turn completed new empirical evidence and independently verified the null.
