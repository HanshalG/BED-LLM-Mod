# Luna-medium semantic/history screen: context pass, discovery-update null

Frozen protocol28fe71dc and executor7a5ba203. All16requests completed cleanly, cost
$.0437314, no retries or new uncertain charges. Both paid process and separate
source/request/forecast/endpoint replay exited successfully. Replay verifies exact
payloads, source-derived histories and targets, all decoded pools, weighting,
scores and accepted receipts. Forecast SHA256
630a1eda7068e0087cbec526da4b929f3bb125efd1e4f13e9123e98882b54a27.

## Results

Mean squared error of independent noisy log responses; lower is better. All four
tasks retained, no empty final arm. These are source-function experiments on the
predeclared code-valid box, not validated physical measurements.

| Task | Initial semantic | Initial blind | Refreshed on new observation | Equal-budget redraw | Symbolic log-linear |
|---|---:|---:|---:|---:|---:|
| 103 | .001936 | .040138 | .001936 | .002138 | .002824 |
| 458 | .002784 | 5.174954 | .002784 | .002784 | .106943 |
| 457 | 1.127190 | .216062 | 1.127190 | .640294 | .770248 |
| 653 | .002348 | .997149 | .002348 | .002348 | .329800 |
| Mean | .283565 | 1.607076 | .283565 | .161891 | .302454 |

Context gate passes: approximately82.4% aggregate reduction and3/4 task wins>.01.
History gate fails:0/4 wins>.01 and refreshed loss is approximately75.2% worse than
redraw. Both final arms numerically condition on the same4observations; only the
proposal-generation context differs. Therefore withholding the new observation
from the redraw posterior is not the explanation. Joint gatefalse, depth not
authorized. This is a four-task development diagnostic, not a significance claim.

## Mechanism diagnosis

The three context-successful tasks already predict near the observation-noise
variance(.0025) from their initial semantic proposals. Their extra observation
produces little useful proposal improvement. This is evidence for useful semantic
prior knowledge, not yet observation-enabled discovery or a non-myopic advantage.

On polygon collision457, semantic context gives a wrong structural formula and is
worse than blind generation. The refreshed proposal interpolates the observed
N values4,6,7,11 with a cubic inside a square root. On four predeclared target inputs
with N=3 it has an invalid square-root domain. The frozen all-public-input validity
filter excludes it before scoring, leaving only the original incorrect formula.
The equal-budget redraw yields another valid formula and improves the final pool.
No filtering rule was changed after this result. Target INPUTS were public to the
numerical evaluator, never target labels; the LLM did not see the target panel.

This links the null to both saturation on easy semantic cases and failure to propose
a globally valid improvement on the hard case. It does not prove equivalence of
refresh/redraw, that every extra observation is useless, or that non-myopic BED is
impossible. The restricted proposal weights are not independently calibrated joint
world probabilities.

## Decision

Close this exact semantic/history interface. No depth sweep, more samples, new seed,
removal of457, or altered gate can rescue it. Preserve the positive context result
as a qualified development observation, not the requested headline.

The next zero-call analysis should measure whether the hard-case failure is solely
proposal validity or also wrong ranking/weighting on already valid candidates, using
only banked responses. A genuinely new observation-conditioned proposal design
would need executable-domain checks in the public feedback and an explicit test
that new observations outperform the same feedback without new observations.
Do not launch that design until the banked error decomposition supports it, and
do not assume domain repair alone creates non-myopic opportunity. No future endpoint
or paid call is authorized by this null.

## Accounting and verification

Thirty focused tests1.25s before launch, including shared budget/uncertainty handling,
full16-call fake completion, sealing, request-tamper replay rejection, strict schema,
empty-support and positive/null gates. Lint passes. Exact scientific code unchanged
after launch. Raw requests/responses/routes/forecasts/outcomes are banked in
physgym_semantic_history_20260909/. Replay returned replay_valid,16calls,.0437314cost.

Latest authenticated credits/usage/balance245/221.218891739/23.781108261; posted
usage had not yet caught up with all accepted receipts. Conservative London-day
spend.80540826 and remaining4.19459174 include full locally accepted cost and the
older.04 uncertainty, which is not released. No new uncertainty; no cluster,
automation or protected-runtime changes. Previous turn froze tested mechanics;
this turn completes new paid evidence. Full research goal remains active/unachieved.
