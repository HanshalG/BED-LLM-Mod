# Source pilot verified: useful h2 signal, h3 plateau, no LLM claim

## What completed

The frozen eight-world source engineering panel completed in 243.449 seconds
using the explicitly selected native engine from pushed commit `34579ab0`.
All six deployable controls and three separately labelled population-oracle
controls completed three real measurements in every world: 72 episodes total.
No model calls, paid requests, GPU jobs or source-physics changes were made.

The primary result is banked at
`chembench_horizon_pilot/run-20260908-v3/RESULT.json`, SHA256
`f15d7df1f7281d3d2ca17a1bf504c734ba7c40d57e48ecede2e6e660cd80180f`.
It passes the literal frozen engineering gates. Public h3 and open-loop initial
planning finished in 45.862 and 51.370 seconds under the unchanged 60-second cap.

Replay V2 independently reconstructs the source worlds, common random numbers,
Gaussian likelihoods, posterior weights, fixed-target forecasts, variances and
MSEs, and reexecutes every policy decision with the same qualified planner.
It verifies all 72 episodes, checkpoints, source bindings and aggregate gates.
This is independent physics reconstruction plus same-algorithm policy replay,
not a new independent proof of the approximate quadrature's accuracy.
The first replay exposed a list-versus-tuple comparison bug for JSON open-loop
sequences. That verifier bug was fixed without changing the primary results;
the failed replay remains banked. All five nonrandom policy types now have
JSON-roundtrip replay tests. Kernel/core tests: 112/112; replay tests: 11/11.

## Results

Loss is MSE on 64 fixed targets of log1p source rate, not the earlier location
RMSE metric. SD is across eight source worlds, not standard error.

| Deployable Policy | Mean Final MSE | SD |
|---|---:|---:|
| Myopic h1 | 0.019405 | 0.019841 |
| h2 | 0.010460 | 0.015410 |
| h3 | 0.010460 | 0.015410 |
| Open-loop-lookahead h3, replanning after real data | 0.010868 | 0.015162 |
| Refined myopic | 0.019405 | 0.019841 |
| Random | 0.022810 | 0.024242 |

Mean prior MSE was 0.119214. h2/h3 improve over h1 by 46.098%, with five paired
wins, three ties and no losses. Refined myopic matches h1 on every trajectory,
so this screen's difference is not fixed merely by more accurate one-step
integration. h3 improves over open-loop replanning by 3.753%, but they differ
in only one realized world. This is not strong evidence for a general
anticipated-adaptivity benefit.

The separate population oracle has mean MSE 0.015701 at h1 and 0.001535 at
h2/h3. It knows the finite eight-truth support and is not a deployable baseline
or evidence for an LLM contribution. The gap from the 16-particle public prior
also warns that model approximation matters.

Mean MSE traces at real rounds 1/2/3:

| Policy | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|
| h1 / refined myopic | 0.069299 | 0.020420 | 0.019405 |
| h2 / h3 | 0.069299 | 0.019070 | 0.010460 |
| Open-loop h3 | 0.069299 | 0.017258 | 0.010868 |
| Random | 0.104979 | 0.025095 | 0.022810 |

All per-world actions, MSE/RMSE traces and model variances are saved in
`chembench_horizon_pilot/summary-20260908-v1/per_world_traces.csv`; its JSON
summary includes paired differences and query sequences, and its Markdown
table includes the separately grouped oracle.

## Why this is not the requested headline yet

1. **Depth three has no additional treatment on this setup.** Every public
   h1/h2/h3 root chooses design 1. Once h2 and h3 take that same first action,
   only two measurements remain; both optimize the same two-step policy with
   the same updater and observations. They are therefore identical beyond the
   root, not merely statistically indistinguishable in eight samples. More
   trials with this same fixed prior cannot create a strict h3-over-h2 gain.
   The engineering gate required nonworsening h3, not strict successive gains;
   passing it must not redefine the broader goal.

2. **Most observed gain is sensitive to query ordering/noise coupling.** Four
   of the five wins have the same final query set as h1, in a different order.
   They account for 76.194% of the aggregate MSE reduction. The frozen CRNs are
   indexed by world, round and design, so an experiment performed at a different
   round receives a different noise draw. This is a valid common-random-number
   construction, not an execution bug, but it can make a small paired panel
   look better or worse when the queried set is unchanged. Equal unordered
   observations would give identical final fixed-model posteriors regardless
   of processing order. This caveat does not prove the expected h2 gain is zero;
   it requires an expected-policy-risk/coupling diagnostic before a strong claim.

3. **Eight worlds are an engineering screen.** No powered efficacy conclusion,
   strict depth-monotonicity claim or significance claim follows. All losses and
   controls are reported, including the weak one-world open-loop difference.

4. **No LLM has done any work in this panel.** Structures and parameter particles
   come from a numerical public prior. This verifies the planning instrument,
   not useful model discovery, real-history LLM refresh, or anticipation of
   future LLM computation. The latter stages of the full plan remain incomplete.

## Next work, without reopening this pilot

First use the accessible prior, not new hidden outcomes, to compare full-budget
expected risk of the deployed myopic policy against the h2/h3 policy. This must
evaluate all arms at the same three-measurement endpoint; their currently logged
root values have different horizons and cannot answer that question directly.
Any alternative-noise-coupling analysis must be explicitly retrospective and
cannot replace or rescue the frozen primary result.

For a strict depth-three claim, prospectively define source contexts that leave
a real h2/h3 decision difference: for example more candidate assays than the
real budget and an additional remaining decision after the common first query.
Audit opportunity from public model information before future outcomes, preserve
all controls and numerical accuracy gates, and do not select contexts by observed
wins. Simply rerunning this same prior or increasing its trial count is not a
solution. These are next design questions, not authorization to alter the opened
panel or automatically launch a new sweep.

Then freeze a genuinely useful LLM-proposal interface with executable/schema,
calibration and held-out semantic gates plus productive numerical and
history-blind controls. Do not use an LLM just to choose among the already supplied
four family names or to replace a numerical parameter fitter. No paid block is
authorized by the present report. The overall goal remains active and unfinished.
