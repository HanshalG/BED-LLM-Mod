# Matched-objective myopic selection already chooses B

Frozen reconstruction1a421bfa restored upstream revision
33b7fa9df96de9c35744efd181ca7e5a8dd60ad5 from the public
[DiscoverPhysics repository](https://github.com/SampsonML/DiscoverPhysics).
Only the exact eight saved agent hypotheses were compiled and simulated. No new
LLM branches or actual physical endpoint maps were run. The numerical-helper
loader bypassed the agent package and made endpoint-generating functions
unavailable in its helper namespace. Every simulated source map was checked
against the eight candidate maps before each trajectory. All216candidate
trajectories completed; the process exited normally.

Reconstruction reproduces all four saved immediate-EIG scores with maximum
absolute error4.44e-16. This is strong observation-model replay evidence, not an
independent reference for every target trajectory. Code/source/model/cache hashes
are recorded in discoverphysics_candidate_reconstruction_20260909/RESULT.json.

## Same-objective root comparison

Expected target-feature MSE after ONE observation, on the frozen initial model:

| Root | One-step prediction risk (order128) |
|---|---:|
| A |2.1265516431|
| B |**1.4696862112**|
| C |2.9427129647|
| D |1.8657881773|

All quadrature orders16,32,64,128 select B. Maximum risk change between64 and128
is6.29e-7, below the1e-4 diagnostic tolerance. This is deterministic predictive
integration conditional on the original eight hypotheses, not held-out efficacy.

Immediate EIG had selected D; terminal-MSE lookahead selected B. But myopic
prediction-risk selection also selects B. Thus the original B-D root contrast
does not demonstrate that lookahead was necessary. Objective alignment alone
recovers that root choice in this example. This resolves the specific ambiguity
identified by the source audit; it does not establish that every deeper policy
or continuation is equivalent to myopic replanning.

Do not compare the one-observation risk1.4697 directly against the old two-query
risk.4955 as evidence of depth advantage: those have different observation
budgets. Do not treat this retrospective calculation as a new physical endpoint
or rescue the closed LLM-support claim. The old12.416% B-D endpoint gain and
failed same-root support increment remain numerically unchanged; their proper
interpretation is a complete-policy contrast, not isolated horizon efficacy.

## Consequence for the main goal

This candidate is weaker evidence for the intended contribution than previously
stated. A future study must use the SAME terminal predictive objective, physical
budget, belief update and initial computational information for every horizon,
and include productive myopic computation and random controls. In particular,
an immediate-EIG baseline alone cannot establish a non-myopic prediction result.

Keep the reconstructed candidate cache as a supporting diagnostic. Do not launch
another dark-matter generation or tune its regional weights, horizon, roots or
endpoints. Return the next scientific selection to useful LLM structural
discovery plus independently established same-objective horizon opportunity.
This turn contributes evidence that closes an attribution question, not a new
positive result. Previous turn was progress; full goal active/unachieved.

No paid calls/cost. Authenticated balance23.693468061 and conservative London-day
remaining4.11174654 unchanged. Upstream checkout is local external tooling, not
included in the project commit; no dependencies installed or cluster used.
