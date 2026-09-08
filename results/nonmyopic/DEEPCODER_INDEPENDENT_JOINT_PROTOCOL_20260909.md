# Independent Luna joint-world screen: prospective protocol

Status: scientific protocol and joint-law mechanics frozen before new responses;
paid executor is not yet implemented/qualified. This document alone must not
dispatch calls. Prior results and failed gates remain unchanged.

## Question

Does a second independently seeded, history-aware Luna proposal call improve a
pre-observation executable joint answer/target simulator relative to the first
call and its structural insertion expansion? Does its branch-conditioned target
law predict the relative quality of regeneration and an equal-call repeat control?
This is a prerequisite screen, not a non-myopic endpoint or a powered claim.

## Frozen cases and calls

Use eight new frozen-source programs, seeds 32100000+i for i=0..7; keep constant,
error-producing, duplicate-behavior, and difficult cases. Input seeds are
33100000+100*i+j, j=0..34. First two inputs supply public observations, third is
the fixed query, last 32 are disjoint fixed predictive targets. Do not inspect
query or target answers until their specified opening. No case replacement.

Exact model openai/gpt-5.6-luna, medium reasoning, 16384 total completion tokens,
standard OpenAI route; reuse the banked strict typed grammar and public-history
prompt, no temperature, no fallback. Four calls per case: root A seed
34100000+10*i, root B seed34100000+10*i+1, post-query regeneration and repeat-old-
history both seed34100000+10*i+2. Root A then B; regenerate/repeat order alternates
by case parity. The latter two are call/cap matched, NOT actual-token matched.
All calls request the same bounded program pool. Never inject simulated or actual
hidden programs into prompts. Target input values are not proposal conditioning.

First generate both root pools using only the initial two observations. Apply
the unchanged one-statement local expansion independently, then syntax-dedupe
their union. Construct three teacher laws before the query answer opens:

1. A: first locally expanded root pool.
2. A insertion: unchanged one-stage insertion expansion of A.
3. A+B: union of both locally expanded pools, the designated primary teacher.

All use source syntax-prior weights restricted to their selected support. This
is not full Bayesian inference and does not correct LLM/search selection bias.
Do not introduce uniform reweighting, smoothing, additional expansions, retries,
or adaptive proposal prompts after results. The two-call teacher has more compute
than A; this screen tests a simulator resource choice, not a fair planning win.

Save the complete weighted program/world outputs and public input identities,
hash and validate the pre-answer seal, THEN open the third answer. Regeneration
receives three observations; repeat receives only the original two. Locally
expand each on exactly the history it received, then union with A and condition
on all three observations to produce its endpoint forecast. Also retain A-filter
and A-insertion forecasts. If no compatible programs remain, abstain with loss1.
There is no target-output access until every case/arm forecast is sealed.

## Metrics and gate

For each teacher report actual-answer probability, unsupported-query count,
answer log loss (infinite at zero), conditional target half-Brier, unsupported
target count and log loss. Unsupported answer implies no conditional prediction,
loss1 and infinite target log loss, not exclusion. Report all eight case rows.
Score actual regeneration/repeat endpoints against sealed true targets. Score
the same forecasts using each teacher's branch-conditioned reference and the
proper expected Brier utility, not the forecast's own impurity.

The joint simulator gate passes only if A+B supports all eight actual answers,
has finite conditional target log loss on all256 outputs, and its mean conditional
target Brier is <=0.15 and nonworse than both teacher controls. These are screening
criteria fixed now, not evidence of population calibration. Separately require
at least three regeneration-versus-repeat pairs with absolute actual mean Brier
difference >0.01, and correct teacher-predicted direction for every such pair
(predicted difference must also exceed0.01 in that direction). Fewer informative
pairs is insufficient evidence, not a pass. Report ties and all losses.

A full pass permits only a separately frozen multi-query ranking test. It does
not authorize or establish h1/h2/h3 gains. A failure closes this exact interface;
no seed/cap/gate rescue. Maintain the original requested equal-budget ordinary
receding-horizon goal, compute-matched myopic/random controls and fresh paired
endpoints for any subsequent policy experiment.

## Operational boundary

Maximum32 calls, reserve0.04 per HTTP attempt, full block cap1.28. Re-read exact
authenticated route prices and credits, reserve full exposure, reauthorize just
before each request, retain existing uncertain reservations, and obey the London
$5 account-wide cap. Day rollover, insufficient budget, route/schema failure,
malformed completion, or transport uncertainty stops without retry. Bank the
first failure and no unsealed endpoints. Use the actual current-day ledger, not
a hardcoded date. An interrupted run is not a partial scientific pass.

Before dispatch: implement and test the complete executor, seal-before-answer
and seal-before-target ordering, exact 2/3-history routing and seeds, all32-call
synthetic completion, empty support, gate/null logic, budget race/rollover and
uncertain-cost preservation. Commit and push those bindings. No API calls yet.
