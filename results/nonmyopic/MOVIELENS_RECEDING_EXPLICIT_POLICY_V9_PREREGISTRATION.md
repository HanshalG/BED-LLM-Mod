# MovieLens Receding Explicit-Regeneration Policy v9

Date: 2026-07-24

Status: design preregistered; no v9 response or endpoint viewed.

## Question

V7 showed that one explicit LLM belief transition ranks queries better than immediate
EIG. V8 showed that composing two simulated transitions into one value estimate is
not reliable. V9 tests the remaining sequential policy: execute the passed one-step
explicit scorer, observe a real rating, regenerate the belief, and replan with the
same scorer at round 2.

This is receding one-transition planning, not a depth-2 claim. It is non-myopic with
respect to the LLM belief machinery because query value includes the downstream
predictive state induced by regeneration; immediate EIG scores only the current
fixed support.

## Paired Policies

All policies share the same six initial profiles, profile-only likelihoods, candidate
pool, held-out movies, and generated transition whenever they occupy the same state:

- `receding_explicit`: at each round, among the top four current-immediate-EIG
  candidates, enumerate all five ratings, regenerate the support, compute
  profile-only downstream likelihoods, and minimize expected held-out predictive
  entropy;
- `immediate_eig`: choose maximum current immediate EIG at each round;
- `seeded_random`: choose uniformly from the same four-candidate menu using seed
  `24310*1000 + user_id + round`.

Round 1 uses one common four-candidate tree per user. After all first-round trees are
frozen, each policy reads its selected recorded rating. At round 2, one common
four-candidate terminal tree is generated for every unique resulting policy state;
policies in the same state reuse it exactly. Only then are second ratings read and
final paths fixed. Held-out ratings enter only the final NLL calculation.

Every transition uses the unchanged v7 representation: six regenerated profiles plus
the two old profiles most compatible with the observed rating, with uniform support
and history-free GPT-5.4 Mini likelihoods. No v8 depth-two score, parser-recovered
response, fitted temperature, MI correction, risk penalty, or lineage weight is used.

The parser policy is frozen prospectively from the v8 serving audit. Structural
trailing commas are removed only outside strings immediately before `}` or `]`.
Every finite nonnegative five-number likelihood row with total in `[.90,1.10]` is
normalized and counted; any row outside that interval fails closed. This avoids a
post-endpoint recovery and does not alter relative probabilities within a row.

## Fresh Cohort

The history remains *Star Wars*, *Fargo*, *Toy Story*, and *The Silence of the
Lambs*. Every v1--v8 user is excluded. Exactly 50 untouched eligible users remain.
Seed `24310` freezes smoke users `684,643` and this ordered 48-user formal screen:

`880,94,868,745,301,246,59,308,184,216,465,632,497,339,121,715,295,312,457,287,`
`913,664,458,1,23,97,561,493,311,292,806,249,896,77,274,435,344,682,363,593,883,`
`514,222,535,343,805,330,44`.

The first eight users with maximum initial immediate EIG at least `.02` are enrolled
before candidate or held-out outcomes. Fewer than eight is a futility stop.

## Frozen Gates

The formal gate passes only if:

- all eight users enroll and every policy path has two distinct queries;
- reasoning-token usage is zero;
- physical requests equal `416 + 40U`, where `U` is the recorded number of unique
  round-2 policy states: 96 screening calls plus 320 common round-1 tree calls plus
  40 calls per unique round-2 state;
- receding explicit lowers mean final held-out NLL by at least `.03` versus immediate
  EIG and wins on at least 5/8 users;
- receding explicit mean final NLL is no worse than seeded random.

Round-1 and final NLL traces, action overlap, and score margins are secondary. This is
an eight-user mechanism-policy gate; passage authorizes a larger confirmation under a
new public-history cohort, while failure closes the MovieLens policy line.

## Cost And Smoke

A one-user, one-candidate-per-round interface smoke uses exactly 22 requests:
two initial calls and ten branch calls at each round. Full formal cost depends on
policy-state overlap: `U` lies in `[8,24]`, so requests lie in `[736,1376]`.
The run cap is `$9.00`, projected cost `$7.00`, and concurrency `64`. Live OpenRouter
and ledger balances must be checked before both paid stages.
