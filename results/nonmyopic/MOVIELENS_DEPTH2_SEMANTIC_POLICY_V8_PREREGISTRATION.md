# MovieLens Depth-2 Semantic Rollout Policy v8

Date: 2026-07-24

Status: design preregistered; no v8 response or endpoint viewed.

## Question

V7 showed that an explicit LLM belief-transition rollout ranks one-step queries better
than immediate EIG. V8 asks the stricter planning-horizon question: does evaluating
the terminal belief after a branch-specific second query improve a two-round policy
over both the same model at depth 1 and greedy immediate EIG?

## Common Transition Tree

All policies start from the same six profiles and profile-only likelihood matrix over
16 legal candidates and eight held-out movies. The first-query menu is the four
highest-immediate-EIG candidates.

For every first candidate and rating 1--5, the runner:

1. appends the hypothetical rating to the public history;
2. regenerates six profiles and retains the two compatible old profiles;
3. obtains profile-only likelihoods for the remaining candidates and held-out movies;
4. selects the branch's second query by immediate EIG;
5. enumerates ratings 1--5 for that second query, regenerates the support again, and
   obtains profile-only held-out likelihoods.

All 240 calls per enrolled user are completed before any candidate outcome is read.
The three policies then traverse this same frozen tree:

- `depth2`: choose the first query with minimum expected terminal entropy after the
  branch-specific greedy second query;
- `depth1`: choose the first query with minimum expected entropy immediately after
  the first transition, then use greedy EIG at round 2;
- `immediate_eig`: choose greedy EIG at both rounds.

This common tree removes semantic-generation noise between policies. The depth-2
score is

`E_r1 E_r2[H(heldout predictions after q1, r1, q2*(r1), r2)]`,

where `q2*(r1)` is the immediate-EIG query in that first-outcome branch. Depth 1
omits the second expectation. Likelihood prompts receive profiles and movie metadata
only, never rating history.

## Fresh Cohort

The public history remains *Star Wars*, *Fargo*, *Toy Story*, and *The Silence of
the Lambs*. Every v1--v7 user is excluded. Of 100 eligible untouched users, seed
`24309` freezes smoke users `250,870` and this ordered 48-user formal screen:

`455,933,49,395,606,620,18,887,503,916,109,630,262,804,72,276,99,230,660,889,`
`658,622,291,786,198,478,773,380,618,738,790,374,751,830,41,92,727,886,177,65,`
`892,151,256,903,174,757,763,58`.

The first four formal users with maximum initial immediate EIG at least `.02` are
enrolled before outcomes. Fewer than four is a futility stop.

## Frozen Gates

The formal gate passes only if:

- all four users enroll and all policy paths contain two distinct queries;
- physical requests equal exactly `1,056`: 96 screening plus 960 tree calls;
- reasoning-token usage is zero;
- depth 2 lowers mean final held-out NLL by at least `.03` versus depth 1 and wins
  on at least 3/4 users;
- depth 2 lowers mean final held-out NLL by at least `.03` versus immediate EIG and
  wins on at least 3/4 users.

Final held-out NLL is computed only after every policy path is fixed. This is a
four-user mechanism gate; passage authorizes a larger fresh-cohort confirmation and
is not itself a powered paper claim.

## Cost And Smoke

The exact formal count is 1,056 requests. The run cap is `$8.00`, projected cost
`$6.00`, and concurrency `64`. A one-user, one-first-candidate interface smoke
enumerates five first outcomes and five second outcomes per branch in exactly 62
requests. Live OpenRouter and ledger balances must be checked before both stages.
