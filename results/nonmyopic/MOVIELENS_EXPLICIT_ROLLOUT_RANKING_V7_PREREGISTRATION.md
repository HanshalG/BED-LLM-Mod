# MovieLens Explicit Regeneration-Rollout Ranking v7

Date: 2026-07-24

Status: design preregistered; no v7 response or endpoint viewed.

## Intervention

V6's one-pass verbal value judgment ranked realized semantic transitions backwards.
V7 replaces judgment with explicit model-aware lookahead. For every enrolled user and
candidate query:

1. use the current profile-conditioned likelihood to assign probabilities to ratings
   1--5;
2. append each hypothetical rating to history;
3. regenerate six profiles with Gemma and retain the two compatible old profiles;
4. use history-free GPT Mini likelihoods to predict the eight downstream movies;
5. compute mean downstream predictive entropy for that hypothetical support;
6. average entropy over the five outcomes under the current predictive distribution.

Candidates are ranked by lower expected downstream entropy. This is a one-step action
whose value includes the induced next semantic belief state, exactly the transition
that fixed-support immediate EIG omits.

## Fresh Cohort

To restore ample untouched population, the public history changes target-blindly to
four popular, genre-diverse movies: *Star Wars*, *Fargo*, *Toy Story*, and *The Silence
of the Lambs*. Every prior v1--v6 user is excluded. Of 122 eligible users, seed `24308`
freezes smoke users `253,654` and an ordered 20-user formal screen:

`201,749,454,910,407,325,248,929,747,305,313,553,429,96,399,577,64,263,5,389`.

As before, 16 legal candidates use rating presence/popularity only and eight held-out
movies are selected from the remainder. The first four formal users with maximum
immediate EIG at least `.02` are enrolled before outcomes; fewer than four stops.

## Gates And Cost

Immediate EIG and seeded random are controls. Passage requires explicit-rollout score
Spearman with negative realized branch NLL at least `.25`, mean top-1 regret at least
`.02` lower than immediate EIG, wins on at least 2/4 users, and regret no worse than
random. No policy is authorized otherwise.

Full execution is projected at 232 requests: 40 screening, 160 hypothetical
regeneration/likelihood, and 32 realized branch requests. A small interface smoke must
pass before formal. Run cap `$3.00`, projected cost `$1.50`, concurrency `64`; live and
ledger balances are checked before every paid stage.
