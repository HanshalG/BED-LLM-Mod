# MovieLens Semantic-Profile Dynamics Gate

Date: 2026-07-24

Status: preregistered mechanism gate. This is not a policy or depth comparison.

## Motivation

BED-LLM evaluates preference elicitation with generated free-text user profiles,
multiple-choice questions, an LLM answerer, and an LLM-as-judge recommendation score.
This gate keeps the open semantic profile and likelihood model but replaces both outcome
and evaluation with recorded MovieLens ratings. It asks whether observing one rating
changes the generated profile support in a way that improves held-out predictive log
loss, and whether one-step EIG on the current support identifies that best update.

Sources:

- BED-LLM: https://arxiv.org/abs/2508.21184
- MovieLens 100K: https://grouplens.org/datasets/movielens/100k/
- MovieLens citation: F. Maxwell Harper and Joseph A. Konstan, *The MovieLens
  Datasets: History and Context*, 2015.
- Download SHA-256:
  `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`
- `u.data` SHA-256:
  `06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490`
- `u.item` SHA-256:
  `553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701`

The dataset is used locally under its research terms and is not redistributed.
Persisted result artifacts omit every source rating and held-out item list; they retain
only derived model outputs and metrics, fixed public query metadata, file hashes, seeds,
and anonymous user IDs needed to replay against an authorized local copy.

## Frozen Population And Splits

Every eligible user rated the same eight globally fixed films. Eligibility uses only
rating presence, never rating values. NumPy seed `24302` selected 12 of 47 eligible
anonymous users:

`13, 62, 141, 213, 422, 447, 479, 552, 588, 592, 655, 919`.

Four fixed ratings form the public history:

1. *Star Wars* (1977);
2. *Fargo* (1996);
3. *Liar Liar* (1997);
4. *The English Patient* (1996).

Four fixed legal rating queries are:

1. *Contact* (1997);
2. *Return of the Jedi* (1983);
3. *Scream* (1996);
4. *Toy Story* (1995).

For each user, seed `24302000 + user_id` selects eight other rated films as a
held-out endpoint. Held-out titles and genres may enter likelihood prompts, but their
ratings never enter any model prompt. Anonymous user IDs never enter model prompts.

## Models And Beliefs

- Non-thinking `google/gemma-4-26b-a4b-it` generates six distinct semantic
  preference-profile hypotheses from the observed rating history.
- Non-reasoning `openai/gpt-5.4-mini` returns five-way probabilities for ratings
  `1,2,3,4,5` conditional on a profile, observed history, and movie metadata.
- After each realized candidate rating, Gemma generates six refreshed profiles.
  The branch support adds the two old profiles assigning the highest probability to
  that observed rating, with exact-text deduplication.
- The empirical belief is uniform over the resulting support, following BED-LLM's
  sample-filter-uniform construction.
- All rating outcomes and held-out endpoints come from `u.data`; there is no LLM
  answerer or LLM judge.

OpenRouter project-ledger ceiling: `$70.38480269545715`. Per-run cap: `$0.75`;
projected reservation: `$0.30`; concurrency: `64`.

## Serving Gate

The first two frozen users run one candidate branch each:

- two initial-profile requests;
- two initial-likelihood requests;
- two profile-refresh requests;
- two branch-likelihood requests;
- two exact refresh-prompt replay requests.

Passage requires exactly 10 physical requests, zero reasoning tokens, valid schemas,
two complete branches, and two complete replay profile sets. It measures interface
mechanics only.

## Formal Mechanism Gate

The formal run has exactly 120 physical requests:

- 12 initial-profile requests;
- 12 initial-likelihood requests;
- 48 realized profile-refresh requests;
- 48 branch-likelihood requests.

For each user, initial and branch quality are mean negative log likelihood in nats on
the same eight held-out recorded ratings. One-step EIG is computed on the initial
profile support for each of the four candidate movies. Lower held-out NLL is better.

All conditions must pass:

1. all 12 users and 48 branches complete with exactly 120 requests and zero
   reasoning tokens;
2. mean oracle best-branch held-out NLL improvement is at least `0.05`;
3. at least 6/12 users have oracle improvement at least `0.05`;
4. at least 6/12 users have branch NLL spread at least `0.10`;
5. mean held-out NLL regret of the immediate-EIG choice is at least `0.03`;
6. at least 4/12 users have immediate-EIG regret at least `0.05`;
7. semantic profiles measurably affect rating likelihoods: mean maximum candidate
   EIG is at least `0.02` nats and at least 8/12 users have maximum candidate EIG
   at least `0.02`.

Passage establishes a path-dependent semantic-belief opportunity and authorizes only
a fresh target-blind scorer/ranking-fidelity gate. Failure closes this exact
MovieLens profile apparatus before policy or depth evaluation. No threshold tuning,
post-hoc user subset, or alternate held-out split is allowed after responses.
