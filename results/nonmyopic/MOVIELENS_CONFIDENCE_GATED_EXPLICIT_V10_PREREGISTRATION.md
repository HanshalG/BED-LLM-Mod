# MovieLens Confidence-Gated Explicit Rollout V10 Preregistration

Date: 2026-07-24

## Motivation and Status

This is a transparent post-hoc development followed by one fresh confirmation.
V7 showed that explicit LLM belief regeneration improved four-candidate query
ranking on 4/4 users. V9 then failed its two-round policy endpoint, although
the explicit score retained positive rank association with realized round-one
NLL.

On the closed eight-user V9 records, the exact target-blind rule

> use the explicit choice only when its predicted downstream-entropy advantage
> over the immediate-EIG choice is at least `.02`; otherwise use EIG

would activate on 4/8 users and improve round-one held-out NLL by `.03636` on
average over all eight. That observation is development only. The `.02`
threshold is now frozen for a wholly fresh user cohort.

## Fresh Cohort

Seed `24326` selects users only by rating presence after excluding every user
named in V1-V9. A new four-film history is fixed:

- `Contact (1997)`;
- `Liar Liar (1997)`;
- `The English Patient (1996)`; and
- `Scream (1996)`.

Two users are reserved for a serving smoke. Forty-four untouched users form the
formal sensitivity screen. The first 16 in frozen order whose maximum
immediate EIG is at least `.02` are enrolled before any candidate or held-out
rating is read. Failure to enroll 16 stops after the 88 screening calls.

## Shared Tree and Selectors

The apparatus is unchanged from V7:

1. Gemma 4 26B A4B nonreasoning generates six semantic preference profiles
   from the four-film history.
2. GPT-5.4 Mini nonreasoning supplies profile-only rating likelihoods for a
   16-film candidate pool and eight held-out films.
3. The four highest immediate-EIG candidates enter one shared tree.
4. For each candidate and ratings 1-5, Gemma regenerates the profile support
   and Mini predicts the held-out likelihoods.
5. The explicit score is negative expected downstream predictive entropy.
6. Only after all four scores are frozen are the four recorded candidate
   ratings read and all four realized branch NLLs computed.

Immediate EIG always chooses branch 0. The V10 selector uses the explicit
argmax only when its score exceeds branch 0 by at least `.02`; otherwise it
chooses branch 0. Thus inactive users are exact paired ties, not discarded.

This is one-round non-myopic query selection over the LLM's path-dependent
profile regeneration. It is not a two-round policy confirmation.

## Gates and Cost

The one-user interface smoke is exactly 12 calls and has no efficacy endpoint.

Conditional formal count is exactly 856 calls:

- 88 screening calls;
- 640 hypothetical profile/likelihood calls for 16 users; and
- 128 realized-branch profile/likelihood calls.

Projected cost is `$3.60`, hard cap `$7.00`. All formal gates are conjunctive:

1. exactly 856 requests, zero reasoning, and all 16 users enrolled;
2. confidence gate active on at least five users;
3. mean held-out NLL improvement over immediate EIG at least `.015`;
4. paired user-level 90% bootstrap lower bound above zero;
5. wins exceed losses by at least two;
6. mean improvement among active users at least `.03`; and
7. mean selected top-one regret no worse than seeded random.

No user, history movie, margin, threshold, or endpoint may be replaced or
changed after responses. Failure closes this confidence-gated one-round route.
