# Countries CA-BED Aligned Semantic V1 Preregistration

Date frozen: 2026-07-24

## Question

Can depth-two Bayesian planning outperform a matched-compute myopic root policy
inside a coherent LLM-native semantic environment?

This is a fresh entity domain, not another Animals repair. The LLM generates
open-ended country questions and defines the complete country-by-answer
function. Exact code performs only finite Bayesian arithmetic over that learned
semantic world.

## Support and Semantic Environment

- Fixed uniform support of 64 named countries spanning regions and development
  levels.
- Full GPT-5.4 generates candidate factual Yes/No questions.
- GPT-5.4 Mini labels every one of the 64 countries Yes or No for each question
  in one target-blind table.
- The same table defines both the deterministic observation and the likelihood.
  There is no separate answerer and therefore no likelihood/environment mismatch.
- Every tree evaluates all 64 possible hidden countries exactly.
- Direct country guesses, duplicate questions, malformed rows, incomplete
  tables, and content repair are forbidden.
- Models use no reasoning. Question temperature is `.7`; table temperature is
  zero.

Each tree contains:

- eight proposed roots, from which the first four with at least four countries
  on each side are retained;
- three retained follow-ups from six proposals for each root's Yes and No
  branch; and
- one complete 64-country semantic table for each retained question.

This requires 41 physical requests per tree and 1,792 retained semantic
country/question relations. The LLM role is load-bearing: without generated
questions and semantic tables, the planner has neither actions nor likelihoods.

## Policies

All policies share the exact same generated tree and semantic tables:

1. `depth_two`: select the root maximizing immediate entropy reduction plus the
   expected best branch-specific follow-up reduction;
2. `depth_one`: select the root with maximum immediate entropy reduction, then
   use the same best branch-specific follow-up; and
3. `random_root`: select a seeded random root, then use the same best
   branch-specific follow-up.

The depth-one control receives identical generation calls, tables, width, and
follow-up optimization. It differs only in root scoring.

Endpoints are exact mean final entropy and truth NLL over all 64 targets. They
are numerically equal under a deterministic uniform partition, but both are
recorded at target level. Per-target wins/losses and selected questions are
public.

## Serving and Structural Smoke

Two fresh style prompts generate two complete trees: exactly 82 requests,
projected `$0.25`, hard cap `$1.00`.

The smoke passes only if:

1. both trees and all semantic tables complete;
2. request count is exactly 82 and reasoning tokens are zero;
3. at least one tree selects a different depth-two root;
4. at least one tree has strictly positive mean depth-two gain over depth one;
   and
5. mean gain across the two trees is at least `.01` nat.

Failure closes this exact country support, model pair, widths, and interface.
Smoke targets cannot enter the formal endpoint.

## Formal Paired Confirmation

Conditional on smoke passage, twelve fresh style prompts generate twelve new
trees: exactly 492 requests, projected `$1.50`, hard cap `$4.00`.

All gates are conjunctive:

1. all twelve trees and tables complete with exact requests and zero reasoning;
2. depth two selects a different root on at least 8/12 trees;
3. at least 8/12 tree-level mean gains over depth one are strictly positive;
4. mean final-entropy/truth-NLL gain over depth one is at least `.02` nat with a
   paired 90% bootstrap lower bound above zero; and
5. mean gain over the stronger random-root-with-optimal-follow-up control is at
   least `.02` nat with a positive lower bound.

No target, tree, or question may be replaced. Failure is reported as a null.
Passage would establish non-myopic planning over an LLM's own coherent semantic
likelihood model. It would not establish external factual accuracy or
path-dependent hypothesis regeneration.
