# Truth-controlled refresh is not the deployed belief transition

## Verified source finding

Source scripts/number_game_generator_aware_bed.py SHA256
1df1eab2d6e7dbbebb89e31d6c9863160d15ce519c8d9c55724de577a500a35a:
generator_aware_score (line 547) loops over simulated truths, adds each truth to
the generated branch via merge_controlled_support, and then chooses best_query.
evaluate_policy_root (line 608 vicinity) instead chooses the next query from the
generated branch alone. Thus the simulator and deployed continuation differ.
This finding concerns this historical entropy scorer, not every later risk
scorer or the recent fixed-support horizon/transport diagnostics.

Two source-extracted tests, without module initialization or model calls, show:

- Two truth vectors (0,1,0,0) and (0,0,1,0) both answer NO at query 0. The same
  generated singleton (0,0,0,1) is supplied in both cases. Inserting the first
  truth selects query 1; inserting the second selects query 2. A continuation
  conditioned on the same public history and shared policy randomness cannot
  legally depend on which hidden truth produced it.
- Root EIG is zero, deployed singleton continuation EIG is zero, but the
  truth-controlled scorer assigns log(2) future entropy. This is a score mismatch,
  not a measured terminal predictive-risk gap or a universal optimism theorem.

Tests initially failed on missing locations in the AST test harness; fixing
locations yields 2/2 passing. Original source and all banked outcomes untouched.

## Required replacement contract

Keep two distinct objects: a distribution over persistent simulated worlds and
the policy's approximate belief/proposal state. Sample a world once per path.
Its only effect on the policy is through the simulated observed answer. Given
the same history and generator randomness, belief refresh and next action must
be identical across worlds. Never inject sampled truth into the policy support,
even to stabilize entropy scoring or avoid zero likelihood.

At a branch, call the same history-conditioned proposal operator used online,
compile/dedupe with the same rules, and apply the same numerical update. Score
the resulting terminal predictor against the simulated persistent world under
the same proper scoring rule used in evaluation. Reuse branch proposals across
worlds sharing an observed history; this is both cheaper and information-correct.
Include generator randomness in expected continuation risk and pair its seeds
across the depth and compute-matched myopic controls.

Truth-controlled contrastive support can be legitimate for a likelihood-ratio
bound, but that does not justify using it to choose a deployed continuation.
Nor does normalizing likelihoods on an answer-selected hypothesis list establish
an exact Bayesian posterior over all possible programs. With deterministic
membership, repeated consistency filtering is idempotent, so the identified
error is NOT numerical double counting of that label. The remaining issue is
support selection and weighting.

Two honest ways to specify the next method are:

1. A coherent generative model with an explicit prior over programs and a valid
   posterior approximation, including proposal corrections where required.
2. An explicitly approximate, learned history-to-belief operator. Qualify its
   predictive calibration and branch-transition fidelity empirically, and do not
   call a changed-support entropy difference exact Bayesian information gain.

The project's practical candidate is the second, with LLM executable structure
proposals and numerical conditional fitting. Its outer simulation distribution
must itself pass predictive-transfer checks; otherwise it reproduces the recent
missing-rule reversal. No claim of monotonic performance follows automatically
from optimizing a longer lookahead under an inaccurate simulator.

Before paid endpoints, the replacement needs an information-boundary test on
its actual branch runner, semantic proposal validation, and a fresh source-backed
predictive qualification. This audit does not authorize reopening old Number
Game trials or choosing new seeds for a failed interface. No new source has yet
qualified; do not disguise that dependency with another depth sweep.

Previous turn was progress. Current turn identifies a concrete historical
simulator/deployment discrepancy and banks executable counterexamples. Cost $0;
authenticated usage 221.306531939, balance 23.693468061, conservative allowance
4.11174654 unchanged. Goal active and unachieved. No cluster or automation changes.
