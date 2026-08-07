# RegretBench Endpoint Alignment And Matcher Amendment

Date: 2026-08-07

**Status: prospective, implemented, unopened. Model calls: 0. Cost: $0.**

## Problems Found

### Terminal transition mismatch

The dynamic scorer regenerated an LLM support after simulated question one,
selected question two by EIG on that support, and predicted terminal Brier by
conditioning its generated reply likelihoods. Realized execution instead used
a new LLM support generated after question two as the primary endpoint.

Those are different state transitions. Even a correctly ranked planner could
look uncorrelated with the old endpoint because an additional unmodelled LLM
redraw intervened after the action it scored.

### Equal-length lexical false positives

The shared conservative lexical matcher selected shorter and longer strings
with independent `min(..., key=len)` and `max(..., key=len)` calls. On a length
tie both calls returned the first string. Therefore different equal-length
strings such as `binary 1` and `binary 0` incorrectly matched.

This could inflate support-recovery truth coverage, generated reply
matchability, terminal truth mass, and every downstream Brier. No RegretBench
response or endpoint had been opened when either issue was found.

## Prospective Fix

The primary terminal endpoint now exactly mirrors the rollout scorer:

1. execute the selected question one and generate the answer-conditioned LLM
   support;
2. choose question two by EIG on that support;
3. obtain the exact official environment reply;
4. assign likelihood one to hypotheses whose generated reply has the same
   lowercase alphanumeric normalization as the exact reply and zero to all
   others; and
5. report normalized hidden-answer truth mass, Brier, and log loss within that
   outcome group.

No represented reply gives truth mass zero. Primary mechanics require at least
40 represented exact second replies for every primary policy. The exact-10
enriched smoke now conditions each branch on the actual official first reply,
requires all three follow-up questions to be supported, and requires every
exact second reply to occur in its generated likelihood partition. It still
opens no answer-alias truth mass or efficacy endpoint.

The fresh support generated after question two remains fully recorded as a
secondary robustness endpoint: fresh truth mass, Brier, log loss, and coverage.
It cannot pass, rescue, veto, or reclassify the aligned primary result.

Luna's free-form naive question two is not one of the first support's four
likelihood-aligned questions. The naive baseline is therefore compared only on
the shared fresh-regeneration endpoint and remains descriptive.

The matcher now returns false immediately for unequal equal-length normalized
strings. Substring matching is considered only when one string is strictly
shorter, has at least two tokens and eight characters, and is contained in the
longer string.

## Verification

- Hand-built posterior: four hypotheses match an observed reply and one matches
  truth, producing exact truth mass `0.25` and Brier `0.5625`.
- Unrepresented reply: mass `0`, Brier `1`.
- Regression cases `binary 1` versus `binary 0` and `New York` versus
  `Newark X` no longer match.
- An adversarial fresh-regeneration reversal leaves every primary scientific
  gate unchanged and appears only in descriptive fresh comparisons.
- Exact-scale synthetic execution traverses smoke, all `8,256` planning calls,
  realized aligned endpoints, fresh secondary endpoints, mechanics, bootstrap,
  privacy, and serialization.
- Support-recovery daily and dynamic policy protocol checks both refuse a
  changed matcher-bearing core hash before requests.

The complete focused source/support/policy/budget suite passes `69/69`.

Tasks, model roles, prompts, request counts, CRN, policies, scientific
thresholds, concurrency, and budget caps are unchanged. This is a pre-response
validity repair, not scientific evidence.
