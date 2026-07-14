# MediQ Step 0 Initial Smoke Manual Review

Run: `20260714T033819_mediq-step0-eig-nonthinking-26b-seed1304`

Implementation commit: `7825898`

Verdict: **AUTOMATED PASS, MANUAL FAIL. Do not use as policy evidence and do not
advance to a claims run.**

## Automated result

- Five official usable iMEDQA cases, two rounds each.
- 9/10 clean response mappings (90%), above the frozen 85% threshold.
- 10/10 replies assembled verbatim from released atomic facts.
- 10/10 replies judged relevant by the automatic judge after bounded repair.
- Zero structured terminal failures, patient relevance failures, runtime errors,
  forced-thinking exits, or target-leaking queries.
- All 50 root finite-target likelihood tables and EIG values independently revalidated.
- 235 requests, 70,115 prompt tokens, 18,128 completion tokens, no reasoning,
  $0.01416921, 208.93 seconds.

The five-case endpoint accuracy was 5/5 but is smoke-only and not evidence of policy
quality. Several cases were already highly concentrated on the correct answer before
questioning.

## Manual failures

1. **Case 0, turn 1: compound query and unsupported inference.** The selected query
   combined travel with unprotected sex/new partners. The only returned fact was
   "Patient is sexually active," which does not establish new partners or unprotected
   sex. The mapper nevertheless selected "Yes, new sexual partners." The reply is
   verbatim but not an explicit answer to the qualified question.
2. **Case 2, turn 2: outcome requires unsupported frequency.** The record says the
   patient feels increasingly hopeless and has diminished interest, but gives no
   daily-versus-occasional frequency. The mapper selected "Yes, occasionally." This is
   an inference rather than a clean category match.
3. **Case 4, turn 1: non-exhaustive numeric support.** The generated glucose outcomes
   were `>500 mg/dL`, `<50 mg/dL`, normal, and unavailable. The actual 450 mg/dL result
   fell into a gap and correctly remained unmapped.
4. **Candidate support overlap.** Several candidate sets contained both "Not recorded"
   and "Information unavailable / not in record," which are overlapping categories.
5. **Atomicity is not enforced.** Candidate questions repeatedly joined variables with
   `and` or `or`, including blood pressure plus heart rate and history of diagnosis plus
   medication. These produce ambiguous patient and branch outcomes.

## Scientific read

The finite target and EIG arithmetic are functioning. The broken link is now localized:
the generated observation space is not always a valid partition of patient responses,
and the relevance/mapping judge can infer qualifiers absent from the selected facts.
That corrupts both deployed Bayesian updates and depth-two synthetic branches. More
rollouts or deeper planning would amplify this error; they cannot repair it.

Required repair before one repeated smoke:

- enforce one-variable atomic questions;
- canonicalize exactly one unavailable outcome;
- validate candidate outcome sets for mutual exclusivity and exhaustiveness;
- judge fact relevance separately from outcome mapping, using explicit entailment only;
- strengthen Fact-Select instructions against implied qualifiers.
