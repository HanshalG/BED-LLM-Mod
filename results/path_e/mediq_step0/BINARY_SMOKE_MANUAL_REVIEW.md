# MediQ Step 0 Canonical-Binary Smoke Manual Review

Run: `20260714T041055_mediq-step0-binary-eig-nonthinking-26b-seed1304`

Implementation commit: `d8a6d78`

Verdict: **AUTOMATED PASS, MANUAL FAIL. Diagnostic only; do not advance to a
claims run.**

## Automated result

- Five hash-pinned usable iMEDQA cases and ten realized turns.
- 10/10 clean mappings, 10/10 verbatim grounded replies, and 10/10 final relevance.
- All 50 candidates used the canonical `Yes` / `No` / unavailable support and passed
  the logged structural critic.
- Zero structured terminal failures, candidate-validation failures, patient-relevance
  failures, runtime errors, reasoning tokens, or forced exits.
- 288 requests, 79,860 prompt tokens, 15,966 completion tokens, $0.01408867, and
  251.75 seconds.
- Endpoint accuracy was 4/5, but remains smoke-only and is not policy evidence.

## Manual failures

1. **Target-decode leakage in case 0, turn 2.** The benchmark target asks which
   medication was given. The selected query asks whether a third-generation
   cephalosporin was given; the four candidates enumerate drug classes or mechanisms
   corresponding closely to the answer options. These are disguised decodes, not
   patient-evidence experiments. The patient correctly returned unavailable, but the
   action itself violates the BED target-decode contract.
2. **Semantic repeat in case 2.** Turn 1 asks whether the patient "reports excessive
   worry" and receives unavailable. Turn 2 asks whether the patient "experiences
   excessive worry." Exact-string deduplication misses the paraphrase, so the policy
   spends a second turn on the same unavailable variable.
3. **Derived rather than explicitly observable predicates.** Unselected candidates
   include "Is the patient hemodynamically stable?" Although clinically meaningful,
   no atomic fact states stability; answering requires interpretation across vitals.
   The valid contract is an explicit observable or numeric-threshold predicate.
4. **Management-state predicates.** "Has a urinalysis already been performed?" appears
   when the target asks for the next management step and one option is to obtain a
   urinalysis. This asks about the decision process rather than pre-decision patient
   evidence and can leak the target structure.

After this audit, the independent analyzer was hardened with the same deterministic
action-contract and semantic-repeat checks. Its retrospective report
`BINARY_REPORT_HARDENED.json` rejects all five treatment-class candidates in case 0,
the repeated worry query in case 2, the derived stability query, and the urinalysis
management-state query. This confirms that the failures are machine-detectable rather
than relying only on subjective transcript review.

## What improved

The canonical binary support fixed every failure from the prior smoke: no compound
queries, duplicated unavailable categories, numeric gaps, unsupported frequency
mappings, or category-assisted relevance inference remained. Direct numeric comparisons
were handled correctly, including glucose 450 mg/dL versus a 250 mg/dL threshold and
blood pressure 80/55 versus a 90 mmHg systolic threshold.

## Required final action-support repair

- Reject current-treatment, diagnosis, test-order/status, or management queries that
  reveal or paraphrase the benchmark target instead of measuring patient evidence.
- Reject semantic paraphrases of prior queries, including prior unavailable answers.
- Require predicates to be directly entailed by an atomic fact or explicit numeric
  comparison, not a derived clinical composite.
- Make the frozen analyzer independently enforce these constraints where deterministic
  checks are possible, then repeat the unchanged five-case gate once.

The finite answer target, Bayesian update, likelihood tables, and binary observation
support are not the remaining problem. The remaining issue is the admissible action set.
