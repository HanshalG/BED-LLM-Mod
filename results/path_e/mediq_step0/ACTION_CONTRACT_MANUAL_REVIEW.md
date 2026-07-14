# MediQ Step 0 Action-Contract Smoke Manual Review

Run: `20260714T042154_mediq-step0-action-contract-eig-nonthinking-26b-seed1304`

Implementation commit: `ec46d62`

Verdict: **AUTOMATED PASS, NARROW MANUAL FAIL. Diagnostic only; one set-level
semantic-dedup repair remains before Claim 1.**

## Automated result

- Five official usable iMEDQA cases, ten turns, and 50 finite-target EIG tables.
- 10/10 clean mappings, 10/10 verbatim grounding, and 10/10 final relevance.
- Zero candidate-validation, structured-output, patient-relevance, or runtime failures.
- The hardened analyzer found no target-decode leakage, management-state action,
  derived predicate, compound query, open-ended query, or repeat of a deployed query.
- 288 requests, 84,482 prompt tokens, 15,586 completion tokens, no reasoning,
  $0.01411368, and 261.18 seconds.
- Endpoint accuracy was 3/5 and remains smoke-only, not policy evidence.

## Manual audit

Every selected query and selected patient fact passed manual review. In particular:

- treatment-class decodes disappeared from the medication-target case;
- the repeated unavailable worry query disappeared;
- numeric predicates mapped by direct comparison rather than clinical inference;
- no management-status or derived-stability predicate survived filtering;
- unavailable replies were used only when no atomic fact explicitly answered the query.

One candidate-set defect remains. In case 3, turn 1, the same five-candidate proposal
contains both "Does the patient have a history of renal calculi?" and "Does the patient
have a history of nephrolithiasis?" These are synonyms for the same latent fact. Both
are individually valid, so per-candidate validation and token-overlap deduplication do
not detect the redundancy. It reduces effective candidate diversity and could bias
comparisons when one method receives more genuinely distinct proposals than another.

## Required final repair

Audit each complete accepted candidate set for semantic duplicates using a
temperature-zero set-level judge. Retain the first representative of each duplicate
group and regenerate only the resulting deficit under the existing bounded retry rule.
Log the set-level validation and require it in the frozen analyzer.

The target, binary observation support, patient simulator, mapper, Bayesian update,
and individual action contract all pass. This final repair is candidate-pool quality,
not another endpoint or simulator redesign.
