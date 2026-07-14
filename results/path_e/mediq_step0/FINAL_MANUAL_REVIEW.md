# MediQ Step 0 Final Manual Review

Run: `20260714T043855_mediq-step0-set-dedup-v2-eig-nonthinking-26b-seed1304`

Implementation commit: `7c8f64e`

Verdict: **PASS. Step 0 environment and probabilistic-mechanics gate is closed;
Claim 1 may be preregistered.**

## Automated gate

- Five hash-pinned usable iMEDQA cases, ten turns, and 50 candidate EIG tables.
- 10/10 clean response mappings, 10/10 verbatim grounded patient replies, and 10/10
  final relevance.
- 50 successful individual candidate validations and ten successful set-level semantic
  validations; zero terminal candidate, structured-output, patient-relevance, or runtime
  failures.
- Every final candidate used the canonical `Yes` / `No` / unavailable observation
  support and passed independent target-decode, atomicity, history-dedup, and finite-EIG
  reconstruction checks.
- 300 requests, 87,563 prompt tokens, 16,222 completion tokens, zero reasoning or
  forced exits, $0.01402286, and 247.05 seconds.

The smoke endpoint accuracy was 3/5 and was not used to make the gate decision. Five
cases are not policy evidence.

## Manual transcript result

All ten selected query/reply pairs pass:

- each query asks for one pre-decision patient fact or explicit numeric threshold;
- none asks for the diagnosis, medication, management decision, test-order status, or
  an answer-option paraphrase;
- unavailable is returned whenever no released atomic fact explicitly answers the
  predicate;
- positive and negative mappings are literal or direct arithmetic comparisons, including
  450 mg/dL versus 250 mg/dL, pH 7.1 versus 7.35, temperature 39.4 C versus 38 C, and
  blood pressure 114/82 versus 90 systolic;
- no selected fact is stretched through clinical or commonsense inference.

All 50 candidate predicates also pass manual review. Within each round they are distinct
clinical variables or meaningfully different tests; the earlier `renal calculi` /
`nephrolithiasis` synonym duplication is absent. Reusing an unselected concept in a later
round is allowed because it was never observed and the posterior has changed.

## Gate consequence

The finite target, canonical observation support, patient simulator, relevance/mapping
contract, Bayesian update, individual action support, and candidate-set diversity now
form one coherent BED process. Claim 1 must be preregistered before any paired comparison
is launched. The smoke remains diagnostic-only and contributes no efficacy estimate.
