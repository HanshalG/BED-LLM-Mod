# ClinDiag Test-Card Action Audit

Date: 2026-07-24

Status: **the direct retrospective test-card route is rejected before model calls.**

## Question

Could each recorded ClinDiag procedure become a fine-grained action, with its finding
hidden until that action is selected? This would avoid the saturation seen when an
entire laboratory, imaging, or other-test block is revealed at once.

## Frozen Static Filter

The audit excludes every case used by the staged generator gate, its sealed holdout,
and the native-block opportunity gate. It then:

1. applies the existing no-diagnosis-leak and case-completeness checks;
2. groups duplicate procedure names within each case;
3. removes confirmatory procedure labels such as biopsy, pathology, genetics, and
   sequencing;
4. removes interventions such as transfusion, surgery, pacing, and transplantation;
5. rejects any remaining action whose normalized name contains the normalized final
   diagnosis.

The filter is target blind except for the final exact lexical-leak check. It makes no
LLM calls.

## Result

The archive contains many nominally rich records. After excluding 98 reserved cases
and 674 malformed, incomplete, or lexically leaking cases, 1,249 fresh cases remained.
The procedure filter removed 1,143 confirmatory entries, 253 intervention entries, and
60 unnamed entries.

| Minimum eligible cards | All cases | Challenging | Rare |
|---:|---:|---:|---:|
| 4 | 1,042 | 903 | 139 |
| 5 | 918 | 805 | 113 |
| 6 | 808 | 718 | 90 |
| 7 | 688 | 615 | 73 |
| 8 | 583 | 524 | 59 |
| 10 | 391 | 359 | 32 |

No remaining procedure name contained the full normalized target string.

## Why The Route Still Fails

Card count is not the limiting issue. The archive is retrospective:

- the set of procedures was selected by clinicians who were responding to the true
  case, so action availability itself is target dependent;
- unperformed tests have no recorded result, so the data do not specify the
  counterfactual branches needed for `p(y | theta, x)`;
- disease-specific names such as organism serologies reveal the source differential
  without requiring a literal diagnosis-string match.

Consequently, a policy could exploit which tests happen to appear in a case, while a
lookahead planner could not evaluate tests absent from the record. Filtering more
aggressively would hide the symptom rather than repair the missing potential outcomes.

## Consequence

No serving smoke, opportunity gate, planner, or holdout is authorized for the direct
test-card construction. A valid clinical continuation must use a target-independent
query interface plus an explicit outcome mechanism. The closest established design is
SDBench's free-text gatekeeper: the diagnostic agent may ask a question or order a
test, and a case-conditioned model returns a specific finding, including a
case-consistent synthetic result when the source record lacks that test.

Before any planner comparison, that alternative needs its own low-cost gate for:

1. grounded response accuracy on findings present in ClinDiag;
2. no diagnosis or interpretation leakage;
3. semantic consistency under exact duplicate requests;
4. meaningful branch variation conditional on the hypothesized diagnosis;
5. a target-blind structural gap under a fixed action budget.
