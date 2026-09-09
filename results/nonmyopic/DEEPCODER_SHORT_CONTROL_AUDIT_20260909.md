# A cheap exact short-program control recovers missed behaviors

Retrospective diagnostic on all eight opened feedback cases, not a fresh
confirmatory comparison. The two-statement component was selected after the
case1 structural failure was known; do not treat the measured advantage as
unbiased out-of-sample evidence for this algorithm choice.

## Result

| Method | Mean target half-Brier | Zero-mass targets /256 | Nonempty support |
|---|---:|---:|---:|
| Banked initial Luna + local expansion | 0.208059 | 47 | 7/8 |
| Exact two-statement component | 0.185439 | 32 | 7/8 |

The symbolic control is descriptively10.87% lower loss on this opened panel.
It removes every zero-mass target in the seven supported cases; the32 remaining
are the fixed abstention in case5. This is not a full-grammar posterior: it
conditions the original prior on length2. Cases were still drawn from lengths
2/3/4 and no difficult case or longer source truth was excluded.

| Case | Compatible short programs | Distinct target behaviors | Short Brier | Luna Brier |
|---|---:|---:|---:|---:|
| 0 | 4 | 1 | 0.000000 | 0.000281 |
| 1 | 36 | 4 | 0.240789 | 0.437500 |
| 2 | 2 | 1 | 0.000000 | 0.000000 |
| 3 | 1 | 1 | 0.000000 | 0.000000 |
| 4 | 64 | 23 | 0.166748 | 0.186378 |
| 5 | 0 | 0 | 1.000000 | 1.000000 |
| 6 | 337 | 174 | 0.075974 | 0.040316 |
| 7 | 1 | 1 | 0.000000 | 0.000000 |

In case1 the search includes Last/Drop despite the identity-looking observed
examples. First-statement types are enumerated and each second statement is
constructed against that type, so it is not confined to type-preserving local
edits of an LLM root. It also reveals the tradeoff: broader supported predictions
can increase realized loss, as in case6. Coverage alone is not calibration.

## Scope and verification

Every source-valid two-statement syntax is visited: exactly3518 programs per
case. Filtering receives only the three public observations. Prediction uses
the original syntax-prior weights restricted to the complete length2 compatible
set. The full source mass of the unfiltered component is exactly1/3, checked with
rational arithmetic; this is not the missing length3/4 mass or an importance
correction for LLM-selected pools. No original exact-full-prior timeout was
rerun or relabeled as complete.

Enumeration plus forecast construction took0.044-0.238s per case (about0.626s
sum), excluding source download, parent replay, and the extra descriptive
behavior-count calculation. Retained program order, candidate counts, history
execution counts, scores and forecasts are banked. The audit first independently
replayed parentb85bd72c. Two tests passed0.99s: source-complete count/mass and a
type-changing counterhypothesis compatible with identity-looking history. Scoped
lint passed; all processes exited. No hidden outcomes newly opened, model calls,
API spend, deployed policy or trainable weights.

## Decision for the goal

Carry this exact-short component into future program-proposal controls. Do not
claim an LLM-native win against only the weaker first-fit symbolic search or
limited local neighborhoods. This result weakens the case for further undirected
Luna prompt variants on the current uniform-syntax source; it does not prove
all program induction is unsuitable or that full length3/4 inference is cheap.

Before another paid proposal variant, require a concrete reason it can create
useful behaviors beyond this control, not merely reproduce its short programs.
An alternative source must provide genuinely useful semantic context and still
have a coherent hidden-world/observation contract; neither convenient oracle
headroom nor artificial restrictions on the baseline establishes an LLM role.
The requested nontrivial non-myopic result remains unachieved. This audit is a
baseline/architecture decision, not a replacement success criterion.

Authenticated account245/220.663020549/24.336979451 unchanged; Sept9conservative
spend0.24474207, remaining4.75525793. No cluster or automation changes.
