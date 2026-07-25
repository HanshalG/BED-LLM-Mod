# tau-Knowledge Proposal Pooling Development Result

## Decision

Two-batch proposal pooling does not clear the development authorization gate.
No fresh tau execution is run.

This is a zero-call post hoc analysis on the two disclosed V3.1 executions.
It is mechanism development, not held-out evidence.

## Motivation

The original and replicated executions preserve similar aggregate structural
opportunity, but regenerate substantially different root queries and retrieved
documents. Pooling both five-root batches should help if the main problem is
proposal recall. A width-matched myopic policy is the relevant control.

## Methods Inspected

For every task, each method selects from one five-root batch or their ten-root
union. The chosen root always uses the already frozen receding continuation
scorer. Deterministic ties prefer the first execution and then the first root.

- `opaque_full_tree`: the existing V3.1 non-myopic root score.
- `myopic`: the isolated first-result semantic score.
- `focused_only`: the maximum count-dominant receding score.
- `raw_sum`: myopic score plus maximum receding score.
- `band_sum`: myopic score plus 30 times the receding new-document band.
- `rank_sum`: within-task rank sum of myopic and receding scores.
- `new_document_band_then_myopic`: lexicographic receding band, then myopic.

The last four are explicit Bellman-style decompositions rather than opaque
full-tree confidence.

## Results

Required-document totals and gains over the scope-matched myopic policy are:

| Method | Original | Replication | Pooled |
|---|---:|---:|---:|
| Myopic | 26 | 23 | 28 |
| Opaque full tree | 30 (`+4`) | 23 (`0`) | 25 (`-3`) |
| Focused only | 25 (`-1`) | 23 (`0`) | 25 (`-3`) |
| Raw sum | 26 (`0`) | 26 (`+3`) | 28 (`0`) |
| Band sum | 26 (`0`) | 26 (`+3`) | 28 (`0`) |
| Rank sum | 27 (`+1`) | 27 (`+4`) | 28 (`0`) |
| New-document band, then myopic | 25 (`-1`) | 25 (`+2`) | 29 (`+1`) |

The most favorable pooled rule wins three tasks, loses two, and ties fifteen.
It gains only one document. The gate required pooled gain at least four,
nonnegative gain in each individual execution, at least four pooled wins, and
at most two losses. No method passes.

The pooled pair oracle reaches 51 documents, so candidate width contains ample
latent opportunity. An unavailable oracle that chooses the better of the two
opaque batch winners per task reaches 33. The bottleneck is therefore
cross-generation value calibration, not merely missing roots.

## Interpretation

Proposal union alone does not stabilize tau. Raw full-tree confidence is not
comparable across independently generated supports, and transparent
immediate-plus-continuation combinations do not recover enough of the latent
oracle opportunity. Buying another same-task execution would test a method that
has already failed its zero-cost development prerequisite.

The next LLM-native method needs either a shared reference belief/value scale
across proposal batches or a fresh environment with externally stable actions
and transitions. It should not rely on selecting the largest self-rated score
from independently generated semantic supports.

## Budget

- New API calls: `0`.
- New OpenRouter spend: `$0`.
- OatML use: none.

## Artifacts

- Analysis:
  `results/nonmyopic/tau_knowledge_proposal_pooling_development/ANALYSIS.json`
- Analysis SHA-256:
  `4b0595f4b9f98939a896f0125b042abe77eb5f1e45fc60705203aa6648cb16b9`
- Script:
  `scripts/tau_knowledge_proposal_pooling_development.py`
- Original artifact SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`
- Replication artifact SHA-256:
  `627dfe7641dca9fade90890dc90f818da1c6967c08b7878208224cd961fd7f20`
