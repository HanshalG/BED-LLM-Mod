# MuSiQue 4-Hop Discrete-Progress V2 Smoke Result

Date: 2026-07-25

## Outcome

**V2 failed the frozen mechanics-and-science conjunction.** Discrete terminal
progress improved pooled root fidelity to the exact `.30` gate boundary, but
collapsed most score vectors and did not improve the executed exact endpoint.
The construction is closed with the other 36 development and all 240 holdout
endpoints sealed.

Public artifact:

`results/nonmyopic/musique_branching_progress_v2_smoke/musique-branching-progress-v2-smoke-20260725T044224Z/SMOKE.json`

## Execution

| Metric | Result |
|---|---:|
| Physical requests / HTTP attempts | 16 / 16 |
| Transport retries | 0 |
| Reasoning tokens / forced exits | 0 / 0 |
| Parsed responses | 16 / 16 |
| Refreshed states differ from initial | 12 / 12 |
| Generated deep and shallow roots | 2 / 2 tasks |
| Prompt / completion tokens | 34,357 / 10,840 |
| Cost | $0.2484925 |

## Frozen Gate Result

| Signal | Required | Observed |
|---|---:|---:|
| Aligned terminal vectors vary | at least 10/12 | **3/12** |
| Aligned differs from shuffled | at least 8/12 | **6/12** |
| Aligned differs from initial | at least 8/12 | 12/12 |
| Deep-root selected continuation reaches child | at least 1/2 | **0/2** |
| Model-aware selects a deep root | at least 1/2 | 1/2 |
| Pooled root-score Spearman | at least `.30` | `.3015` |
| Model-aware at least myopic | 2/2 | 2/2 |
| Model-aware strictly beats myopic | at least 1/2 | **0/2** |
| Model-aware at least fixed/shuffled | 2/2 each | 2/2 each |

## Task Diagnosis

On `493923...62462`, generated roots included three deep and two shallow
candidates. Model-aware selected a deep root and predicted terminal progress
3 for one follow-up, but that follow-up retrieved the independent shallow
support rather than the annotated deep child. The selected exact prefix was 1,
tying myopic, even though another generated follow-up under that root made
exact prefix 2 reachable. Root-level Spearman was `.4472`; continuation
selection within the chosen root was wrong.

On `21483...24137`, generated roots again included both roles, but every root's
maximum predicted terminal progress was 2. The model-aware and myopic tie-break
therefore selected the first shallow root and both reached prefix 1. Exact
root endpoints still varied between 1 and 2, so root Spearman was undefined
under the constant prediction vector.

## Interpretation

Putting current and terminal values on one discrete scale repairs part of the
cross-root calibration problem: pooled fidelity rises from V1's `.073` to
`.302`. It creates a new resolution problem. The scorer assigns terminal band
2 to almost every follow-up, is only weakly sensitive to correct belief
alignment, and fails to identify the deep child even after selecting a deep
root.

The two MuSiQue smokes jointly isolate both required links:

1. free 0--100 utilities preserve variation but rank exact progress poorly;
2. discrete progress improves root ordering but collapses candidate resolution.

No band definition, tie rule, scorer prompt, or task is repaired after seeing
these outcomes. There is no V3. MuSiQue higher-hop branching remains a strong
external opportunity audit and a clean negative mechanism result, not the
headline non-myopic LLM-native positive.
