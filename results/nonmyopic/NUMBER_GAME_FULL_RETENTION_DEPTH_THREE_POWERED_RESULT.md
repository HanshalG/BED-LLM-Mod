# Number Game Full-Retention Depth-Three Powered Result

Date completed: 2026-07-28.

Status: **gated null**.

## Frozen Design And Transport

The preregistered run evaluated 20 fresh GPT-5.4 Mini planning trees
(`28000..28019`) against 20 fresh Gemini 2.5 Flash target supports
(`28100..28119`). Both models were nonreasoning at temperature `0.7`.
Consistent parent particles were retained at both refreshes, and the frozen
`0.005` Brier risk-set selector was used without modification.

The run completed exactly 1,000 accepted requests in 1,000 HTTP attempts, with
zero retries, provider errors, reasoning tokens, or forced exits. Reported cost
was `$3.2480865`, below the frozen `$3.60` cap.

Public artifact hashes:

- `RESULT.json`:
  `3702be26d651fc14a2f19ce50b63713570dc573d17c9e74d0ad2adaa85f062b1`
- `TREES.json`:
  `0df3fe0d7441cc3481cc13bd8bb5b4913b199f870ee4bcb3d7074e0d74512598`
- private raw responses, retained locally and not committed:
  `a055546a57ead981ebf6a308e6dbca30f6343ff09fd5fc7c6c4b110826cb8ce6`

## Primary Result

| Policy | Mean Brier | Relative candidate gain | Mean Hamming | Root changes | Candidate wins | Paired Brier difference 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Full-retention depth three | 0.162454 | - | 0.084371 | - | - | - |
| Predictive-risk depth two | 0.163177 | 0.44% | 0.078673 | 13/20 | 5/20 | [-0.008685, 0.005557] |
| Retained parent-only depth three | 0.163882 | 0.87% | 0.076545 | 15/20 | 7/20 | [-0.005672, 0.002376] |
| Second-generated-only depth three | 0.163459 | 0.61% | 0.078003 | 11/20 | 7/20 | [-0.004811, 0.002741] |
| Myopic EIG | 0.168999 | 3.87% | 0.076433 | 19/20 | 12/20 | [-0.013897, 0.000882] |
| Fixed-support depth three | 0.172496 | 5.82% | 0.080415 | 20/20 | 14/20 | [-0.016599, -0.003430] |
| Uniform random | 0.171166 | 5.09% | 0.080973 | 20/20 | 18/20 | [-0.013007, -0.004592] |
| Positive-test strategy | 0.171348 | 5.19% | 0.080532 | 20/20 | 15/20 | [-0.014612, -0.003600] |

The candidate changed 13 of 20 depth-two roots, but improved mean Brier by
only `0.44%`; its tree-bootstrap confidence interval crossed zero and it won
only five trees. Mean Hamming regressed by `0.00570`, mean exact-support
coverage regressed by `0.00240`, and the novel-target aggregate regressed in
Brier, Hamming, and coverage. The primary monotonic-depth claim therefore
fails.

The candidate did beat fixed-support depth three and uniform random by about
5%, with Brier intervals below zero. These controls show that the previously
established proposal-aware signal remains present, but they do not establish
incremental value from depth three over depth two.

## Mechanism Diagnosis

Full retention was mechanically consequential:

- first-refresh retention passed every support gate;
- depth-three roots differed from depth two on 13/20 trees, from parent-only
  on 15/20, and from second-generated-only on 11/20;
- mean source-risk versus independent-target Brier Spearman was `0.4988`
  with 95% bootstrap interval `[0.4012, 0.5964]`;
- mean pairwise concordance was `0.6893`, with interval
  `[0.6500, 0.7304]`.

However, one tree contained a fully retained second support of size six, below
the frozen minimum of eight. More importantly, the source simulator's valid
but moderate Brier ranking did not translate into a stable incremental
depth-three decision advantage. Hamming ranking was uninformative
(`-0.1047` mean Spearman), and novel targets were worse than depth two.

The `0.005` risk-set selector chose exactly the same root as pure Brier on all
20 trees. Its Hamming/coverage tie-break never affected the result, so tuning
that tolerance cannot explain or repair the null.

## Interpretation

The earlier zero-call audit correctly showed that retaining hypotheses repairs
the LLM belief transition. This powered run shows that transition repair is
not sufficient for monotonic planning depth. The remaining failure is root
ranking under a second generated transition, especially generalization to
independent novel targets. The appropriate next step is a zero-call,
same-tree decomposition of depth-two and depth-three ranking fidelity before
any new paid policy run. There will be no threshold repair, selective tree
removal, or rerun of this protocol.
