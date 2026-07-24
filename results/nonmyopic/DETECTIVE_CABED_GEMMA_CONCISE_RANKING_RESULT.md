# Detective CA-BED Gemma Concise Ranking Result

Date: 2026-07-24

## Verdict

**The frozen depth-ranking gate fails decisively. Do not run the sealed 24-case
sequential confirmation.**

Depth-two CA-BED changed the selected root on 6/12 cases and accurately ranked
which roots would reduce posterior entropy, but it ranked truth-posterior gain
backwards. This is confident-wrong concentration caused by mismatch between
the LLM's textual likelihood table and its hidden-role answers, not policy
collapse or insufficient planning separation.

## Protocol Integrity

- Exact released CA-BED `DetectiveCases.json`, SHA-256
  `049ea3003753b15e3319483d15993591b5d50ac7eedb3187dca6ef3951cd2a57`.
- Frozen cases: `31, 96, 91, 65, 99, 56, 2, 4, 50, 6, 87, 34`.
- Gemma 4 26B A4B, non-thinking, temperature zero.
- Uniform four-suspect prior; three roots; three follow-ups per binary branch;
  four two-turn hidden-role answer rollouts per root.
- Shared questions, likelihood tables, observations, and exact Bayesian updates
  for depth one, depth two, and seeded random.
- Exact `624/624` requests, zero reasoning tokens, zero forced exits, no content
  retry, parser repair, case replacement, or missing tree cell.
- Cost: `$0.06102971` (`583,650` prompt and `20,349` completion tokens).

## Frozen Gates

| Metric | Required | Observed | Pass |
|---|---:|---:|:---:|
| Rankable cases | >=9 | 11 | yes |
| Different d2 root | >=4 | 6 | yes |
| d2 score vs truth-gain Spearman | >=.20 | **-.208** | no |
| d2 Spearman advantage over d1 | >=.10 | **-.364** | no |
| d2 truth-NLL gain over d1 | >=.02 | **-.262** | no |
| 90% bootstrap CI over d1 | lower >0 | **[-.544, -.013]** | no |
| d2 wins over d1 | >=7/12 | **2/12** | no |
| d2 truth-NLL gain over random | >=.02 | **-.312** | no |
| 90% bootstrap CI over random | lower >0 | **[-.589, -.071]** | no |
| Final entropy d2 minus d1 | <=.02 | **-.056** | yes |

The negative intervals are especially informative: this is not merely an
underpowered wash.

## Per-Case Selection

| Case | d1 root | d2 root | d1 truth gain | d2 truth gain | d2-d1 | d2 entropy advantage |
|---:|---:|---:|---:|---:|---:|---:|
| 31 | 2 | 2 | -.698 | -.698 | +.000 | +.000 |
| 96 | 2 | 0 | -.151 | -.898 | -.747 | +.073 |
| 91 | 2 | 1 | -.144 | +.231 | +.374 | +.099 |
| 65 | 1 | 0 | +.106 | -1.504 | -1.610 | +.123 |
| 99 | 0 | 2 | +.323 | -.773 | -1.096 | +.331 |
| 56 | 0 | 1 | -.022 | +.198 | +.220 | +.088 |
| 2 | 0 | 0 | +.704 | +.704 | +.000 | +.000 |
| 4 | 1 | 0 | +.513 | +.231 | -.282 | -.038 |
| 50 | 2 | 2 | -.673 | -.673 | +.000 | +.000 |
| 6 | 1 | 1 | +.483 | +.483 | +.000 | +.000 |
| 87 | 1 | 1 | -.566 | -.566 | +.000 | +.000 |
| 34 | 1 | 1 | -.358 | -.358 | +.000 | +.000 |

## Mechanism

The score is not generally random:

- Mean root-ranking Spearman with realized **entropy reduction** is `.750` for
  depth two versus `.182` for depth one.
- Mean root-ranking Spearman with realized **truth-log-probability gain** is
  `-.208` for depth two versus `+.045` for depth one.
- Five of the six changed d2 roots improve realized entropy relative to d1, but
  only two improve truth probability.
- The truth-conditioned textual likelihood assigns the realized root answer
  probability below `.5` on 9/36 roots. At selected follow-ups it does so on
  10/36 roots.
- All four answer replicas agree on 35/36 roots; the remaining root has two
  unique answer pairs. The endpoint noise is therefore much smaller than the
  likelihood-model bias.

The direct numerical estimator and hidden-role answerer are solving different
semantic problems. The likelihood prompt sees public case context and invents
counterfactual response behavior under each murderer. The answerer sees the
target suspect's private story and actual murderer/innocent role. Depth two
maximizes over more such mismatched likelihood rows, finds trajectories that
look sharply discriminative, and confidently concentrates on the wrong
suspect.

## Decision

Close this exact concise-Gemma textual-likelihood design and leave the 24
confirmation cases untouched. The one principled follow-up is a development-only
likelihood-alignment diagnostic: build each question's likelihood from the same
counterfactual hidden-role response function used by the environment, then
rescore the already generated shared trees. This must restore positive
truth-gain ranking on the inspected 12 cases before any fresh case is spent.
