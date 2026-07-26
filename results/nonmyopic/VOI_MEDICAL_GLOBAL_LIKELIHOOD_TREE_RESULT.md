# VoI Medical Global-Likelihood Tree V2 Result

## Decision

V2 fixed the response-model coherence defect and passed every serving,
likelihood, branch-dynamics, and score-range gate. It failed the decisive
non-myopic opportunity gate: exact myopic and depth-two EIG selected the same
root.

The exact V2 interface is closed. There will be no alternate seed, threshold
change, partial-policy claim, or patient-level validation.

## Frozen Run

- Preregistration and implementation commit: `399425f`
- Run ID:
  `voi-medical-global-likelihood-tree-20260726T060946Z`
- Interface: `voi-medical-global-likelihood-tree-mechanics-2`
- Model: `openai/gpt-5.4` through OpenRouter
- Source SHA-256:
  `e851864a9cb53c36304245bc3213a8a894cf7b86f8945d60978923e1f1ef0169`
- Public mechanics SHA-256:
  `b8c33b57b2ebe4093e423a83a37f4f4eff7fdb549b32a82e972aee3484cd0552`
- Private raw SHA-256:
  `593c5d9ef5b7470b09a9b53fccb93de212c10d03b96c183d29b32c05ec44bec1`

## Execution

- unique new follow-up questions: `21`
- reused question occurrences: `3`
- expected and realized requests: `17 + 21 = 38`
- HTTP attempts: `38`
- retries: `0`
- reasoning tokens: `0`
- forced exits and forced-final requests: `0`
- prompt tokens: `6,808`
- completion tokens: `4,010`
- cost: `$0.07717`

Every root had all three outcomes at positive probability. All twelve branch
question sets parsed, all four roots had answer-conditioned follow-up sets, and
each normalized semantic action used exactly one isolated likelihood map.

## Exact Scores

| Root | Immediate EIG | Depth-two EIG | Future increment |
|---|---:|---:|---:|
| Diarrhea | 0.917020 | 1.809424 | 0.892404 |
| Right-lower abdominal pain | 0.847729 | 1.836492 | 0.988763 |
| Cough | 0.817190 | 1.740815 | 0.923625 |
| Vomiting | 0.941328 | 1.901256 | 0.959928 |

All values are nats under the frozen empirical diagnosis prior and deterministic
global semantic likelihood maps.

The immediate score range was `0.124138` nats and the depth-two range was
`0.160441` nats, so the tree was informative and rankable. Nevertheless,
vomiting was both the myopic and depth-two optimum. The frozen
depth-two-over-myopic selected-root gain was therefore exactly `0`.

## Interpretation

V1's local-batch instability was a real implementation issue, and V2 removes
it. The remaining null is structural for this generated tree: the
right-lower-pain root has the strongest continuation value, but its future
increment exceeds vomiting's by only `0.028835` nats while its immediate EIG is
lower by `0.093599` nats. Lookahead cannot overturn that immediate deficit.

This is useful mechanics evidence for a coherent LLM-native semantic Bayesian
tree, but not evidence that non-myopic selection improves over greedy VoI. The
preregistered patient-grounded validation is not authorized.

## Budget

V2 spent `$0.07717`, leaving:

- frozen research allowance: `$1.13174155`;
- live OpenRouter balance: `$34.203232594`;
- balance above the protected `$25` Monday reserve: `$9.203232594`.

OatML and cluster jobs: `0`.
