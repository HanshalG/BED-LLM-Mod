# ChemBench MDA Second-Pass Architecture Review

Date: 2026-08-14 (Europe/London)

## Scope

This review translates arXiv:2608.09696v3, *Model Discovery Agent*, into
changes for the LLM-native non-myopic BED project. It uses only public paper
text, public source, and already-open ChemBench v3 development states. It opens
no v4/v5 response and makes no model/API call.

## What MDA Actually Gets Right

MDA is not an LLM planner. The LLM proposes a small batch of executable model
structures from the full pool's residuals. Numerical code then fits parameters,
computes marginal evidence, prunes the pool, chooses a one-step VoI experiment,
and performs a prequential predictive check. Three details are especially
important for our descendant:

1. The proposal transition is narrow and residual-conditioned: four new
   structures per ChemBench refinement round, not a uniform jump to every law.
2. Structures carry parameter posteriors and marginal evidence. A source
   parameter version is a truth draw, not a separate fixed candidate model.
3. ChemBench uses raw multiplicative-Gaussian observations and continuous
   seven-dimensional design optimization. Pool shrinkage removes the
   near-duplicate models introduced by open-world expansion.

The paper's chemistry ablation supports these choices independently: M-open
expansion enables compound recovery, CMA-ES improves hard-tier discrimination,
and ESS-adaptive pruning improves hard-tier symbolic accuracy.

## What Our V2/V3 Mechanics Got Wrong

The current mechanics are useful structural tests but not yet a faithful MDA
belief model:

- V2 compresses all missing structures into a scalar mass with a uniform
  three-bin observation model.
- The first speculative V3 replaces that scalar with all 48 outside structures
  at uniform weight. This gives the planner a richer truth prior but bypasses
  the proposal bottleneck it is supposed to plan through.
- Each registry entry is one fixed source parameter version. This conflates
  model structure with parameter uncertainty and makes action rankings brittle
  across versions.
- Fixed global terciles discard the magnitude and sign of the residual that
  MDA exposes to its proposer.
- The receding-horizon root score values a truncated contingent plan, but after
  the real observation the executor replans at the same horizon. The scored
  continuation is therefore not the executed continuation.

The verified V2 null localized the last issue as low or negative root-score
rank correlation despite near-perfect terminal truth support. The opened-v3
terminal-tail experiment did not repair it: d1/d2/d3 aggregate risk was
`.024430/.024883/.025048`.

## Immediate Architecture Correction: Policy Ladder

Define `pi_1` as the one-step dynamic-support policy. For levels `k > 1`, score
each action by its expected full-budget terminal risk when the continuation is
`pi_(k-1)`, then execute the resulting improved policy at every state:

```text
pi_k(s, t) = argmin_a E[J_(pi_(k-1))(T(s,a,Y), t-1)]
```

The candidate action set is shared across levels and includes the predecessor's
action. With exact branch transitions, finite budget, and exact expectations,
the policy-improvement theorem gives non-worsening Bayes risk from one level to
the next. This is a stronger and more interpretable depth definition than
receding truncated search: a reversal in realized risk can only come from a
misspecified speculative/proposal model or approximation error.

On the already-open v3 development states, the first exact prototype gives:

| Level | Aggregate terminal MSE | Change | Paired wins/ties/losses |
| --- | ---: | ---: | ---: |
| d1 | 0.03652561 | - | - |
| d2 | 0.03256128 | -10.85% | 37 / 83 / 24 |
| d3 | 0.02461982 | -24.39% | 36 / 92 / 16 |

Planned prior risk and uniform truth-conditional replay agree within `1e-16`
on every slice. This is development evidence only.

## Next Fidelity Changes

### 1. Proposal-aware speculative belief

Replace the 48-world uniform oracle with a distribution over reachable proposal
transitions. The LLM returns up to four structures plus machine-checkable
predicted signatures: which inputs should expose the mechanism, expected
direction of effect, and a discriminating assay region. Code verifies these
claims by compiling and simulating the proposed laws. Planning particles are
weighted by proposer recall and evidence, not by uniform registry membership.

For tractable branch planning, precompute a seed-bound proposal atlas over a
small vocabulary of residual signatures. Dynamic and call-matched myopic
policies consume the same atlas. The first-link gate measures proposal recall,
branch answer obedience, posterior predictive calibration, and action-ranking
fidelity.

### 2. Structure/parameter separation

Represent one executable expression as one structure particle with a posterior
over positive parameters. Use adaptive-tempered SMC and marginal evidence as in
MDA. Generate truth parameters independently from the candidate parameter
particles. This removes the current source-version shortcut and permits honest
generalization to v5 parameters.

### 3. Raw continuous observations

Replace fixed global terciles with log-rate likelihoods under the released 1%
multiplicative noise. During planning, use common-random-number posterior
predictive samples or deterministic quadrature. Preserve the signed residual
vector and its input correlations for the proposal transition.

### 4. Continuous designs

Use the released seven-dimensional bounds. Generate 48 common-random-number
design candidates, then optimize the best seeds with CMA-ES under a fixed
evaluation cap. The discrete 18-assay menu remains an exact debugging fixture,
not the final environment.

### 5. Pool control

Keep at most 12 live structures and four new proposals per expansion. Prune by
marginal evidence plus an explicit complexity prior, while retaining a small
reserve. Do not let the speculative prior contain dozens of near-duplicate
laws with equal weight.

## Environment Changes Worth Trying

1. **ChemBench continuous M-open** remains primary because the fixed-support
   opportunity and policy-ladder development screen are both positive. Add a
   four-to-eight experiment budget and raw held-out log-rate risk.
2. **NeuronBench deterministic** is the strongest secondary environment. Its
   novel currents are deliberately silent under textbook protocols and become
   visible only after conditioning pulses, creating a natural expose-then-
   identify sequence. Use summary-likelihood particles and policy levels d1-d3.
3. **NeuronBenchStoch** should follow only after deterministic calibration. Its
   repeat-count action introduces a real diversity-versus-replication tradeoff,
   but likelihood approximation can otherwise obscure the planning result.
4. **ForceBench hidden-source localization** is a clean low-dimensional backup.
   A first probe establishes that a hidden source exists; later probes localize
   it. The proposal structure is small (`K=0,1,2`) and parameter uncertainty is
   the dominant state, so it is ideal for validating the policy ladder.

Do not engineer a synthetic action prerequisite merely to force a depth win.
The primary result must use released environment dynamics and a sealed held-out
endpoint.

## Recommended Order

1. Freeze and verify the policy-ladder oracle mechanics on untouched v4.
2. Replace fixed parameter worlds and categorical bins on opened v3; require
   exact calibration and preserved depth opportunity before v5.
3. Run a zero-call proposal-atlas coverage audit from the public mechanism
   grammar.
4. Only then serve a small LLM semantic/ranking gate.
5. Execute v5 efficacy only if the LLM proposal model clears the first link.
