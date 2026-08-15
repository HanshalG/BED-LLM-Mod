# ChemBench Moment-Shortlist Sampling Protocol

Date frozen: 2026-08-15 (Europe/London)

## Purpose

Test a practical action-generation layer for posterior-sampling planning after
the full 14-action IID and RQMC estimators failed rank coverage. Candidate
actions are selected without reference outcomes using a moment-matched task-
variance-reduction proxy, then valued with unbiased shared-IID common random
numbers.

This changes the planner's prospective action set, not the posterior,
likelihood, task targets, saved references, metrics, or thresholds.

## Frozen Bindings

- RQMC failure result SHA-256:
  `a966d5cf4984c9907649a0dae5d6bb8a19982f942c83f463f7ec61e4e2d439f2`.
- IID failure result SHA-256:
  `d71964ff247bc80408b0b5c78c4b168c5117ece372446990c4a5dfd17b42c0f6`.
- Pooled reference result SHA-256:
  `637c031946268e8015687d5eb4af4d037bfc1399300e56c0e50bdf049b4fab07`.
- Component-bank reference result SHA-256:
  `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`.
- Authorized V3 result SHA-256:
  `dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1`.

Use the exact 36 source-only pooled roots, 14 remaining actions, 128 task
targets, and saved 2,048-outcome full-action references.

## Endpoint-Blind Shortlist

For each root posterior, compute transformed observation means `m_a(theta)`,
observation variances `sigma_a(theta)^2`, and terminal target values
`f_j(theta)` over the pooled particles.

For each action, define the linear-Gaussian moment proxy

```text
gain(a) = mean_j Cov(m_a, f_j)^2
                    / (Var(m_a) + E[sigma_a^2])
```

under equal pooled-particle weights. Select the top eight actions by descending
gain, breaking exact ties by ascending frozen assay index. Do not use saved
action risks, component references, mechanism names, or outcomes in selection.

## Shared IID Sampling

For each case and replicate, draw 256 pooled-particle indices uniformly and
256 independent standard-normal noises from seed base `2026084000`, stably
mixed with replicate, difficulty, and domain. Reuse the same ordered particle
indices and noises for all eight shortlisted actions. Do not mix action into
the seed.

Evaluate nested prefixes `32`, `64`, `128`, and `256` for four independent
replicates. The four-replicate 1,024-sample mean is diagnostic only.

## Shortlist Gates

Before sampling fidelity can pass:

1. the full-reference best action is in the top-eight shortlist in at least
   90% of cases;
2. the best shortlisted action has full-reference regret at most 3% of root
   risk in every case;
3. mean shortlist-oracle normalized regret is at most 0.5%.

These conditions prevent action reduction from manufacturing a ranking pass.

## Sampling Gates

All four 256-sample replicates must independently satisfy, over the eight
shortlisted actions:

1. every binding, posterior, proxy, shortlist, random stream, outcome, weight,
   and risk is finite and reproducible;
2. median within-shortlist Spearman at least 0.90;
3. at least 90% of cases with within-shortlist Spearman at least 0.80;
4. at least 90% of cases with selected-action full-reference normalized regret
   at most 3%;
5. mean selected-action full-reference normalized regret at most 1%;
6. for each component-bank reference, at least 90% regret coverage at 3% and
   mean normalized regret at most 1%.

Report all lower prefixes, selected-action agreement, proxy/reference rank,
shortlist composition, and the four-replicate ensemble. They are diagnostics.

## Decision

A pass freezes the moment top-eight action generator and 256 shared-IID CRN
trajectories per action for an oracle-support horizon-opportunity gate. It
authorizes no LLM call.

A failure closes this ChemBench numerical route at the current root/action
construction. Move to a prospectively designed stronger-horizon compositional
environment before any LLM spend; do not tune K, proxy terms, seeds, or
thresholds against this panel.

This protocol contains no planning depth, structural proposal transition, LLM,
API, network call, benchmark endpoint, or paid resource.
