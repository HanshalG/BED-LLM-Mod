# ChemBench MDA Third-Pass Architecture Decision

Date: 2026-08-15 (Europe/London)

## Decision

Keep ChemBench M-open as the primary route and keep the exact policy-improvement
ladder as the definition of depth. Do not serve an LLM proposal atlas yet. The
next dependency is a continuous-observation structure/parameter posterior whose
planning transition and realized update use the same likelihood.

The intended paper contribution is narrower than MDA and genuinely distinct:
MDA uses an LLM to expand a mechanistic hypothesis pool, but chooses each
experiment myopically. Our method plans non-myopically through future
observation-conditioned support expansion and then executes the same improved
continuation policy that it scored.

## New Empirical Finding

The first parameter-particle prototype represented each source parameter
version as a particle. On already-open v4 truths its aggregate d1/d2/d3 MSE was
0.201955/0.186878/0.191165. Planned particle risk was monotonic, but held-out d3
regressed 2.29% from d2.

An audit then found that v4 was outside the coordinate-wise v0-v3 parameter
range in about 40% of dimensions. A deterministic 16-particle Latin-hypercube
prior was therefore constructed from v0-v3 only, with positive kinetic scales
sampled in expanded log space and pKa/exponent/modifier parameters sampled in
linear space. The v4 values did not enter prior construction.

With 1.5-span expansion and the same three opened v4 slices, held-out truth MSE
was:

| Policy level | Aggregate MSE | Successive change |
| --- | ---: | ---: |
| d1 | 0.335446 | - |
| d2 | 0.331921 | -1.05% |
| d3 | 0.364507 | +9.82% |

The d2-vs-d1 practical cells were 76/3/65 wins/ties/losses; d3-vs-d2 were
45/18/81. Planned particle risk remained monotonic on all slices.

This is a diagnostic null, not a reason to narrow the prior. The current belief
engine updates on three global rate bins. Broad parameter particles that land
in the same bin are observationally indistinguishable, so widening the prior
increases forecast variance without supplying the continuous data needed to
identify the parameters. MDA instead evaluates the raw likelihood and fits a
per-structure parameter posterior.

## Required Architecture

### 1. Raw continuous belief state

Store each observation as `(design, log1p(rate))`, not `(assay, tercile)`. Use
the released one-percent multiplicative Gaussian observation model, transformed
to log-rate with a numerically stable Gaussian approximation. Structure
evidence and parameter weights must be recomputed from this exact history.

The realized update and every simulated branch must call the same likelihood.
The fixed tercile engine remains only a regression fixture.

### 2. Adaptive parameter inference

Each executable structure declares typed parameter bounds. Start with a
deterministic prior particle set, then add ESS-triggered resampling and bounded
Metropolis rejuvenation. Compute structure weights from marginal evidence,
rather than giving each structure a point estimate or equal version mass.

For development, candidate bounds may be inferred from v0-v3 and broadened by
a frozen rule. Untouched v5 evaluation must draw or use truth parameters that
never entered those bounds or particles.

### 3. Continuous observation branches

Non-myopic planning cannot enumerate a continuous response. At each candidate
design, form three to five deterministic posterior-predictive quantile branches.
Each branch carries its probability and a representative raw log-rate, and the
child posterior updates on that representative value. Use common quantile
levels across policy depths and controls.

Validate this approximation against a high-sample Monte Carlo one-step value
before allowing deeper planning. Gate on root-action ranking and value error,
not only final monotonicity.

### 4. Task-risk objective

Optimize expected terminal held-out log-rate MSE, matching the evaluation
functional. Information gain over structures is a useful diagnostic but not
the deployment objective: it can prefer distinctions irrelevant to forecast
risk. This follows MDA's task-facing posterior-predictive evaluation while
preserving our non-myopic contribution.

### 5. LLM proposal atlas

After the numerical gate passes, the LLM emits a strict executable proposal:

- a typed rate-law AST;
- bounded parameter declarations;
- the residual pattern it explains;
- a qualitative response signature;
- one exposing region and one falsifying region in design space.

Code compiles the AST, rejects invalid or duplicate laws, fits parameters, and
checks the claimed signature. The LLM is never asked for posterior weights,
likelihoods, entropy, or rollout values.

For tractable planning, call the LLM before endpoint execution to build a
seed-bound atlas indexed by quantized public residual summaries. Dynamic d1-d3,
call-matched myopic, fixed-atlas, history-blind, and random-proposal controls all
reuse the same responses. This makes future support expansion simulatable
without model calls inside the tree.

### 6. State-dependent designs

First validate the raw-likelihood architecture on the frozen 18-assay menu.
Then replace the menu with a state-dependent candidate generator over the
released seven-dimensional box: common random designs plus local CMA-ES
refinement under one shared evaluation cap. MDA's chemistry ablation indicates
that sharper continuous design matters particularly on hard mechanisms, but it
should not be introduced simultaneously with the likelihood repair.

### 7. Pool control and M-open trigger

Use a prequential residual aligned with held-out log-rate prediction to trigger
support expansion. Keep a small live pool and reserve, prune by evidence plus a
complexity prior, and re-expand only when prediction is poor. This prevents the
near-duplicate proliferation observed in both our registry oracle and MDA's
ablation.

## Environment Strategy

### Primary: ChemBench continuous M-open

Use primitive initial support, compound/novel hidden structures, four to eight
sequential experiments, raw rates, and disjoint query assays. The natural
non-myopic event is diagnostic-residual acquisition followed by a targeted
parameter-identification experiment. Freeze cohorts and truth parameters before
LLM responses.

### Secondary: deterministic NeuronBench

This is the strongest natural horizon environment. Several hidden currents are
silent under textbook steps and exposed only by a conditioning pre-pulse; the
subsequent protocol then identifies the current. Begin with deterministic
summary likelihoods and only later add stochastic repeat allocation.

### Backup: ForceBench hidden particles

Use the hidden-source task as a low-dimensional validation of structure plus
parameter planning. It is useful for debugging but less compelling than
ChemBench for an LLM-native mechanism-proposal result.

## Gates Before Any Paid Efficacy Run

1. Raw posterior normalization, evidence, and no-truth-leak tests pass.
2. Quantile-branch one-step values rank actions against high-sample Monte Carlo
   with Spearman rho at least 0.8 and low top-one regret.
3. On opened v4, held-out d2 improves at least 5% over d1, d3 improves at least
   5% over d2, and practical paired wins exceed losses for both links.
4. A zero-call grammar audit shows the proposal atlas can represent every
   sealed structure family without importing truth IDs.
5. A small paid gate establishes strict parse rate, executable validity,
   residual-conditioned proposal movement, truth-family recall, and positive
   proposal-action ranking fidelity.
6. Only then open untouched v5 for paired dynamic d1-d3, compute-matched myopic,
   fixed/history-blind proposal, random-design, random-proposal, and naive
   thinking comparisons.

## Immediate Implementation Order

1. Replace empirical tercile histories with raw log-rate histories.
2. Add predictive-quantile branch construction and Monte Carlo fidelity tests.
3. Re-run the widened prior on opened v4.
4. Add adaptive SMC rejuvenation only if static particles still collapse.
5. Freeze and implement the proposal-atlas codec and controls.
6. Spend on the LLM semantic/ranking gate, not on an efficacy sweep.

No API/model call or new sealed endpoint was opened for this decision.
