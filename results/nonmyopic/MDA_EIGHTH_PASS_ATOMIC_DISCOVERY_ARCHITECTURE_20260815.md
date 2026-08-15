# MDA eighth pass: atomic discovery architecture

Date: 2026-08-15

## Decision

The next ChemBench descendant should plan over **hypothesis creation**, not ask an
LLM to emit hidden-registry identifiers. It should use public atomic mechanism
edits, a deterministic compositional compiler, quantitative task-anchored
residuals, and numerical Bayesian fitting. The environment must contain staged
compound mechanisms; the standard one-edit source subset is too easy for a
monotonic depth test.

No paid model call is authorized by this note.

## What Murphy et al. actually imply

The current MDA paper is deliberately hybrid:

- the LLM is only a structure proposer;
- it sees the mechanism grammar, raw experiment inputs and rates, all tried forms,
  per-form residuals, the input correlated with the best model's residual, budget,
  and phase;
- candidate equations are compiled and fitted by SMC;
- experiment design is numerical VoI, and the shipped method is myopic;
- M-open expansion is triggered by predictive inadequacy;
- the pool shrinks only when evidence is concentrated and fit is adequate;
- the residual/summary must contain the downstream task target, or model search can
  fit the summary while missing the task;
- stochastic NeuronBench makes repeat count a costed design variable.

The strongest opportunity for this project is therefore not to duplicate MDA. It
is to make its M-open support transition part of the non-myopic belief dynamics.

Paper: <https://arxiv.org/abs/2608.09696>

## Diagnosis of the failed proposal-atlas gate

The primitive-pool gate closed after 126/126 responses because the tested
interface differed from the scientific MDA interface in four consequential ways:

1. It required a full registry signature and a redundant edit operation instead
   of one atomic scientific edit. Sixty schema-valid proposals disagreed with
   their declared parent operation.
2. A public grammar described valid compositions, but the private compiler
   accepted only combinations already present in a fixed 57-model registry. This
   rejected 143 otherwise structured signatures as non-executable.
3. The three-bin residual report discarded quantitative rate error, scale,
   parameter uncertainty, and the continuous log-rate target. Residual-aware
   proposals did not beat the history-blind arm.
4. The cheap endpoint also had a transport problem: only 107/126 responses were
   clean, with most failures caused by image-control-token corruption. The clean
   subset still failed semantically, so transport was not the primary cause.

Scaling the same interface to a stronger LLM would test compliance with an
artificial registry task, not scientific hypothesis discovery.

## Source-only atomic opportunity audit

`scripts/chembench_atomic_edit_opportunity.py` restricts each frozen source slice
to the nine initial primitives plus 17 standard one-edit successors. Eligible
truths use Michaelis-Menten, Hill, substrate-inhibition, or ping-pong cores and one
standard inhibition, product, pH, or Arrhenius edit. It makes zero model calls.

Across 51 paired truth cells, exact oracle terminal target MSE is:

| Planner | Mean MSE | Sample SD |
|---|---:|---:|
| d1 | 0.0010209541 | 0.0036317323 |
| d2 | 8.25e-16 | 5.89e-15 |
| d3 | 0 | 0 |

d2 improves eight cells and ties 43. d3 improves zero cells and ties all 51 at
numerical zero. This subset has a strong d1-to-d2 gap but no usable d2-to-d3 gap.
It cannot support the desired monotonic depth result.

## Successor architecture

### 1. Atomic edit proposer

The LLM returns only:

```json
{
  "parent_id": "h3",
  "edit": "add_arrhenius",
  "rationale_axis": "T",
  "confidence": 0.78
}
```

The finite public vocabulary should include add/remove inhibitor type, Arrhenius,
pH bell, product inhibition, and second-substrate dependence, plus replace-core
edits among the standard substrate laws. The compiler, not the LLM, constructs the
equation, parameter list, bounds, canonical signature, and inverse edit. Invalid
compositions fail before inference. One field determines one fact.

### 2. Compositional executable model pool

Replace the hidden registry lookup with a real expression graph. A hypothesis is
`core + modifiers`; compatible modifiers compose deterministically. Fit each
structure's parameters to accumulated continuous rates with SMC or a cheaper
Laplace/importance approximation during development. Evidence, not name matching,
chooses among structures.

### 3. Task-anchored diagnostic state

The proposer state should contain:

- the top structures, posterior weights, parameter intervals, and evidence;
- every collected assay and observed continuous rate;
- per-structure signed and relative residuals by assay;
- weighted residual correlations and paired diagnostic contrasts for each input;
- posterior predictive error on a public target-query sketch drawn from the same
  target distribution used by terminal risk;
- all tried edits with their evidence change;
- remaining budget and explore/refine phase.

This preserves the target signal while avoiding endpoint labels. The residual
representation gets an explicit answer-obedience test before any policy run.

### 4. M-open belief transition inside planning

Use the Markov state

`(history, structure pool, parameter posteriors, residual state, tried edits)`.

For every simulated `(action, observation)` branch:

1. update structure and parameter evidence;
2. run the calibrated inadequacy trigger;
3. sample/rank atomic edits from a cached proposal transition model;
4. compile, fit, merge, and evidence-prune the expanded pool;
5. continue planning from the expanded belief.

This is the missing non-myopic object. An early experiment can be valuable because
it makes the right hypothesis representable before later experiments discriminate
or estimate it.

### 5. Separate proposal fidelity from policy efficacy

Before depth sweeps, require a small branch-matched semantic gate:

- edit is executable and novel;
- residual-aware edit beats a history-blind edit;
- proposal distribution changes appropriately under simulated answer flips;
- true edit recall and induced target risk beat typed retrieval and random edits;
- recursive fidelity holds after the first generated hypothesis enters the pool.

Only then compare d1/d2/d3. A call-matched myopic control receives the same cached
proposal branches but cannot condition future actions on their outcomes.

## Successor environment

Build a **staged compound ChemBench corridor**, frozen before LLM responses:

- truths contain two ordinary compatible modifiers, not obscure core families;
- parameter uncertainty prevents one assay from identifying the mechanism;
- a broad screening assay reveals which residual axis needs an edit;
- a second targeted assay identifies the modifier created after that screen;
- a third assay estimates or reveals the remaining modifier;
- budget is four cost units, with optional repeats consuming units;
- continuous noisy observations replace categorical bins;
- common random numbers pair all policies and sealed target-query outcomes.

The source-only oracle must first pass prospective gates: material d2 over d1 and
d3 over d2 gains, root-action changes on multiple slices, truth-cell majorities,
and nontrivial residual risk after d2. If those fail, redesign the corridor before
calling an LLM.

## Secondary environments

1. **NeuronBenchStoch with costed repeats.** This is the cleanest secondary test of
   non-myopic resource allocation: protocol and repeat count are jointly chosen,
   and the LLM proposes channel edits while numerical likelihoods update beliefs.
2. **Unknown-kernel location finding.** Let the LLM propose signal-law components
   while analytical inference handles source locations. This is a useful control,
   but spatial reasoning should not remain the headline environment.
3. **ForceBench compositional laws.** Its compact equation grammar may be the
   easiest semantic-proposal gate, but a natural depth-three opportunity must be
   demonstrated source-only first.

## Recommended order

1. Implement the atomic expression compiler and quantitative residual fingerprint.
2. Generate a prospective staged-compound source package and run an exact oracle
   d1/d2/d3 opportunity audit with parameter uncertainty.
3. Freeze a 12--24 branch semantic proposal gate and test a stronger text model.
4. Require recursive expanded-pool fidelity.
5. Run paired d1/d2/d3 policy efficacy against call-matched myopic and random
   controls.
6. Add costed repeats or NeuronBenchStoch only after the primary link works.

The key architectural bet is concise: **the planner should choose experiments that
create useful future hypotheses, while the LLM only proposes typed scientific
edits and the Bayesian layer owns every numerical claim.**
