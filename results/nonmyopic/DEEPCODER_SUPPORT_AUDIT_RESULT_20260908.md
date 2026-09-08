# Independent programs expose finite-bank overconfidence

## Result

The diagnostic frozen at `da164aa0` completed all four panels in .881 seconds.
It independently drew 512 programs from the unchanged grammar prior and checked
all eight single-query conditions per program. All original input, program and
reference-matrix hashes reproduced before independent outcomes were evaluated.
No policy was optimized or rerun. The preceding finite-prior opportunity null
and its thresholds remain unchanged.

| Panel | Unsupported query answers | Supported-only held-out Brier | Supported-only internal risk |
|---|---:|---:|---:|
| 0 | 362/1024 (35.35%) | .40756 | .21836 |
| 1 | 301/1024 (29.39%) | .45860 | .26371 |
| 2 | 404/1024 (39.45%) | .43147 | .20550 |
| 3 | 338/1024 (33.01%) | .42202 | .21253 |
| Pooled | 1405/4096 (34.30%) | .43047 | .22609 |

There are 2691 supported conditions. The last two pooled columns weight those
conditions, not panels equally. No forecast exists for the other 1405 conditions;
they are NOT counted as correct, smoothed, replaced or omitted from the failure
rate. Conditions share programs and inputs, so these are descriptive counts,
not 4096 independent statistical replicates.

Among supported conditions, 35298/86112 target outputs (40.99%) receive zero
predictive probability. The corresponding log loss would be infinite for those
targets; no arbitrary epsilon is introduced. The reported half-multiclass Brier
loss stays finite and exposes the discrepancy with internal posterior risk.

Before conditioning, held-out losses .45961--.47043 are much closer to internal
risks .44255--.46324. Good-looking aggregate prior uncertainty did not protect
against severe conditional support failure.

## What this establishes

The 128-draw bank is not adequate for weight-only conditioning over independent
programs under this source law. Small-bank exact search was mathematically
coherent for its stated empirical population, but not an adequate reference for
the broader grammar. Its 9.05% headroom bound cannot rule out stronger effects
there. Conversely, these new failures do not show that more particles or LLM
proposals will produce a positive depth effect.

This identifies a concrete approximation failure without LLM noise: exact
updates can be confidently wrong when the finite support misses relevant
program behavior. It does not establish the cause of historical location,
ChemBench or other environment failures, which need their own evidence.

## Next architectural decision

Do not select another environment or launch another depth grid merely because
this small-bank reference saturated. First qualify evidence-conditioned support
construction under the SAME generative law on independent programs. Keep the
original finite pilot closed; this is a new numerical-adequacy dependency, not
a new version of its efficacy gate.

The reference baseline should condition the grammar prior on actual observations,
rather than only reweight an initially sampled bank. A bounded prior-rejection
sampler is a transparent first reference: accepted draws have the conditional
prior law, but rare outputs may make it computationally unusable. Report that
failure explicitly, not a smaller target, a changed output alphabet or a retry
until success. Productive enumerative search is the other required comparison.

An LLM may eventually help by finding compatible executable programs under a
limited computation budget. Its proposals are not automatically draws from the
conditional prior: unknown proposal probabilities and data-dependent selection
must not be hidden behind a Bayesian label. Validate predictive performance on
independent programs and distinguish finite-pool conditional inference from
population posterior accuracy. Only then reconsider non-myopic planning and
the stronger future-discovery objective.

## Verification and accounting

Six focused tests passed in .39 seconds and scoped lint passed. Tests cover
hand-computed multiclass scores, confident-wrong predictions, explicit undefined
updates, shape rejection, and agreement of Brier/internal risk when truths really
are drawn from the same finite bank. No fresh efficacy threshold was introduced.

Artifact `DEEPCODER_SUPPORT_AUDIT_20260908.json`, SHA256
`e5c56e61c5fc73626bb904bffb258f46e5bb7f456765b6003e55f38dba05d180`.
It records aggregate counts, original matrix bindings and independent program
and output hashes. The process exited normally; no scientific worker remains.

Model calls:0; spend:$0. Authenticated account credits/usage 245/220.376693994
match the London Sept8 ledger. No cluster, protected runtime or automation
changes. The full plan remains unfinished and no paid experiment is authorized.
