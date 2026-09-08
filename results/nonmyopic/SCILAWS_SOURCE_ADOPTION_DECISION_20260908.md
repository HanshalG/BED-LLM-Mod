# SciLaws source audit: a candidate, not an authorized experiment

## Source and motivation

[SciLaws-Bench](https://arxiv.org/html/2609.01552v1) distinguishes fixed-record
prediction from active measurements in residual-calibrated parallel worlds.
Its reported candidate-selection bottleneck motivates numerical predictive
weighting rather than an LLM choosing its favorite formula. Its Parallel score
is judged structural recovery, not our fixed-target predictive risk. Neither
the paper nor this audit establishes a non-myopic or adjacent-depth gain.

Audited [code](https://github.com/yiyihum/SciLaws-Bench/tree/9239f66b921cb89c7a9d14061f782fdce49dcfb5)
at commit `9239f66b921cb89c7a9d14061f782fdce49dcfb5`, without checking out tasks.
The code LICENSE is MIT; dataset documentation specifies mixed per-task terms.
No task datasets, serialized states, released formulas or endpoint outcomes
were loaded. All dynamic tests use an artificial constant signal plus +/-1 noise.

## Reproduced contract

The [runtime](https://github.com/yiyihum/SciLaws-Bench/blob/9239f66b921cb89c7a9d14061f782fdce49dcfb5/harness/sim_runtime.py)
supports explicit point measurements with row accounting. Reusing a seed at
the same point replays the noise, rather than providing an independent replicate.
Coordinates outside support are clipped and the realized coordinates returned.

Its alternative `fetch_where` method filters a generated pool using a predicate
that can mention the target. In both fixtures, selecting y>10 returns only11
from the artificial9/11 distribution. That is selected sampling, not the
unconditional point likelihood. In the TypeII fixture, the outer max(1, ...)
permits one returned row even after a four-row budget is exhausted, advancing
used to5. TypeI rejects the same exhausted request. This is a runtime contract
defect; it is not evidence of its use in a reported experiment.

The [baseline task wrapper](https://github.com/yiyihum/SciLaws-Bench/blob/9239f66b921cb89c7a9d14061f782fdce49dcfb5/baseline_agent/task.py)
uses `fetch_data`, not `fetch_where`, for experiments. It forwards a supplied
seed. Runtime bundles also expose law-information methods, and their serialized
state contains executable hidden source. These must remain evaluator-only;
their existence is not proof of leakage through the published baseline.

Banked result: `SCILAWS_SOURCE_CONTRACT_20260908.json`. Three focused tests
pass in2.67s and lint passes. The tests execute selected pinned definitions
with fake state-loading and formula functions; they do not deserialize joblib.

## Adoption requirements

Proceed only to a point-measurement adapter and source-only public-manifest
audit, not to paid proposals or a depth sweep. Required interface:

- Only explicit, finite, in-support points and declared groups; reject clipping
  requests before dispatch. Hide where, sample, law-info and raw state methods.
- Evaluator-owned random streams, indexed by paired episode, round and replicate.
  The policy cannot select seeds or observe another policy's future measurements.
- Separate hard measurement ledger that reserves before calling the simulator;
  count attempted exposure after uncertain failure rather than retrying for free.
- Isolated evaluator process and filesystem. A Python object facade alone does
  not prevent a code-enabled proposer from inspecting hidden state.
- Fixed, policy-independent predictive targets and an explicit observation
  model. Local residual noise is not automatically Gaussian or centered at every
  point; audit the actual runtime distribution before claiming calibrated Bayes.
- Public-rule task inclusion, per-task source/license checks, bounded complete
  opportunity panel, and fresh semantic/calibration gate before any policy cohort.

The source provides measurement machinery but no deployable prior over unknown
laws or evidence of adequate horizon opportunity. Do not choose tasks based on
released model scores, added-law success, or whichever noise settings produce
monotonicity. No existing closed route is reopened. The overall goal and the
fresh controlled proposer/weight study remain incomplete.

Authenticated account/ledger rechecked: usage220.376693994, balance24.623306006,
zero daily spend. No model calls or paid authorization. Automation stays paused.
