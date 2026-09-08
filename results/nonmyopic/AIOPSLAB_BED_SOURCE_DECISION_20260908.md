# AIOpsLab: executable faults, but no qualified BED contract

## Decision

Do not deploy AIOpsLab as the next depth sweep. It provides genuine executable
runtime faults, unlike a static collection of incident labels. However, the
inspected interface does not supply the fixed-context, costed experiment and
predictive-target contract required by our current planner. This is a bounded
source decision, not an empirical opportunity null or a permanent rejection of
the framework. No candidate incidents were selected or executed.

The preceding goal turn was progress: it implemented and tested the public-context
conditioning boundary. This turn resolves whether a concrete runtime source can
be attached directly to that boundary. It cannot without a new scientific adapter.

## Verified source

[Microsoft AIOpsLab](https://github.com/microsoft/AIOpsLab) was pinned through
the GitHub API to `ddf7e40619689dad75eaf8f2174e263c4157ec76`. Its documentation
supports local ARM/x86 Kind deployments and live fault injection. We did not
install dependencies, create a cluster, or infer runtime feasibility from this.

The pinned implementation establishes the following:

- `actions/base.py`: metrics and traces use windows ending at `datetime.now()`;
  shell execution delegates to a general shell after a small command blocklist.
  The interface therefore does not itself ensure stationary responses or
  read-only experiments. Log collection is service-specific; metric export
  covers a namespace, so equal API-call counts need not mean equal information
  or collection cost.
- `orchestrator.py`: initialization deploys the app, injects the fault and starts
  workload before returning the description and actions. Thus initialization
  is execution, not a harmless metadata preflight. The later loop evaluates
  actions against the live system and cleans up after termination.
- `tasks/analysis.py`: the submitted answer is a system-level/fault-type pair,
  with application summary as initial context. This does not define our
  fixed held-out behavioral prediction target or latent-world prior.
- The public `revoke_auth` implementation supplies an actual runtime injection
  parameterized by faulty service. It is a useful concrete mechanism example,
  not a sampled population or proof of multi-step headroom. Its public evaluator
  was read as source; no private outcomes or telemetry were accessed.
- `tasks/base.py` can invoke an LLM judge when qualitative evaluation is enabled.
  A nominal zero-call launch would need to disable and verify that path.

These observations do not prove every action changes state, that a single log
solves every incident, or that the full framework lacks additional facilities.
They suffice to reject a direct substitution into a static likelihood matrix.

## What would have to change

There are two distinct experiments, which must not be conflated:

1. **Snapshot diagnosis:** freeze a telemetry collection window and acquire
   portions of that immutable record. This is active information acquisition,
   not intervention on the running system. It needs an explicit collection-cost
   model, context-conditioned population and sealed predictive endpoints.
2. **Interventional diagnosis:** run workload/configuration probes from reset
   replicas with shared nuisance seeds, or explicitly model evolving system
   state. This is closer to sequential BED, but needs repeated execution,
   likelihood calibration and a new state-aware reference. The current exact
   static-world solver does not establish those properties.

Neither contract is supplied merely by renaming tool calls as experiments.
Do not spend on model calls, deploy Kubernetes, or implement a telemetry adapter
until one is selected and its counterfactual sampling/cost/target law is fixed.
RCAEval's released telemetry benchmark was found but not downloaded: replaying
recorded incidents would not automatically resolve these counterfactual needs.

The actionable consequence is to stop the direct-debugging-adapter path here,
not repeat SWE-smith retrieval work or build infrastructure around an undefined
estimand. The project still lacks a single source that jointly demonstrates
useful executable LLM proposals and sufficient non-myopic predictive headroom.
That is the remaining scientific design problem, not a missing test harness.

## Reproducibility and accounting

Raw-byte SHA256 at the pinned commit:

| Source file | SHA256 |
|---|---|
| `aiopslab/orchestrator/actions/base.py` | `ef2684b76dc13ac51449963639331e7850f2b696fd2f3d3027eccb27300b0d63` |
| `aiopslab/orchestrator/actions/analysis.py` | `a17fda786bc883a0432408a510488713a51eda87a2dd26323201c41e4193f6e2` |
| `aiopslab/orchestrator/tasks/base.py` | `e09a0e62fd7074309670873698bd154087176fd89e9e69a1f7677442b2f24c43` |
| `aiopslab/orchestrator/tasks/analysis.py` | `4a54f37b1b6ea97cf3b1d0c39fb9f5538ddcd9ad2a4b0b772580a3f7d66a15f3` |
| `aiopslab/orchestrator/problems/revoke_auth/revoke_auth.py` | `cc3b1903d286703ef8075c9d2f279fce18cbbaf7862124ca2a904184f863f53d` |
| `aiopslab/orchestrator/orchestrator.py` | `f4c9b8fde87656a5ebee935804245e10614451597cde1095f39f48eeed91d131` |

This report changes documentation only; no runtime tests were run or claimed.
Source retrieval and authenticated credit reads are not model calls. Live
credits/usage are 245/220.376693994, matching the London Sept8 ledger: $0 spend,
$24.623306006 balance. No cluster, cache, protected runtime, paid endpoint or
automation changes. No positive result is claimed; the goal remains incomplete.
