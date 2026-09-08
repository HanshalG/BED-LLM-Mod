# GO-CBED source contract: reproducible planning code, missing discovery task

Previous turn: progress, through the tested headroom condition and focused
literature reassessment. This pass resolves source availability and the proposed
LLM role before installing or running another environment.

## Source Availability Corrected

GitHub repository search found
[HowIII/gocbed](https://github.com/HowIII/gocbed), which ordinary web queries had
missed. Pin inspected source to `db4e81000a2a40c0f8f53ac77deeafc8cf68813a`.
The README links the paper and supplies CPU smoke commands. MIT license with
third-party notices is present. Thus the earlier statement was only 'no verified
release found in that pass', not evidence that code was unavailable.

Only source/documentation text and repository-tree metadata were read. No bundled
posterior tensor, observation data, model checkpoint, experimental result or
source program was executed or loaded. No full repository clone or install.

## Paper And Implementation

The [paper AppendixC](https://arxiv.org/html/2507.07359v1) specifies linear Gaussian
and fixed neural-network mechanism priors, ER/SF graph distributions and
interventional quantities of interest. Its fixed-graph motivating example tests
goal alignment, not an h1/h2/h3 ladder. Continuous observations and intervention
values mean our exact binary solver is not a direct implementation. These facts
do not establish a tractable exact full-budget reference or a useful LLM prior.

The inspected [goal trainer](https://github.com/HowIII/gocbed/blob/db4e81000a2a40c0f8f53ac77deeafc8cf68813a/scripts/train_goal_policy.py)
requires `p1.pt`, observational tensors and per-node mechanism posteriors.
`_supported_parent_structures` conditions graph mass on parent sets having saved
mechanisms; `simulate_dag_and_weights` samples an acyclic graph from that support
and instantiates neural mechanisms using saved parameter samples. It does not
generate a new executable mechanism family after a real observation. The number
of supported structures was not inspected because that would require data reads.

This supported-posterior conditioning is explicit source behavior, not an
allegation of benchmark leakage. It also means silently replacing those artifacts
with an LLM proposal pool would change the simulator's model law and experiment.

## Decision

Do NOT adopt the shipped goal-training example as the LLM-native headline route.
The code is available, but the missing ingredient is a source-grounded discovery
problem in which an LLM contributes predictively useful mechanisms under equal
information and resource access. Naming known linear/NN families is insufficient.
Neither a CPU training smoke nor a full policy training run answers that question.

GO-CBED remains a relevant classical planning reference and methodological source.
Its graph space need not be tiny; lack of a meaningful LLM role is a separate
issue from enumeration cost. The paper's success does not imply a9.75% Brier
headroom bound for our desired two-step gain ladder under a different objective.
Do not discretize its observations, narrow its model support, or substitute our
own target while describing the outcome as reproduction of the paper.

The next task is not another source-availability search. Before selecting another
benchmark or implementing an adapter, specify a concrete scientific target,
source-supported unknown mechanism family and an observation that could change
which subsequent experiment is useful. Identify why executable LLM proposals
could outperform the runnable symbolic/parametric alternative. Assess that joint
discovery-and-planning contract before building infrastructure. Current evidence
does not supply such a qualified candidate, so no experiment is authorized here.

This is a substantive unresolved research-design problem, not a software bug
whose fix guarantees a positive result. Preserve the strong objective rather
than relabel another classical reference as completion. All previously closed
studies and the $5 daily budget remain unchanged.

## Pinned Evidence

| File | SHA256 |
| --- | --- |
| README.md | 8bfed76eca159192d191dcdc43c58036a00b40403b5be8361d582ab83ce9f3fe |
| LICENSE | ff0aeb0a79f5484818b0bb3782cc89b1f8862aac7f03b6330ab06102aeae1e09 |
| src/gocbed/synthetic/linear.py | 185ae3da36219ac581cacddcfeb88bc55a2895f8237e457a2ea8d8cd658d9395 |
| src/gocbed/synthetic/distributions.py | f2a58ef0d42eee0aedad8c545ea396609b286b6d92a16ccbc1f5cb9a7f7fe5a9 |
| scripts/train_goal_policy.py | acabc494849f9c103e5eff9c230e7f0b9caa1795be07ca6ae87838ba7ce6b307 |

Documentation-only result; no runtime tests claimed. Source-code syntax was
parsed for inspection, not imported. Account refreshed245/220.376693994/
24.623306006; London Sept8 spend0, remaining$5. All read commands completed.
No paid calls, cluster use or automation changes; full goal unfinished.
