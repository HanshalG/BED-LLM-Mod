# CausaLab LLM-Native Source Admission Protocol

Date frozen: 2026-08-13, before reading any released graph record or episode
trajectory.

## Objective

Determine whether the newly released CausaLab benchmark can support a fresh
non-myopic BED experiment in which an LLM is irreducibly responsible for an open
causal-hypothesis support or likelihood model, rather than proposing experiments
inside a finite source-enumerable SCM family.

This is source admission only. A pass authorizes a separately frozen, zero-call
horizon-opportunity screen. It does not authorize model serving, task endpoints,
development, confirmation, or a paper efficacy claim.

## Immutable Source

- official repository: `https://github.com/DylanZSZ/CausaLab-Benchmark`;
- commit: `42ba47fb88e60dc1eca17bd29c47ace4e8e9960e`;
- Git tree: `5700be5d041eda64c56d610cdaa9dfaa1e7736c1`;
- license: Apache-2.0.

The audit may read repository paths, README and release metadata, environment and
agent source, prompts, launch scripts, and evaluator/parser source. It may not read
any JSONL graph record, bundled episode trajectory, raw model response, selected
task value, hidden graph, target frequency, intervention outcome, or endpoint.

## Frozen Gates

All gates must pass, in order:

1. **Bindings and release:** the repository commit and complete Git tree match the
   immutable source, the license is Apache-2.0, and the release metadata declares a
   nonempty versioned benchmark population.
2. **Native sequential experiment:** source defines a hidden SCM shared between an
   intervention object and a held-out transfer object, a finite intervention
   budget, observable intervention transitions, and a machine-parseable final
   causal hypothesis.
3. **Replay and sealing:** graph identity and environment randomness can be bound
   and replayed locally, while the policy-facing interface need not reveal the
   hidden graph, structural equations, coefficients, or transfer endpoint.
4. **Open-support ownership:** the released policy interface must not supply a
   complete candidate-world bank or a deterministic executable filter that can
   enumerate every candidate in the active task family and evaluate each observed
   intervention transition exactly. A source-provided complete finite filter fails
   this gate even if its candidate list is omitted from the LLM prompt.
5. **Irreducible LLM role:** after conditioning on all public source and policy
   observations, there must remain a semantic hypothesis or likelihood object that
   is neither exactly enumerable from the released task family nor replaceable by
   direct execution of the benchmark SCM code. Natural-language explanation,
   navigation, DSL formatting, or selecting among an exact source-owned candidate
   bank does not satisfy this gate.
6. **Prospective classical control:** the source admits a compute-matched myopic
   control and a stronger exhaustive/oracle causal-discovery control without giving
   either control privileged task outcomes. The intended LLM method must be capable
   of differing from both because of its generated support, not merely because the
   LLM executes a weaker approximation to their finite search.
7. **No privileged publication:** the result may contain only repository/release
   hashes, aggregate metadata counts, source-interface booleans, gate outcomes, and
   code-path citations. It may not contain a graph ID, node/edge/equation/parameter
   value, task description, intervention observation, target, prediction, or model
   output.

The audit stops at the first failure. Later gates are recorded as not evaluated, and
no graph record or episode may be opened after a failure.

## Decision

- `source_pass`: authorize only a separately frozen zero-call mechanics and
  horizon-opportunity screen. That screen must establish reproducible common-random-
  number branches, a depth-two changed first intervention, and positive terminal
  mechanism/transfer separation against compute-matched myopic and random controls
  before any model call.
- otherwise: close this exact CausaLab release as a headline LLM-native route. The
  benchmark may remain useful as an interactive-science or classical-BED baseline,
  but no paid experiment or efficacy claim is authorized.

Any future route based on a materially different CausaLab release or task family
requires a new protocol frozen before its records or outcomes are read. Gates may
not be weakened in response to this audit.
