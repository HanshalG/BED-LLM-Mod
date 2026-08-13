# RevengeBench Source Admission Protocol

Date: 2026-08-13

Status: **frozen before cloning the repository or opening target policies,
released traces, or benchmark outcomes**.

## Scientific Question

Can the public RevengeBench release support a genuinely new sequential BED
environment in which an LLM generates executable hypotheses about an opaque
game policy and designs probe opponents, while a held-out executable policy
provides a target-blind endpoint?

This is a source and mechanics admission audit. It is not evidence that a new
non-myopic method works.

## Immutable Source

- repository: `https://github.com/bethgelab/revenge-bench`;
- branch-tip commit observed before cloning:
  `351a5a7c2671150bae44c8bc46d7115ec996615f`;
- the checkout and every initialized submodule must be clean;
- the exact root tree, submodule commits, and relevant file hashes will be
  recorded by the audit.

Changing the source commit after this protocol requires a new dated protocol.

## Privacy Boundary

Before the source gates pass, the audit may read only:

- repository paths, file sizes, and hashes;
- `README.md`, `LICENSE`, `pyproject.toml`, `.gitmodules`, and public package
  metadata;
- benchmark, condition, and baseline configuration files;
- simulator, trace-parser, tournament, and evaluator interfaces needed to
  establish action, observation, reset, and endpoint semantics;
- target directory names and entrypoint existence, but not target-policy file
  contents;
- aggregate statements in the public paper or project page.

It must not read or serialize:

- any target-policy source code;
- released model messages, simulations, tournaments, per-target scores, or
  probe outcomes;
- held-out target actions or action-distance outcomes;
- generated reconstruction code;
- development or confirmation observations.

The audit must fail closed if the release cannot be inventoried without
crossing this boundary.

## Frozen Population Split

Within each arena, sort target directory names by
`SHA256("revengebench-bed-20260813:" + arena + ":" + target_name)` and allocate:

1. first 1 target to mechanics;
2. next 3 targets to opportunity;
3. next 4 targets to development;
4. next 4 targets to confirmation;
5. all remaining targets to reserve.

Only target IDs and their salted ordered hashes may be serialized. Target code
remains sealed in this audit. A later opportunity protocol may open only the
mechanics and opportunity target code after this split is banked.

## Source Gates

All gates are conjunctive:

1. The root commit and tree are reproducible, the checkout is clean, and all
   declared submodules are pinned and clean.
2. A research-permissive license covers the benchmark package.
3. At least four executable arenas and sixty target policies are present, with
   at least ten targets per retained arena.
4. Every selected target has a unique public ID and an expected executable
   entrypoint, without reading its contents.
5. The active-probe and no-probe conditions use the same target population,
   endpoint, and base interaction budget except for probe availability.
6. A probe is an ordinary executable opponent policy with no privileged access
   to target source or internal state.
7. The target response is an externally simulated behavioral trajectory, and
   the terminal endpoint is held-out target-action distance rather than an LLM
   judgment.
8. The release exposes explicit seeds or another exact reset mechanism for
   common-random-number paired replay. If exact replay cannot be established
   from interfaces alone, source status is pending until a separate zero-call
   replay audit passes.
9. Target source, prior run logs, released outcomes, and endpoint actions can
   remain unavailable to the policy and planning prompts.
10. The split contains exactly one mechanics and three opportunity targets per
    retained arena, at least twenty development targets, at least twenty
    confirmation targets, and a nonempty reserve.

## Irreducibility Contract

Passing the source gates does not by itself establish an LLM-native result. Any
later paid policy protocol must freeze all of the following before responses:

- the policy receives game rules, passive traces, and its own probe outcomes,
  never target code or released target-specific logs;
- LLM hypotheses are runnable target-policy programs or semantic rules that
  compile into runnable policies;
- myopic chooses the probe with maximum immediate predictive disagreement;
- dynamic depth two values how each possible probe trajectory changes the next
  LLM-generated hypothesis support;
- fixed-support depth two, compute-matched history-blind regeneration, random,
  and a released Bayesian Program Inference or known-pool classical control are
  all reported;
- all methods receive paired targets, passive traces, simulator randomness,
  probe budgets, and endpoint states;
- endpoint outcomes are sealed until serving, schema, executable-hypothesis,
  answer-obedience, repeated-fork stability, and source-level horizon gates
  pass;
- the primary claim requires dynamic depth two to improve held-out action
  distance, not merely entropy over its own hypotheses.

If a known-pool classical controller can reproduce the dynamic policy and
endpoint without LLM-generated support, the environment is supporting rather
than headline evidence.

## Decision Rule

- **Pass:** all source gates pass. This authorizes only a separately frozen,
  zero-call deterministic replay and structural-opportunity audit on mechanics
  plus opportunity targets.
- **Pending replay:** all non-replay gates pass but exact paired reset cannot be
  established from interfaces. This authorizes only a zero-call replay audit.
- **Fail:** any other source gate fails. No target content, model call, or
  endpoint may open.

No source outcome authorizes OpenRouter spend.
