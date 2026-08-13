# RevengeBench Deterministic Replay Protocol

Date: 2026-08-13

Status: **frozen after source admission returned `pending_replay` and before
opening any target-policy source or simulator trajectory**.

## Purpose

Determine whether RevengeBench can provide exact paired common-random-number
observations for future non-myopic, myopic, history-blind, and random policies.
This is a zero-model-call mechanics audit. It cannot establish a planning gap
or policy efficacy.

## Source Bindings

- RevengeBench commit:
  `351a5a7c2671150bae44c8bc46d7115ec996615f`;
- RevengeBench tree:
  `8991d42d09f3f8fb095580d87a68ba21b7dc6f5c`;
- CodeClash submodule:
  `a66d63eee9a1f4f5bcda7ca753c404dcfdb63e92`;
- source-admission artifact and protocol hashes are recorded in the replay
  result.

The release's arena Dockerfiles clone moving upstream branches. Before any
replay, bind these observed upstream commits and patch only clone checkout plus
RNG plumbing in the temporary audit checkout:

| Arena | Upstream commit |
| --- | --- |
| BattleSnake | `aaf48003cfa5034a14d7053bb0ccc0bc6ae99cee` |
| Halite | `173ef1a63ea9987d9df3439d4d34b3346a5cadad` |
| HuskyBench | `9963f13b2d545cbdb12d716ad0c603e9219d46a1` |
| RoboCode | `1f360189e50a2cbab25e8ac7ef280736610753a7` |
| RobotRumble | `4f019c91e6854242f5e247ca82fee7b66833b227` |

No upstream source content is a policy input.

## Frozen Mechanics Targets

Use only the source-admission mechanics target in each arena. Target IDs are
already banked in the admission artifact. The replay runner may copy and
execute the selected target entrypoint and one deterministic public opponent,
but it must never serialize their source content or expose it to a model.

The public opponent is the lexicographically first non-target target ID in the
same arena. This is fixed before reading any policy source.

## Deterministic Wrapper

For each arena and replay arm:

1. start a fresh process/container from the pinned sources;
2. set Python `random`, NumPy, and environment seeds from
   `SHA256("revengebench-replay-20260813:" + arena + ":" + simulation_index)`;
3. replace in-place global shuffles with a runner-owned `random.Random(seed)`;
4. pass the same explicit simulator seed wherever the arena supports one;
5. use the same target/opponent ordering, board/game arguments, and simulation
   count in both arms;
6. hash canonical target-visible state/action trajectories, normalized final
   scores, and action-distance evaluation inputs;
7. run the two arms in separate fresh environments.

If an arena has hidden nondeterminism that cannot be controlled without
changing its game semantics, that arena fails. No partial trajectory may be
used as scientific evidence.

## Gates

Each arena must pass all of:

1. pinned external arena checkout verified inside the runtime;
2. selected target and opponent compile/validate in two fresh arms;
3. exact equality of canonical initial target-visible state;
4. exact equality of every target-visible state and target action;
5. exact equality of trajectory length, terminal state, and normalized score;
6. exact equality of offline action-distance evaluation inputs;
7. at least three nontrivial target actions per arm;
8. no target code, target provenance, released logs, or prior outcomes are
   serialized in the public result;
9. OpenRouter calls and cost remain zero.

The primary replay gate requires at least four of five arenas to pass, with no
unexplained mismatch. An arena may be excluded only for a banked deterministic
infrastructure incompatibility, not because its trajectory is inconvenient.

## Decision Rule

- **Pass:** at least four arenas pass all gates. This advances only those arenas
  to a separately frozen source-level structural-opportunity audit.
- **Fail:** fewer than four pass or any mismatch is unexplained. Close this
  RevengeBench route before target-content opportunity analysis or model calls.
- **Infrastructure pending:** the local container runtime is unavailable or a
  pinned image cannot be built for reasons unrelated to the arena logic. Bank
  the exact condition; do not infer scientific failure and do not spend.

This protocol authorizes no OpenRouter request and no endpoint claim.
