# RevengeBench Deterministic Replay Partial Result

Date: 2026-08-13

Status: **pending replay, one of five arenas passed**

## Decision

BattleSnake passes the frozen exact replay gate. RevengeBench remains
unadmitted because the primary gate requires at least four of five arenas.
Halite, HuskyBench, RoboCode, and RobotRumble remain unopened replay
dependencies. This partial result authorizes no target-content opportunity
audit and no model call.

## Frozen Execution

- RevengeBench commit:
  `351a5a7c2671150bae44c8bc46d7115ec996615f`;
- BattleSnake upstream commit:
  `aaf48003cfa5034a14d7053bb0ccc0bc6ae99cee`;
- replay protocol SHA-256:
  `dea501bd7c17768cb847e7998c1ddd0bb765a57f775a70f27f2661c9398fc9e9`;
- selected mechanics target:
  `cs4-5-20250929__3c2a3a422fb5`;
- fixed public opponent:
  `cs4-20250514__0787b4d66216`;
- simulator seed: `20260813`;
- runtime: two separate fresh containers built from the pinned BattleSnake
  source.

The release target programs require the public BattleSnake `server.py` at
runtime. The first wrapper attempt correctly failed before simulation when that
public dependency was absent. The repaired audit image supplied the pinned
public server module, required both HTTP health checks, rejected empty logs and
engine error text, and preserved the frozen target, opponent, and seed. The
failed attempt generated no trajectory and is not an evidence arm.

## BattleSnake Result

- 303 canonical target-visible states per arm;
- 302 inferred target actions per arm;
- exact state hash in both arms:
  `84adcd1c5fb36fdb9fca3c084a3c118b1a61892a51b4ee7b6ac7b8fe0a25eb42`;
- exact action hash in both arms:
  `f3bc3b9269583326406332401e967dd6d9f784589735ccd7c1a1cb3c98eb42ad`;
- exact terminal-result hash in both arms:
  `786bd7f0322721074c3324e2f8058fb95d41d0a87526fb2da075f9ce165bbca4`;
- normalized score and action-distance evaluation inputs match exactly;
- target wins both arms;
- raw trajectories, runtime IDs, policy source, provenance, and logs are not
  serialized in the public result;
- OpenRouter calls and cost are zero.

The engine writes one concurrently arriving snake request per turn, so raw logs
can differ in `you`, UUID, and latency metadata even when the full board is
identical. The frozen comparator reconstructs the target's `you` state from the
named snake in the complete board snapshot, removes only runtime IDs and
latency, sorts unordered board collections, and compares every state and
inferred action. This normalization is covered by positive and adversarial
tests; behavioral divergence fails closed.

Machine-readable audit:
`results/nonmyopic/revengebench_deterministic_replay/BATTLESNAKE_AUDIT.json`.

## Next Dependency

Run the same two-fresh-runtime replay audit on the four unopened arenas, using
their protocol-bound commits and mechanics targets. If at least three more pass
with no unexplained mismatch, freeze a source-level structural-opportunity
audit before reading any opportunity target content. Otherwise close the route.

No efficacy, planning-gap, or LLM-native claim is supported yet.
