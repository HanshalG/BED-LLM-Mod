# RevengeBench Source Admission Result

Date: 2026-08-13

Status: **pending deterministic replay**

## Decision

The public RevengeBench release passes every frozen non-replay source gate and
is the strongest currently released candidate for the next LLM-native
non-myopic BED environment. It is not yet admitted for source-opportunity or
model work because exact paired common-random-number replay is not established
in the required four of five arenas.

The only authorized successor is the zero-call deterministic replay audit.

## Bound Release

- RevengeBench commit:
  `351a5a7c2671150bae44c8bc46d7115ec996615f`;
- root tree: `8991d42d09f3f8fb095580d87a68ba21b7dc6f5c`;
- CodeClash submodule:
  `a66d63eee9a1f4f5bcda7ca753c404dcfdb63e92`;
- admission protocol SHA-256:
  `90103e558a836c599442efd5e89e6e296d8c7c43d4c590f2ccb5053a1ad828a0`;
- admission audit SHA-256:
  `fe8f8afdf8be66e791583a2ae84b4061e70fc83805b62f1366d73c8c684af50f`.

## Source Findings

- Five executable strategy-game arenas are present.
- The release contains 200 target-policy directories, 40 per arena. This is
  the released candidate pool; it is distinct from the paper's evaluated
  target count.
- All 200 required target entrypoints exist.
- The salted, per-arena split is complete and disjoint:
  mechanics 5, opportunity 15, development 20, confirmation 20, reserve 140.
- Active-probe and no-probe configurations match after removing only the
  frozen intervention fields.
- Probes are ordinary executable opponent policies with no privileged access
  to target source or internal state.
- The target does not participate in probe simulations.
- The primary endpoint is held-out target action distance. It is not a probe
  reward or target-identification label.
- Public BPI and known-pool baselines exist and are mandatory future controls.
- Target source, released trajectories, logs, provenance, and outcomes were
  not opened or serialized by this audit.
- OpenRouter calls and cost were zero.

## Blocking Reproducibility Finding

The released harness uses process-global unseeded shuffles for agent and
BattleSnake player order, the BattleSnake launcher omits the simulator's
supported explicit seed, and arena Dockerfiles clone moving upstream branch
tips. These are repairable mechanics defects, but the source audit cannot infer
exact common-random-number pairing from code inspection alone.

The separately frozen deterministic replay protocol binds the observed arena
commits, passes explicit simulator seeds, owns runner RNG, and requires exact
target-visible state/action replay in at least four arenas. No target-content
opportunity audit and no model request may open before that gate passes.

This result is source/mechanics evidence only. It makes no claim about a
non-myopic structural gap, LLM semantic calibration, policy efficacy, or paper
headline.
