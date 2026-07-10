# Path E Step 0: Environment Standing

## Decision

Paprika customer service **passes the structural Step 0a acceptance check**, subject to
the still-pending real-model answer-space coverage gate. The adapter does not invent an
issue label, causal graph, prerequisite graph, or success condition.

## Released Evidence

- Repository: `https://github.com/tajwarfahim/paprika.git`
- Pinned commit: `8470461554f12f9f9215ee5e78ea19ba1d576b5c`
- Released file: `llm_exploration/game/game_configs/customer_service.json`
- SHA-256: `0ebadf112db200501d90cda51aa9df618e889d408a6f4ecb7a8a6dcfac3d3c5e`
- Validated split sizes: 628 train tasks and 200 evaluation tasks.

Each released task has exactly the two fields needed for the adapter:

- `agent`: the public symptom/scenario shown to the customer-service agent.
- `env`: the private solution sentence supplied only to the customer simulator and
  success judge.

The released simulator prompt directs the customer to answer only asked questions,
withhold the private solution, and emit exactly `Goal reached` when the proposed action
solves the issue. Paprika also releases a solution-grounded judge prompt returning
`<VALID>` or `<NOTVALID>`. The adapter preserves both success paths: exact customer
termination or semantic judge validation.

The hidden truth is therefore an explicit instance-specific natural-language solution,
not a symbolic class ID. The success verifier is semantic and model-based, not a
hardcoded deterministic checker. Those are properties of the released benchmark and
must remain explicit limitations in reporting.

## Adapter Evidence

Implementation through commit `cfdc9a0` provides:

- hash-verified loading without vendoring the upstream dataset;
- LLM-generated cause/remedy hypotheses, history-conditioned refinement, explicit
  consistency filtering, and full-history categorical posterior updates;
- candidate questions/actions with 3--5 mutually exclusive proposed outcomes;
- free-text customer simulation and judge mapping to the proposed answer set;
- one-step categorical EIG and full depth-two categorical lookahead;
- belief-free naive execution with the same native success endpoint;
- prompt-scoped replay for paired roots/simulator responses at identical states;
- batched root, branch, and follow-up likelihood evaluation;
- per-turn resolution, turns-used, answer-space coverage, and cache diagnostics.
- bounded structured-output repair with retry and terminal-failure metrics.

The deterministic mechanics smoke uses five tasks from the pinned official evaluation
split for two rounds and reports answer-space coverage 1.00. It is explicitly **not LLM
evidence**; its purpose is adapter lifecycle verification. Artifacts:

- `results/path_e/step0a_adapter_smoke/REPORT.json`
- `results/path_e/step0a_adapter_smoke/paprika_smoke.json`

Focused tests cover data validation, configuration aliases, categorical EIG, exact
Mastermind depth-1/depth-2 planning, posterior refresh/filtering, paired candidate
replay, belief-free naive behavior, native early stopping, and batched full-two-step
branch expansion.

## Remaining Gate

Run the configured five-task 26B A4B thinking smoke on `msc,llm`, excluding `oat12`.
Proceed only if clean answer mapping is at least approximately 85%, transcripts are
semantically reasonable, native success checks do not show obvious false positives, and
structured-output failures are controlled. The launch is currently pending because
`ssh oat0` returns `No route to host` from the active machine.
After completion, run `scripts/analyze_paprika_smoke.py` on the run directory. An
automated pass still requires manual review of every surfaced query/reply transcript.

## Distribution Caveat

No obvious `LICENSE` or `COPYING` file was found in the pinned upstream checkout during
the audit. Accordingly, this repository records provenance and provides a fetch script,
but does not vendor Paprika's full source or dataset. Licensing must be rechecked before
redistributing benchmark data or derived task text in a release artifact.
