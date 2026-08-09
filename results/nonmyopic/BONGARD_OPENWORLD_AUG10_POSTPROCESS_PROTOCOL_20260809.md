# Bongard OpenWorld August 10 Postprocess Protocol

Frozen: 2026-08-09 (Europe/London), before any August 10 Bongard response,
candidate label, or endpoint label was opened.

Status: **zero-call execution protocol; authorizes no model request**.

## Purpose

The frozen August 10 paid wrapper can terminate as a serving null, a mechanics
null, a mechanics pass, or a separately banked component failure. Three
downstream instruments already exist: the mechanics disposition classifier,
the endpoint-sealed DINOv2 plus SigLIP2 classical suite, and the path-mediation
replay. Running them by hand creates an avoidable ordering and omission risk.

`scripts/bongard_openworld_aug10_postprocess.py` is the sole registered
postprocessor for this stage. It makes no model or provider call and does not
invoke the paid August 10 executor. It verifies the bound implementations,
classifies the terminal artifact, and writes immutable provenance records.

## Frozen State Machine

1. Classify the supplied terminal wrapper, serving/mechanics result, or
   component failure with the frozen disposition implementation.
2. An invalid or stale artifact produces one `FAILURE.json`. It authorizes no
   retry, paid call, development stage, label access, or claim.
3. A valid disposition other than a wrapper-bound `mechanics_pass` produces a
   terminal disposition-only `RESULT.json`. No classical label or raw belief
   replay is opened.
4. Only an exact August 10 wrapper `RESULT.json` whose hash-bound mechanics
   component passed and whose pre-existing authorization is true can continue.
5. Run the classical-suite outcome adapter first. It independently replays the
   wrapper and mechanics result before loading candidate or endpoint labels.
6. Only after the classical suite checkpoints a complete zero-call result, run
   the path-mediation report. It independently replays the same authorization
   before reading hash-bound raw beliefs and endpoint-bearing trees.
7. Checkpoint one composite `RESULT.json` binding the input, disposition,
   classical suite, path mediation, protocol, and all implementations.

Existing component files are never overwritten. A clean component checkpoint
may be reused after an interrupted process only if its interface, stage,
zero-call invariants, source path, and source hash all match. Any exception is
banked once in `FAILURE.json`; later invocations return that failure and do not
resume or rerun a component.

## Non-Authorization Invariants

Every composite result and failure records:

- `model_calls: 0` and `cost_usd: 0.0`;
- `authorizes_paid_calls: false`;
- `authorizes_rerun: false`; and
- `this_record_authorizes_development: false`.

The postprocessor may report
`existing_wrapper_authorizes_development: true` only when that exact Boolean
was already carried by the independently verified paid wrapper. It creates no
authorization and cannot change the paid chain, claim tier, confirmation gate,
classical scope, or paper headline.

## Bound Inputs

- paid August 10 wrapper:
  `adf0cede0c14e1ac96206461371f2f53f434f5b748327f9cf93ae0e7f521f9a5`;
- mechanics disposition:
  `870c6fe5ba1a9dd09250193bc36e4bb108708a10dc07bc225dc075d596f4d8bb`;
- classical-suite outcome adapter:
  `8c2bc93b3d416a47c2d1e19112a670f270b79712c22e4ba3d2181308e3d0e032`;
- path-mediation implementation:
  `1aef1c9eb90757bd31fec4beb077ddf79965e1a42b2715b4f7a6788e57e8b912`.

The postprocessor changes no model, effort, prompt, schema, task, seed, action,
endpoint, policy, threshold, request count, or budget. Unfavorable classical or
mediation evidence remains visible and cannot rescue or stop the pre-existing
development authorization.
