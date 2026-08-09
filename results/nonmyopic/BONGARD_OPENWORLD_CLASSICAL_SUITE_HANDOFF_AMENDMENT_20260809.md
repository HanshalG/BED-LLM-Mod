# Bongard OpenWorld Classical-Suite Handoff Amendment

Frozen: 2026-08-09 (Europe/London), before any Bongard mechanics,
development, or confirmation endpoint was opened.

Status: **implemented and fail-closed; zero model calls and zero outcomes**.

This amendment adds the endpoint-sealed SigLIP2-So400m comparator to the
already frozen DINOv2 reporting path. It changes no Luna request, model, task,
seed, action, endpoint, science gate, result tier, budget, or headline rule.
The original Luna renderer and DINO plan bank remain byte-identical.

## Authorization Before Labels

`scripts/bongard_openworld_classical_suite_outcome.py` is the registered
classical outcome adapter. Its mandatory order is:

1. verify the frozen DINO and SigLIP protocols, plans, and implementations;
2. load both endpoint-sealed plan rows;
3. independently replay the existing Luna stage authorization;
4. only then load candidate and endpoint labels;
5. follow both encoders' precomputed realized branches;
6. pair Luna, DINO, and SigLIP by exact opaque task ID;
7. checkpoint one immutable suite result.

Mechanics requires the complete Aug10 wrapper plus independently verified
mechanics artifact. Development and confirmation each require their exact four
replay-verified blocks and combined result. A missing, stale, partial,
null-authorizing, or mismatched predecessor fails before the opened-label
loader. An executable bomb test enforces this boundary.

The adapter reports both policies for both encoders, both horizon contrasts,
Luna myopic minus each classical myopic policy, and Luna dynamic depth two
minus each classical depth-two policy. Brier, log loss, accuracy, and truth
probability all receive fixed 20,000-draw paired task bootstraps. Every value is
reported regardless of direction.

## Mandatory Paper Path

After a verified development or confirmation result, the complete paper path
is now `scripts/bongard_openworld_paper_with_classical_suite.py`. Direct use of
the Luna-only renderer or the earlier DINO-only wrapper is scientifically
incomplete. The new wrapper:

- runs the original hash-bound Luna renderer unchanged in a temporary
  directory;
- independently replays the saved DINO+SigLIP suite result;
- preserves the original Luna text and headline file byte-for-byte;
- appends deterministic, direction-agnostic DINO and SigLIP paragraphs with
  myopic Brier, depth-two Brier, horizon difference and interval, and paired
  Luna-dynamic minus classical-depth-two difference and interval;
- writes combined TeX and metadata binding both provenance chains.

For a confirmation mechanics failure, no endpoint comparator can exist. The
wrapper preserves the frozen no-efficacy interpretation and says both
classical comparisons are unavailable. Supplying a suite result on that path
is invalid.

The wrapper never changes headline authorization. Only the original literal
`full_llm_native_confirmation` conjunction may populate abstract and
contribution macros. Synthetic tests prove exact headline preservation, and a
local `pdflatex` test compiles the complete classical paragraph.

## Stage Sequence

After a successful Aug10 mechanics wrapper, run the classical-suite adapter
once for descriptive mechanics evidence. It cannot authorize development.

After a verified Development64 or Confirmation96 combined result:

1. run the classical-suite outcome adapter once with the exact stage and block
   artifacts;
2. run the combined classical-suite paper wrapper with that immutable result;
3. never invoke or publish a Luna-only or DINO-only final fragment.

Any failed adapter or renderer verification is banked. It permits no Luna
rerun, endpoint reissue, altered classical plan, omitted comparator, or manual
paper repair. An unfavorable SigLIP or DINO result must remain visible.

## Bindings

- SigLIP baseline protocol SHA-256:
  `e4b126b9b95f18d62a8968959c67b1b5e2c9e8b01c7794b0dc23e7b8a80f5b86`;
- existing DINO baseline protocol SHA-256:
  `578e2ce911457a0c33f275e3f982f78dfa6d6216c30595277415e778264bbde8`;
- classical-suite outcome adapter SHA-256:
  `8c2bc93b3d416a47c2d1e19112a670f270b79712c22e4ba3d2181308e3d0e032`;
- combined paper wrapper SHA-256:
  `c9d74f0e188cee89a25c1bd3bd4ad7335786bb6b6b03dbf1a6b1c4b97bbdf639`;
- outcome tests SHA-256:
  `6936f98a7bbf10982f58a0aadb7a40e0bf9c35d82e141986aaa5cb69c9dbf797`;
- paper-wrapper tests SHA-256:
  `aa73b803928f9a73c1442550d9618a3c27b68459d6ce86472cd3540d3d8745cc`.

The complete Bongard regression suite passes `202/202`; the twelve new
SigLIP/classical-suite tests are included in that total.

This amendment authorizes no paid call, endpoint opening, new claim, or
result-dependent model or policy change.
