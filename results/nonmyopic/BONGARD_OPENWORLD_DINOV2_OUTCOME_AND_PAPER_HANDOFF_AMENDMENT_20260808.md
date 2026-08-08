# Bongard DINOv2 Outcome and Paper-Handoff Amendment

Frozen: 2026-08-08 (Europe/London), before any Bongard mechanics,
development, or confirmation endpoint was opened.

Status: **implemented and fail-closed; zero model calls and zero outcomes**.

This amendment operationalizes the already frozen DINOv2 classical-baseline
protocol. It changes no Luna request, model, task, seed, action, endpoint,
science gate, result tier, budget, or headline rule. The original Luna paper
renderer remains byte-identical.

## Authorization Before Labels

`scripts/bongard_openworld_dinov2_outcome.py` is the only registered outcome
adapter. Its ordering is mandatory:

1. verify all frozen DINO protocol, plan, and implementation hashes;
2. independently replay the existing Luna stage authorization;
3. only then load candidate and endpoint labels;
4. follow the precomputed DINO branches and score endpoints;
5. pair Luna and DINO by exact opaque task ID;
6. checkpoint one immutable comparator result.

Mechanics requires the complete Aug10 wrapper and independently verified
mechanics artifact. Development requires the replay-verified four-block
combined result. Confirmation requires the replay-verified four-block combined
result and therefore inherits the frozen development authorization. A missing,
partial, stale, null-authorizing, or mismatched predecessor fails before the
endpoint-labelled task loader is called. Tests enforce this with a bomb loader.

The adapter reports both DINO policies, Luna myopic minus DINO myopic, and Luna
dynamic depth two minus DINO depth two for Brier, log loss, accuracy, and truth
probability. It uses the frozen 20,000-draw paired bootstrap. Lower differences
favor Luna for Brier/log loss; higher differences favor Luna for accuracy/truth
probability. Results are reported regardless of direction.

## Mandatory Paper Wrapper

After a verified development or confirmation result, the complete paper path
is now `scripts/bongard_openworld_paper_with_dinov2.py`, not direct invocation
of the old renderer. The wrapper:

- independently executes the original hash-bound Luna renderer in a temporary
  directory;
- independently replays the saved DINO comparator result from the same stage
  and block artifacts;
- preserves the original Luna text and headline file byte-for-byte;
- appends a deterministic DINO paragraph with myopic Brier, depth-two Brier,
  their paired difference and interval, and Luna-dynamic minus DINO-depth-two
  difference and interval;
- writes combined TeX and metadata with both provenance chains.

For a confirmation mechanics failure, no DINO endpoint result can exist. The
wrapper preserves the frozen no-efficacy interpretation and states that the
classical endpoint comparison is unavailable. Supplying a DINO result on that
path is invalid.

The addendum never changes headline authorization. Only the original literal
`full_llm_native_confirmation` conjunction can populate the abstract and
contribution macros. A direct old-renderer fragment that omits DINO is
scientifically incomplete and must not be used as the final manuscript output.

## Stage Sequence

After a successful Aug10 mechanics wrapper, run the zero-call DINO mechanics
adapter once using that wrapper and mechanics result. It cannot authorize
development and is descriptive mechanics evidence.

After the verified Development64 combined result and claim report:

1. run the DINO development outcome adapter once with all four block results;
2. run the combined paper wrapper with that immutable DINO result;
3. never invoke or publish a direct Luna-only fragment.

After any authorized verified Confirmation96 combined result, apply the same
sequence using the confirmation stage and four confirmation blocks.

Any failed adapter or renderer verification is banked. It does not permit a
Luna rerun, endpoint reissue, altered DINO plan, omitted comparator, or manual
paper repair.

## Bindings

- DINO baseline protocol SHA-256:
  `578e2ce911457a0c33f275e3f982f78dfa6d6216c30595277415e778264bbde8`;
- original Luna renderer SHA-256:
  `2083e7f93de8a8acd939f4842ac6f5dbeef95d5219fe3ad7b866d3bc4f507a74`;
- DINO outcome adapter SHA-256:
  `2987eee93a5b797a8e621071c20008d12e3781f133ea1e8be0c9e8fb1c9253af`;
- combined paper wrapper SHA-256:
  `b1b82feb983d6091e427ed6ea720f37b49aa3497120df728513253be00a703d6`;
- outcome tests SHA-256:
  `1c15eb4c0877e560aa45a3bc546f25f9d6cd5ebab1871e5f091565f10b60ca3c`;
- paper-wrapper tests SHA-256:
  `bd9a58e95e40d9dcaca932bd34323d50dcfa3480e9c6963ce211082a8e641739`.

The complete Bongard regression suite passes `190/190`; the six new focused
adapter and wrapper tests are included in that total.

This amendment authorizes no paid call, no endpoint opening, no new claim, and
no result-dependent model or policy change.
