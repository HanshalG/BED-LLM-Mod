# AutumnBench LLM-Native Source Admission Protocol

Date frozen: 2026-08-13, before downloading the public dataset manifest.

## Objective

Determine whether AutumnBench can support a fresh non-myopic BED experiment in
which an LLM generates semantic/programmatic hypotheses about an unknown world's
dynamics, while a local interpreter supplies exact observations and sealed derived
tests.

This is source admission only. It may not inspect task programs, prompts, answers,
or model responses. A pass authorizes only a separately frozen, zero-call
horizon-opportunity screen.

## Immutable Sources

- `BasisResearch/Autumn.cpp` commit
  `0929d175c3d296de091ac4b998d09b9b55a38f67`;
- `BasisResearch/MARAProtocol` commit
  `d0d4e9251151778415cc5195c5dfb8296a5ded24`;
- Firebase public-manifest URL used verbatim by the pinned MARA downloader:
  `https://firebasestorage.googleapis.com/v0/b/prolific-server-123.firebasestorage.app/o/manifest_public.json?alt=media`.

The audit may read repository trees, source interfaces, and the manifest object.
It may read and execute only the already versioned public `ice` example fixture.
For all other tasks it may inspect manifest metadata only. It must never download
their programs, prompts, or answers during source admission.

## Frozen Gates

All gates must pass:

1. **Bindings:** both commits and complete Git tree hashes match the frozen values;
   both repositories are MIT licensed.
2. **Population:** the public manifest contains exactly 129 unique task IDs and
   exactly 43 unique base-program IDs.
3. **Complete triplets:** every base program has exactly one task of each canonical
   type: `masked_frame_prediction`, `change_detection`, and `planning`.
4. **Path separation:** the pinned downloader maps each task to distinct
   `programs/`, `prompts/`, and `answers/` paths; source admission downloads only
   the manifest.
5. **Native experiment interface:** pinned code exposes seed propagation, ordinary
   environment actions, `noop`, `reset`, and `go-to-test`; reset reconstructs the
   interpreter with the same program and seed.
6. **Sealed derived tests:** pinned code loads interaction programs separately from
   MFP answers, change-detection answers/wrong programs, and planning goals; the
   interaction phase does not load those derived outcomes.
7. **Local deterministic handshake:** on the pinned public `ice` fixture, two fresh
   interpreter instances under seed `20260813` produce byte-identical initial and
   post-action observations for `right, noop, down`; a reset exactly restores the
   initial observation in both arms.
8. **No privileged publication:** the public result contains only repository and
   manifest hashes, aggregate counts, type histogram, tree hashes, interface booleans,
   fixture replay hashes, and gates. It contains no non-fixture task ID, base-program
   ID, program text, prompt, answer, observation, goal, or correct option.

## Decision

- `source_pass`: authorize only a separately frozen zero-call opportunity screen
  over an opaque hash-ordered mechanics cohort. That screen must establish a real
  depth-two changed first action and endpoint separation before any model call.
- otherwise: close this exact source route without task payloads or model calls.

Even after a pass, an LLM-serving design must separately require a value-free exact
provider/schema rehearsal, semantic hypothesis and likelihood calibration,
compute-matched myopic and random controls, common-random-number endpoints, and a
fail-closed account-wide budget wrapper.
