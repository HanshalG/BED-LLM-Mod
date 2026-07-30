# Number Game Qwen History-Blind Matched-32 V3 Preregistration

Date frozen: 2026-07-30

## Purpose

V3 is the fresh-seed successor to two endpoint-unaccessed failures:

- V1 failed one frozen auxiliary novelty gate after exact mechanics;
- V2 encountered a mechanics-dispatch recursion bug before gate evaluation.

V3 changes only the V2 wrapper dispatch. It captures the original V1
`mechanics_gates` function before entering the patch context and calls that
captured object, preventing recursion. The V2 scientific protocol and
resilient mechanics are unchanged.

No V1 or V2 response, parsed support, mechanic, or endpoint may be reused.

## Frozen Failure Bindings

- V1 `FAILURE.json` SHA256:
  `f394c889a56fdb20854d28cda0fa2927ae1a8d40e02c54f0f400f3f2757fdcb3`
- V2 `V2_RUNNER_FAILURE.json` SHA256:
  `4b3abc38059f11dceffb82ba1996d11ec7ecc1eae51616f4a2d63c895da2abd6`
- V2 controls SHA256:
  `f2c8c345eb0b1c3d7658372b5f9cb78d8b78ccd5b1b23b61ff96425d4a252e15`
- V2 private raw-response SHA256:
  `3057d850408338efdad1fe344d9210ebfd4fdbaefe174e65b3832113b4086994`
- V2 endpoint accessed: `false`

Source and smoke bindings remain:

- source result:
  `04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e`
- source trees:
  `8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82`
- source targets:
  `2fd09b75bcce734e17f237ced9a49e97e13e290e2f5229b9354ad7a8a1bdafd6`
- serving smoke:
  `0e4ba2e5bedd000d5b22c46e16a91404ec8913ba620ab54db286309fef9b8ef4`

## Fresh Seeds And Code Change

V3 seed schedule:

`9400000 + 1000 * tree_index + 2 * history_index + draw_index`

The V3 wrapper must:

1. bind both prior failure artifacts and require endpoint access `false`;
2. save `ORIGINAL_BASE_MECHANICS_GATES = base.mechanics_gates` at import;
3. have the V3 mechanics function call that captured object;
4. patch the base runner to the V3 interface, seed start, and V3 mechanics;
5. retain V2's descriptive second-draw novelty report.

No base scientific or scoring function changes.

## Required Tests Before Launch

- mechanics dispatch is called inside the configured V3 context with
  second-draw novelty below two and returns without recursion;
- the context restores every base global afterward;
- both prior failure bindings validate;
- the full synthetic 3,072-response path runs through V3 controls, mechanics,
  canonical scoring, bootstrap, and serialization;
- focused V1/V2/V3 tests pass.

## Frozen Mechanics And Science

Identical to V2:

- exact `32` trees, `1,536` slots, and `3,072` accepted requests;
- raw draw valid at least `16`, pooled support at least `24`;
- all draws strict JSON;
- retries/provider retries at most `32`;
- zero reasoning and forced exits;
- cost cap `$4.25`, starting balance at least `$5.00`;
- second-draw novelty descriptive only.

Scientific gates remain the four V1/V2 canonical quality gates. Bootstrap
seed/samples remain `9100000` / `20000`.

## Budget

- live authenticated balance after V2: `$9.001245349`;
- projected V3 cost from V1/V2: approximately `$3.21`;
- expected post-run balance: approximately `$5.79`.

V3 runs once after its exact wrapper and required tests are committed.
