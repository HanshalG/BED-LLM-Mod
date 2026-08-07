# Bongard-OpenWorld Partition Integrity Amendment

Date frozen: 2026-08-07, before any Bongard serving, mechanics, development,
or confirmation model response.

## Problem

The source audit proved unique metadata paths and disjoint source rows, but it
did not compare image bytes across those paths. A complete archive audit found
that the original seeded validation partition contained:

- two development rows with a byte-identical image repeated inside the task;
- four byte-identical image groups shared by development and confirmation;
- one additional confirmation row sharing image bytes with another
  confirmation row.

Requests are stateless across tasks, so this is not model-context leakage. It
does violate the intended image-level independence and can make task-bootstrap
uncertainty anti-conservative. No paid response, scientific endpoint, or model
output existed when this issue was found.

## Frozen Repair

Keep the source audit's exact seeded row ordering. Fill mechanics,
development, and confirmation in that order to sizes 4, 32, and 64. Reject a
row if either:

1. two of its 14 source paths contain byte-identical image data; or
2. any image is byte-identical to an image in an earlier accepted
   experimental row.

Rejected rows move to reserve in encounter order, and the next row in the
already frozen seeded ordering fills the target partition. The rule uses no
labels, concepts, captions, model responses, or endpoint outcomes.

This leaves mechanics unchanged, replaces 2 of 32 development rows, and
replaces 7 of 64 confirmation rows. The repaired partitions retain their
original sizes. Their UID hashes are:

- mechanics: `69de5a5fc444f3d14e9c07f570f4fe1fa8a4e915dd15926466c7c677d68bd323`;
- development: `356365ce90c964f6e99de5ca1585f9f582eebf8d4043afb55d1aee2a96c7c99b`;
- confirmation: `1537b43d37e03287520bd1c8bd583e7a7d4680c09ba2203e8238c8831205c631`.

## Post-Selection Checks

The repaired development and confirmation partitions have:

- zero byte-identical images within or across selected experimental tasks;
- zero exact normalized concept or caption matches across the boundary; and
- zero strict perceptual near matches at both aHash Hamming distance at most 2
  and dHash Hamming distance at most 2.

Semantic and perceptual checks are descriptive gates after the byte-only
selection. They do not select tasks. This strict screen cannot prove broad
semantic independence, but it closes the observed exact-content failure.

## Binding

All unopened development and confirmation loaders, manifests, verifiers, and
execution wrappers must bind the supplementary partition-integrity manifest
and repaired partition hashes before a request. Historical source and image
integrity manifests remain immutable evidence of the discovery path. Any
prior development or confirmation manifest using the original partition is
superseded and cannot authorize calls.

This amendment changes no model, prompt, policy, random seed, request count,
budget, date, endpoint, threshold, or claim gate. It authorizes no paid call by
itself.
