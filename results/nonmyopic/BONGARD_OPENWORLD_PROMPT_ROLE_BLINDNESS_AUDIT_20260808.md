# Bongard OpenWorld Prompt-Role Blindness Audit

Date: 2026-08-08 (Europe/London)

Status: **pass; zero model calls and zero scientific endpoint access**.

This audit was run before any Bongard mechanics, development, or confirmation
response. It changes no model, task, seed, action, endpoint, request count,
science gate, threshold, or budget. It does not authorize paid calls.

## Threat

The belief VLM sees all 14 task images so it can provide likelihoods for the
two final evaluation images. Candidate and endpoint IDs are not explicitly
marked in the prompt. That protection would nevertheless be ineffective if a
stable public image position or the numeric suffix of an opaque ID reliably
revealed which unlabelled images were endpoints.

The source rows do have a private convention: class index 6 is held out as the
endpoint. The loader first assigns every source position a task-specific
shuffled opaque ID, however, and then presents the sorted opaque IDs. Neither
the source path nor source position enters the public request.

## Executable Audit

The audit loads endpoint-sealed mechanics (4 tasks), development (64), and
confirmation (96) partitions and verifies:

- every task has 4 initial, 8 candidate, and 2 endpoint images with disjoint
  roles;
- endpoint truth is absent from every loaded task;
- all tasks use only `image-00` through `image-13` and expose no role field;
- each of the 14 public prompt positions occurs as initial, candidate, and
  endpoint somewhere in the frozen 164-task corpus;
- the request `image_order` exactly matches the attached task order;
- sampled prompts from every partition pass the existing hidden-state scan.

As an out-of-sample predictability check, a slot/suffix-only endpoint model is
fit on mechanics plus development. It estimates
`P(endpoint | public position, unobserved)` with a Beta(1, 4) prior and is
evaluated once on all 960 unobserved confirmation rows.

## Result

| Confirmation diagnostic | Public-position model | Constant 20% prior |
|---|---:|---:|
| AUC | 0.525258 | 0.500000 |
| Brier score | 0.159914 | 0.160000 |
| Log loss | 0.500228 | 0.500402 |

A forced top-two slot guess identifies 36 of 192 confirmation endpoints:
18.75% precision and 18.75% recall, below the 20% endpoint base rate. Every
fail-closed gate passes. There is no useful held-out role signal in public
prompt position or opaque-ID suffix, so the feared fixed-last-slot leakage is
not present and the frozen tasks do not need a prospective reshuffle.

## Limits

This result does not prove that a frontier VLM has never memorized a public
Bongard-OpenWorld image, nor does it test whether private source rank has a
visual artifact. It establishes the narrower interface claim: source role is
not encoded in the public IDs, image order, or request metadata. Those limits
must remain explicit in the paper and cannot be upgraded into a semantic or
planning-quality claim.

Executable manifest:
`results/nonmyopic/bongard_openworld_prompt_role_blindness_audit/bongard-openworld-prompt-role-blindness-audit-20260808/MANIFEST.json`

Manifest SHA-256:
`40077db8f1caccdd9eafd0f6b0d714b32bebf8319d43dd773b3b08e188bc6857`
