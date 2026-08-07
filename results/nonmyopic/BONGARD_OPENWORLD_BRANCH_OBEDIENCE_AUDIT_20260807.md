# Bongard Simulated-Branch Obedience Audit

Date: 2026-08-07. This audit and correction were completed before any Bongard
model response or scientific endpoint was opened.

## Finding

The existing branch-sensitivity gate was necessary but not sufficient. It
correctly ignored the newly labelled query when testing whether positive and
negative branch calls changed the remaining belief state. However, a model
could pass that test by emitting different rules or predictions while still
assigning, for example, probability `0.9` to the query being positive under
both simulated labels.

An adversarial fixture demonstrates the gap exactly. It passes the old
material branch-sensitivity check because the two calls change unobserved
predictions and rule support, but its branch-label Brier scores are `0.01` for
the positive branch and `0.81` for the negative branch. The negative simulated
answer has therefore not been incorporated into the purported posterior.

## Correction

The frozen amendment
`BONGARD_OPENWORLD_LUNA_BRANCH_OBEDIENCE_AMENDMENT.md`, SHA-256
`18d81a18dea963b71cce3fc3395d72a1c46a29c0631fdd5f93499b152ff36e06`,
adds a separate class-balanced mechanics gate. For answer-conditioned branch
calls, mean Brier on the newly supplied query label must be strictly below
`0.25` in both the simulated-positive and simulated-negative classes. Each
branch history must contain the exact supplied label.

The matched history-blind control is excluded because its prompt intentionally
omits that answer and its frozen analytical update remains unchanged. The
original still-unobserved branch-sensitivity gate also remains unchanged.

This creates three distinct validity requirements:

1. The generated rule must contrast the official positive and negative
   classes.
2. Opposite simulated answers must materially alter the remaining belief.
3. The resulting branch belief must itself respect the supplied answer.

The correction adds no model calls and changes no task, image, prompt, response
schema, policy, width, branch weight, seed, date, budget, endpoint, or science
threshold.

## Authoritative Bindings

- Development interface: `bongard-openworld-luna-vlm-development32-8`.
- Development manifest SHA-256:
  `64f80983b3922556c279982fdbf966a861046345f698f2196cf823077b14ba46`.
- Confirmation manifest V4 SHA-256:
  `622ad102a2ed22a7e67722532902a4720012abf4852a60af1282f433d0f2317f`.
- Confirmation execution core SHA-256:
  `d226ff26831cb1659c7e14fb68c1e7b45d228ef20ae6ac284fb2dbdcd398a39a`.
- Confirmation execution amendment SHA-256:
  `bac2c7dbb27e4dfc4fc43a59edbe31a4602b19a3924ecacb6062219654f26164`.

Confirmation manifests V1 through V3 remain immutable historical artifacts;
V4 is the only authoritative confirmation execution binding.

## Verification

- Adversarial old-gate/new-gate regression passes.
- Full Bongard suite: 115 passed in 44.45 seconds.
- Development manifest verification passes with exact SHA-256 above.
- Independent confirmation protocol and execution-binding verification pass.
- Authenticated August 10 preflight: `ready_without_paid_calls`; live balance
  `$24.886393846`; component-cap sum `$2.00`; model calls 0; files written 0.

The next permitted paid action remains the frozen August 10 Luna serving and
mechanics sequence. This audit does not authorize early development,
confirmation, model substitution, or endpoint access.
