# Bongard Matched Fixed-Score First-Action Audit

Date: 2026-08-07. This audit and correction were completed before any Bongard
model response or scientific endpoint was opened.

## Finding

The original `dynamic_depth2` versus `fixed_depth2` comparison was a fair
complete-policy comparison, but not a clean test of the first planning link.
The two policies changed both:

1. the score used to choose query one; and
2. the updater used to choose query two after the real first answer.

Dynamic depth two uses an answer-conditioned regenerated semantic support at
stage two. Fixed depth two analytically updates and continues on the root
support. A dynamic endpoint win could therefore come entirely from the better
realized updater, even if anticipating path-dependent support selected a worse
first query.

This matters because the intended irreducible claim is specifically that the
first query benefits from looking ahead through the LLM's answer-conditioned
belief generator.

## Correction

The frozen amendment
`BONGARD_OPENWORLD_LUNA_MATCHED_FIXED_SCORE_AMENDMENT.md`, SHA-256
`1f860d135663bd370fb140c7fe402bdef37c25b998955d4a7fef947d6b8099a2`,
adds `fixed_score_dynamic_update`:

- query one is selected by the existing fixed-support depth-two score;
- query two is selected from the same realized answer-conditioned regenerated
  branch used by `dynamic_depth2`;
- the terminal belief, endpoint, requested task-level seed, and dispatch batch
  are otherwise shared.

The all-first-action audit already generates that terminal history for every
possible first query. The control therefore adds zero VLM requests and leaves
all request maxima, prompts, model seeds, budgets, and endpoint access rules
unchanged.

The strongest development and confirmation families remain conjunctive with
the original fixed-policy gates and now also require changed non-tied first
actions, every-block differences, at least 3% relative Brier improvement,
paired uncertainty, and non-worse log loss against the matched control.

## Adversarial Regression

A synthetic confirmation fixture assigns endpoint Brier `0.10` to dynamic,
`0.12` to the complete fixed policy, and `0.08` to the matched fixed-score
control. Dynamic therefore passes every old fixed-policy efficacy gate while
losing the matched first-action comparison. The corrected classifier returns
`confirmation_null` and does not authorize the first-action claim.

This adversary is also represented at the development claim boundary: all old
path-dependent gates can pass while one matched-control gate fails, yielding
the non-authorizing partial tier
`policy_and_matched_regeneration_without_fixed_support_superiority`.

## Literature Alignment

BED-LLM identifies model formulation and belief updating as critical, showing
that raw in-context belief updates can be inconsistent and that explicit
prior-likelihood EIG plus filtering is substantially stronger. It also notes
that the true sequential objective is terminal belief quality, while
intermediate beliefs matter through future decision making:
<https://arxiv.org/abs/2508.21184>.

CA-BED performs explicit multi-turn lookahead with LLM likelihoods and reports
that planning quality depends on calibrated likelihoods and coherent answer
spaces:
<https://arxiv.org/abs/2606.01182>.

The matched control applies the corresponding experimental principle here:
hold the realized belief updater fixed when testing whether a different
lookahead model improved the upstream action.

## Authoritative Bindings

- Development interface: `bongard-openworld-luna-vlm-development32-9`.
- Development manifest SHA-256:
  `4785d95e470285d380780dd0f1254c22994e81fb2d1cd0ec21285eee6c6e74ff`.
- Claim-report interface: `bongard-openworld-luna-claim-report-3`.
- Confirmation manifest V5 SHA-256:
  `d827a9fdd6694bfd69ba05550e5e1248f9750b5785be1f6be5e9e5a73e1d8b24`.
- Confirmation execution core SHA-256:
  `81da7cce28220b29b7f029d85c9793d8374fb26d6ec77bb42b6811bbd28ae6d5`.
- Confirmation execution amendment SHA-256:
  `0f284c475e04c58f546d21baf9d2f0b41975832c0f11b1fc54317d3de124a351`.

Confirmation manifests V1 through V4 remain immutable historical artifacts;
V5 is the only authoritative execution binding.

Verification: the complete Bongard suite passes 117 tests in 53.68 seconds;
development, confirmation protocol, and confirmation execution verifiers all
recompute the exact hashes above. The authenticated August 10 preflight is
`ready_without_paid_calls`, with live balance `$24.886393846`, component-cap
sum `$2.00`, model calls 0, and files written 0.

## Boundary

This correction makes no scientific claim and spends `$0`. The next permitted
paid action remains the frozen August 10 Luna serving and mechanics sequence.
Only a later endpoint result that beats the matched control can support the
first-query path-dependent-lookahead claim.
