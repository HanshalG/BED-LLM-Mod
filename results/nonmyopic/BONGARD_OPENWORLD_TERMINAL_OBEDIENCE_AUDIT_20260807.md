# Bongard Terminal Label-Obedience Audit

Date: 2026-08-07. This audit and correction were completed before any Bongard
model response or scientific endpoint was opened.

## Finding

The frozen simulated-branch gate checks that a one-answer generated belief
obeys the supplied positive or negative label. The terminal generator is a
separate LLM call after the policy has observed two real query labels. The
protocol previously verified that both labels appeared in the prompt and in
the parsed history, but it did not require the regenerated terminal predictive
mixture to retain them.

That left a measurement-validity gap. A terminal generator could ignore both
new observations while still producing a schema-valid belief. A favorable
held-out endpoint difference from such beliefs would not establish sequential
learning through path-dependent LLM belief updates.

## Correction

The frozen amendment
`BONGARD_OPENWORLD_LUNA_TERMINAL_OBEDIENCE_AMENDMENT.md`, SHA-256
`0706ff63310ee3b2c7ca603cadf43308d0e427868a11ae99959d81d126219daf`,
adds a terminal mechanics gate. For every distinct terminal belief it:

1. verifies that the four initial labels are unchanged;
2. identifies the exactly two newly queried labels;
3. recomputes the predictive mixture on those two queried images; and
4. pools Brier scores separately for positive and negative queried labels.

Both label classes must be present and each class-conditional mean Brier must
be strictly below `0.25`, the constant-half baseline. The metric is persisted
and independently replayed in mechanics, development, and confirmation.

The correction adds no model request and changes no prompt, response schema,
task, image, policy, action, seed, batch, token, date, budget, endpoint, or
scientific efficacy threshold. It can only stop an invalid run; it cannot turn
a null endpoint into a positive one.

## Why This Is Necessary

Classical sequential BED and DAD condition each design on the accumulated
history. BED-LLM explicitly constructs and updates a probabilistic belief from
the conversation history, while CA-BED propagates beliefs and expected
information through simulated multi-turn conversations. In this experiment
the transition operator is itself an LLM generation, so history presence in a
prompt is not enough: the generated predictive state must measurably reflect
the observations it claims to condition on.

This gate is necessary but not sufficient. It establishes observation
retention, not calibration, policy efficacy, or a path-dependent advantage.
Those remain governed by the frozen held-out Brier, matched fixed-score,
fixed-support, history-blind, and ranking-fidelity gates.

Primary references:

- Foster et al., Deep Adaptive Design:
  <https://proceedings.mlr.press/v139/foster21a.html>
- Choudhury et al., BED-LLM: <https://arxiv.org/abs/2508.21184>
- Arnould et al., CA-BED: <https://arxiv.org/abs/2606.01182>

## Authoritative Bindings

- Development interface: `bongard-openworld-luna-vlm-development32-11`.
- Development manifest SHA-256:
  `a0b70ff8bbe3e36eba56b357e563504f12e4792d92cedee237d4b261d15a7708`.
- Confirmation manifest V7 SHA-256:
  `02c38bf7e27d7b825fe608a171b975fa43a8bb7015eb9bb1b2089b61881969d6`.
- Confirmation execution core SHA-256:
  `0235ceb2d7b41e84c9258f5d34723de41e092df3300231e164facb860230a05e`.

All earlier manifests remain immutable historical artifacts and cannot
authorize the current execution path.

## Verification

- A clean terminal fixture passes.
- An adversarial terminal generator that fits positive labels but ignores
  negative labels fails the new class-balanced gate.
- Full Bongard suite: 120 passed in 57.88 seconds.
- Development manifest verification passes.
- Independent confirmation V7 protocol verification passes all 12 checks.
- Confirmation execution-binding verifier v2 passes.
- Authenticated August 10 preflight returns `ready_without_paid_calls` with
  Luna live at `$0.10/$0.60` per million input/output tokens, balance
  `$24.886393846`, model calls 0, and files written 0.

The next permitted Bongard paid action remains the exact frozen August 10 Luna
serving and mechanics sequence.
