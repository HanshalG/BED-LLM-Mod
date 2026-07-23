# Mushroom Predictive-Evidence GPT-5.4 Mini Smoke Result

Status: **failed preregistered S0 quality gate; line stopped without rerun.**

Seed `24176` completed all ten logical cells on the first response with legal policies,
zero projection, zero reasoning tokens, zero forced exits, and no scoring-time model
calls. The prompt exposed only branch-local predictive observation probabilities and
post-observation class probabilities; it contained no expected entropy, information
gain, action rank, preferred action, or utility field.

The model chose an exact-optimal continuation on `139/172` positive-probability
branches (`80.81%`), passing the absolute `50%` criterion. The registered uniform-menu
reference was unexpectedly high (`70.64%`) because `104/172` branches had every legal
continuation tied under the empirical posterior. The observed advantage was therefore
only `10.17` percentage points, below the frozen `20`-point requirement.

The preregistered gate is failed as written. No seed, threshold, or prompt was changed,
and the conditional proposal-quality run was not launched.

## Descriptive mechanism audit

These quantities were not registered gates:

- Among the `68` branches with nonzero continuation-value range, GPT selected an exact
  optimum on `35/68` (`51.47%`), versus a `25.75%` uniform-menu reference.
- Unweighted mean regret relative to the exact branch continuation was `0.03336` nats,
  versus `0.07200` for uniform menu choice, recovering `53.66%` of that gap.
- Weighting each branch by its root-outcome probability gave `72.99%` recovery.
- Only `11/40` root policies used different follow-ups across their outcome branches.

The raw predictive representation therefore conveyed substantial value, particularly
on probable branches, but the model often collapsed an observation-contingent policy
to one globally plausible feature. This supports the paper's broader mechanism: exact
utility grounding repairs ranking fidelity, while an unscored belief dump leaves
branch-conditioned aggregation unreliable.

## Usage

- Model: `openai/gpt-5.4-mini`, non-thinking, temperature zero
- Requests: `10`
- Prompt tokens: `238,211`
- Completion tokens: `520`
- Cost: `$0.17391345`
- Project spend after run: `$38.50724716 / $110`

Raw responses, reconstructed contexts, compiled policies, mechanics, and usage are in
`mushroom_predictive_evidence_gpt54mini_smoke_20260723/SMOKE.json`.
