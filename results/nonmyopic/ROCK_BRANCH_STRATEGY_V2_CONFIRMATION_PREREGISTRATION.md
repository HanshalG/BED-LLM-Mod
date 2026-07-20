# Rock Branch-Policy StrategyEIG Confirmation Preregistration

Registered 2026-07-20 before launching or inspecting any arm outcome from the formal run.

## Question

Does an LLM proposal prior over explicit two-action branch policies, followed by exact
rollout-EIG verification, improve sequential Bayesian experimental design in Rock
Diagnosis relative to controls that remove either lookahead or LLM strategy quality?

## Frozen Design

- Maps: paper maps `3-6` and `5-7`, analyzed separately.
- Trials: 30 paired trajectories per map.
- Rounds: 8 per trajectory.
- Seed: `12041`, unused by the engineering smokes.
- Strategy count: K=6 per decision.
- Planning: receding horizon 2, shortened to horizon 1 only at the final round.
- Generator: `google/gemma-4-26b-a4b-it`, non-thinking, temperature 0.
- Schema: `branch_policy_v2` with exactly three distinct movement roots followed by
  direct checks and three direct-check roots at horizon 2. At horizon 1, all six roots
  are distinct and followups are empty.
- Verification: finite exact posterior, exact branch enumeration, exact rollout EIG,
  and deterministic argmax tie-breaking.
- Pairing: all arms share each trial's latent rock state. Observation uniforms are
  common for matching position/rock/repetition coordinates and deterministic from the
  registered seed.
- Inference: 10,000 paired nonparametric bootstrap replicates, with deterministic
  comparison-specific bootstrap seeds.

## Arms

1. `strategy_eig`: Gemma proposes K branch policies; exact depth-2 EIG selects the root.
2. `exhaustive_d2`: exact unrestricted depth-2 action lookahead, as an oracle reference.
3. `shared_d1`: the same LLM strategy cell and roots as StrategyEIG at a shared state,
   scored only by immediate exact EIG. After histories diverge, each arm regenerates on
   its own realized state.
4. `width`: one LLM legal-action ordering, exhaustive one-step EIG over all distinct
   legal actions, with repeated exact evaluations to match StrategyEIG scorer units.
5. `random_strategy`: K legal branch policies sampled from the identical registered
   root-slot grammar, followed by the same exact depth-2 scorer.

## Endpoints

The primary endpoint is paired entropy-AUC gain, implemented as the baseline's mean
posterior entropy over rounds minus StrategyEIG's mean posterior entropy over rounds.
Positive values favor StrategyEIG. It is evaluated separately for each map and each of
`shared_d1`, `width`, and `random_strategy`.

The corroborating endpoint is paired truth-log-posterior-AUC gain, implemented as
StrategyEIG's mean log posterior probability on the true rock vector over rounds minus
the baseline's value. Final entropy, final truth log probability, final MAP accuracy,
round traces, and comparison with `exhaustive_d2` are secondary diagnostics.

## Success Criteria

The registered positive result requires all six primary comparisons (three controls on
each of two maps) to have a strictly positive lower endpoint of the paired 95% bootstrap
CI for entropy-AUC gain.

Truth-log-posterior AUC corroborates the claim when its point estimate is positive for
all six comparisons and none of its six 95% CIs is entirely below zero. Failure of this
corroborating condition is reported as endpoint disagreement even if the primary gate
passes. No pass criterion is imposed against the exhaustive-d2 oracle; its role is to
measure proposal coverage.

## Mechanics And Repair Policy

The run is valid only if every selected action is legal, no terminal strategy cell
fails, initial StrategyEIG/shared-d1 cells are shared, width matches both logical LLM
calls and exact scorer units, every random cell has K policies, and rollout scoring
makes zero LLM calls.

The parser selects the model's last complete fenced JSON object, if fenced, then applies
the full registered count, legality, branch, and behavioral-distinctness validation.
One semantic feedback retry is allowed. At horizon 1 only, nonempty followups are
deterministically replaced by `{}` because they lie outside the scoring horizon and
cannot change the root action or exact score; the number of repaired strategy items is
logged. No root, missing strategy, duplicate behavior, horizon-2 branch, or illegal
action is repaired.

## Cost And Stop Rules

The clean 16-request mechanics smoke cost `$0.00417305`; scaling its observed request
rate projects approximately `$0.36`. The run declares `$0.50` projected cost and has a
hard `$1.00` run cap inside the cumulative `$40` ledger. Transport requests use a
60-second timeout and five retries. A failed-closed or interrupted run produces no
confirmatory conclusion; outcomes from a partial run are not inspected.

## Frozen Command

```bash
set -a; source .env; set +a
python scripts/nonmyopic_rock_strategy_prior.py \
  --config configs/config_nonmyopic_rock_branch_strategy_confirmation_openrouter.yaml \
  --run-id nonmyopic-rock-branch-strategy-v2-confirmation-20260720 \
  --output-dir results/nonmyopic/rock_branch_strategy_v2_confirmation_20260720 \
  --num-trials-per-map 30 \
  --num-rounds 8 \
  --num-strategies 6 \
  --seed 12041 \
  --bootstrap-replicates 10000 \
  --trial-concurrency 32 \
  --strategy-schema branch_policy_v2 \
  --primary-endpoint entropy_auc
```

## Engineering Evidence

- Actual 10-cell serving/quality gate:
  `results/nonmyopic/rock_branch_strategy_smoke/gemma26b_k6_terminal_gate_20260720/SMOKE.json`
  (`10/10` accepted first pass; all depth-2 cells captured the exhaustive depth-2 value).
- Clean end-to-end mechanics gate:
  `results/nonmyopic/rock_branch_strategy_mechanics_smoke/recovery6_20260720/L1.json`
  (`16/16` accepted, zero raw rejects, all mechanics true, seven logged terminal-item repairs).
