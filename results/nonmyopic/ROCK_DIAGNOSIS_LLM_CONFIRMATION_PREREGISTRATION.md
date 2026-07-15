# Rock Diagnosis LLM Depth Confirmation Preregistration

Registered: 2026-07-15, after one completed exploratory pilot and before any
confirmation request.

## Claim Under Test

With the LLM restricted to three legal action proposals at each exact Rock Diagnosis
state, two-step exact incremental EIG produces lower final exact posterior entropy
than both shared-cell one-step EIG and a one-step policy given the same number of LLM
candidate-proposal calls for current-state width.

## Frozen Configuration

- Independent confirmation seed: `6173`; 30 paired trajectories; 8 action rounds;
  Figure 4 `3-6` map; fixed start `(0, 3)`; candidate width `K=3`.
- Same `google/gemma-4-26b-a4b-it` non-thinking OpenRouter candidate proposer,
  temperature `0`, maximum 128 output tokens, and exact prompt format as the pilot.
- Same exact 8-state posterior, `pomdp_py==1.3.5.1` dynamics/likelihood, latent
  full-vector target, MAP decode, CRN observation keys, and arms
  `d1_shared`, `d2`, and `d1_call_matched_width`.
- Same strict parser and one validation-feedback retry. The code never pads,
  substitutes, or programmatically repairs action cells; a second invalid completion
  is terminal.
- Frozen runner/analyzer: `scripts/nonmyopic_rock_diagnosis_pilot.py --confirmatory`
  and this configuration, committed before launch. It writes per-call raw candidates,
  per-step traces, paired percentile bootstrap intervals (10,000 resamples), and a
  compact Markdown report.

## Power And Cost

The eight-trajectory exploratory retry run estimated paired final-entropy reductions
of `0.3841 +/- 0.2123` SD versus shared one-step and `0.3775 +/- 0.2063` SD versus
call-matched width (paired standardized effects `1.81` and `1.83`). A two-sided normal
approximation gives 90% power at four pairs for each contrast. The fixed 30 paired
trajectories deliberately provide a greater than seven-fold margin over that
pilot-derived minimum to absorb winner's-curse and distributional uncertainty.

The observed pilot cost was `$0.03054` for 8 trajectories, implying `$0.11451` for
30. This run projects `$0.18` and has an adapter-enforced `$0.35` hard cap. The prior
failed interface attempt and completed exploratory retry cost `$0.04975` combined;
the full screen plus confirmation remains far below the `$1` exploration-scale cost.

## Analysis And Decision Rule

Primary outcome: paired final exact posterior-entropy reduction, defined as
`H(control) - H(d2)`, separately for shared one-step and call-matched width. A
positive value favors depth two.

The confirmation passes only if all of the following hold:

1. both paired 95% percentile-bootstrap intervals have lower bounds strictly above
   zero;
2. every selected action is legal, all root cells are shared, and width's logical
   candidate-call count equals depth two's virtual root-tree allocation at every
   decision; and
3. no candidate cell exhausts its one validation-feedback retry.

Secondary outcomes are entropy AUC, MAP accuracy, true-vector log posterior, root
movement rate, selected EIG, candidate diversity, retry count, token usage, cost, and
full action traces. The complete report is read only after execution finishes. No
other confirmation or environment/model/hyperparameter search will be launched before
this one is recorded.
