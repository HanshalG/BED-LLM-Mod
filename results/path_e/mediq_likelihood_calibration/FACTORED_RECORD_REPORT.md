# MediQ Likelihood Calibration Replay

Verdict: **FAIL**

This is a frozen-action likelihood diagnostic, not policy-efficacy evidence.

## Summary

| Metric | Value |
|---|---:|
| `num_tasks` | 5 |
| `num_turns` | 10 |
| `num_available_turns` | 6 |
| `num_unavailable_turns` | 4 |
| `mean_eig` | 0.006899 |
| `mean_realized_entropy_drop` | 0.126370 |
| `mean_truth_log_probability_gain` | -0.070874 |
| `available_mean_truth_log_probability_gain` | -0.118123 |
| `available_true_label_favored_rate` | 0.666667 |
| `maximum_unavailable_likelihood_span` | 0.000000 |
| `maximum_unavailable_posterior_linf_change` | 0.000000 |

## Checks

- [x] `frozen_replay_shape`
- [x] `unavailable_is_label_independent`
- [x] `unavailable_does_not_move_posterior`
- [x] `available_true_label_favored_rate_at_least_60_percent`
- [ ] `available_mean_truth_log_gain_positive`
- [ ] `overall_mean_truth_log_gain_nonnegative`

## Turns

| Task | Round | Outcome | EIG | Truth log gain | True favored | Query |
|---|---:|---|---:|---:|---|---|
| `mediq:imedqa:0` | 1 | Information unavailable / not in record | 0.0000 | +0.0000 | no | Is the joint swelling limited to a single joint? |
| `mediq:imedqa:1` | 1 | Yes | 0.0364 | +0.3642 | yes | Has the child had similar episodes in the past? |
| `mediq:imedqa:2` | 1 | Information unavailable / not in record | 0.0009 | +0.0000 | no | Has the patient experienced excessive worry? |
| `mediq:imedqa:3` | 1 | Yes | 0.0008 | +0.0507 | yes | Is the patient's temperature above 38.0 degrees Celsius? |
| `mediq:imedqa:4` | 1 | Yes | 0.0036 | -1.2432 | no | Was the patient's blood glucose level above 250 mg/dL? |
| `mediq:imedqa:0` | 2 | Information unavailable / not in record | -0.0000 | +0.0000 | no | Has the patient had a recent history of sexually transmitted infections? |
| `mediq:imedqa:1` | 2 | Yes | 0.0272 | +0.1196 | yes | Has the child had abdominal pain during these episodes? |
| `mediq:imedqa:2` | 2 | Information unavailable / not in record | 0.0001 | +0.0000 | no | Does the patient experience muscle tension? |
| `mediq:imedqa:3` | 2 | No | 0.0000 | -0.0001 | no | Is the patient's blood pressure below 90 mmHg systolic? |
| `mediq:imedqa:4` | 2 | Yes | 0.0000 | +0.0000 | yes | Is the patient's arterial pH below 7.35? |
