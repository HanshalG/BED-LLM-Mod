# Paprika Step 1 Gap Pilot

Status: **claim1_matched_fail_rescue_or_stop**

| arm | resolution@5 | mean censored turns | coverage | cost (USD) | requests |
|---|---:|---:|---:|---:|---:|
| naive_nonthinking | 0.400 | 4.600 | 0.875 | 0.0050 | 124 |
| naive_thinking | 0.400 | 4.600 | 0.850 | 0.1015 | 137 |
| EIG | 0.300 | 5.200 | 0.844 | 0.2529 | 3365 |
| Full2StepEIG | 0.200 | 5.600 | 0.917 | 5.1571 | 70004 |

## Claim 1 matched: EIG vs naive non-thinking

Wins/losses/ties: 3/3/4. 
Mean paired censored-turn delta: 0.600; 95% bootstrap CI [-0.9, 2.1024999999999636]; Wilcoxon p=None.

## Claim 1 adversarial: EIG vs naive thinking

Wins/losses/ties: 1/4/5. 
Mean paired censored-turn delta: 0.600; 95% bootstrap CI [-0.7, 1.8]; Wilcoxon p=None.

## Claim 2: full2 vs EIG

Wins/losses/ties: 0/1/9. 
Mean paired censored-turn delta: 0.400; 95% bootstrap CI [0.0, 1.2]; Wilcoxon p=None.
