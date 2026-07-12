# Paprika Arbitration Headline

Automated status: **claim_b_not_confirmed_requires_manual_review**

Manual endpoint status: **INVALID**. Review found an exact private remedy falsely
reported as failing on task 13 in the best-N arm (replace/refill the empty receipt-paper
roll). Under the preregistered whole-headline rule, the complete five-arm result is
invalid as policy evidence. The automated values below are retained only as diagnostic
output from the frozen one-time analysis.

| arm | resolution@5 | mean censored turns | coverage | cost (USD) | cost/resolution | requests |
|---|---:|---:|---:|---:|---:|---:|
| arbitration | 0.320 | 5.120 | 0.883 | 2.0287 | 0.1268 | 15707 |
| candidate0 | 0.400 | 4.960 | 0.917 | 0.1933 | 0.0097 | 849 |
| naive_thinking | 0.400 | 4.680 | 0.975 | 0.5059 | 0.0253 | 1039 |
| naive_nonthinking | 0.380 | 4.900 | 0.967 | 0.0467 | 0.0025 | 1070 |
| best_n_eig | 0.320 | 4.960 | 0.930 | 2.6724 | 0.1670 | 23570 |

## Primary: arbitration vs thinking naive

Turn wins/losses/ties: 6/16/28.
Mean paired censored-turn delta: 0.440; 95% bootstrap CI [0.0, 0.9]; Wilcoxon p=None.
Resolution delta: -0.080; discordant wins/losses 3/7; exact p=0.34375.

## Co-primary: arbitration vs candidate 0

Turn wins/losses/ties: 6/12/32.
Mean paired censored-turn delta: 0.160; 95% bootstrap CI [-0.24049999999999983, 0.56]; Wilcoxon p=None.
Resolution delta: -0.080; discordant wins/losses 3/7; exact p=0.34375.

## Context: arbitration vs non-thinking naive

Turn wins/losses/ties: 8/12/30.
Mean paired censored-turn delta: 0.220; 95% bootstrap CI [-0.26, 0.7004999999999927]; Wilcoxon p=None.
Resolution delta: -0.060; discordant wins/losses 5/8; exact p=0.5810546875.

## Context: best-N EIG vs thinking naive

Turn wins/losses/ties: 11/12/27.
Mean paired censored-turn delta: 0.280; 95% bootstrap CI [-0.14, 0.74].
Resolution delta: -0.080.

## Context: best-N EIG vs arbitration

Turn wins/losses/ties: 10/7/33.
Mean paired censored-turn delta: -0.160; 95% bootstrap CI [-0.62, 0.3].
Resolution delta: 0.000.

## Mechanism

Override rate: 113/222 = 0.509.
Immediate resolutions after overrides: 7.
