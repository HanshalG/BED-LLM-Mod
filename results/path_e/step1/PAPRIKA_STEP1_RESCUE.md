# Paprika Step 1 Generation-Thinking Rescue

Status: **rescue_insufficient_stop_and_discuss**

| arm | resolution@5 | mean censored turns | coverage | cost (USD) | requests |
|---|---:|---:|---:|---:|---:|
| naive_nonthinking | 0.400 | 4.600 | 0.875 | 0.0050 | 124 |
| naive_thinking | 0.400 | 4.600 | 0.850 | 0.1015 | 137 |
| rescue_eig | 0.300 | 4.800 | 0.902 | 0.4999 | 4221 |

## Matched rescue vs naive non-thinking

Wins/losses/ties: 1/1/8.
Mean paired censored-turn delta: 0.200; 95% bootstrap CI [-0.6, 1.2]; Wilcoxon p=None.

## Adversarial rescue vs naive thinking

Wins/losses/ties: 2/2/6.
Mean paired censored-turn delta: 0.200; 95% bootstrap CI [-0.9, 1.5]; Wilcoxon p=None.
