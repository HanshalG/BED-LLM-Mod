# Animals Depth Reanalysis

## Decision

**Track 1 collapses as a credible non-myopic paper spine.** The banked results still show that one-step EIG beats both naive baselines, but they do not show a defensible benefit from deeper planning. The full 40-trial depth-2 point estimate is slightly above depth 1; however, those 40 trials are not paired. On the only directly paired seed block, the sign reverses. Depth 3 is then materially worse than depth 2.

Paper implication: use animals as evidence that LLM-generated queries can support one-step BED, not as the non-myopic result. MediQ must carry the non-myopic claim.

## Metrics

- **Accuracy-AUC:** mean exact-guess accuracy over questions 1-20, normalized to [0, 1].
- **Q@80:** first question where mean exact-guess accuracy reaches 0.80. `>20` means it never reaches 0.80.
- Intervals are deterministic 95% percentile bootstrap CIs (20,000 replicates; seed 1304).

## Arm Results

| Arm | n | Accuracy-AUC | 95% CI | Q@80 | Q@80 95% CI | Accuracy at Q20 |
|---|---:|---:|---:|---:|---:|---:|
| Naive, non-thinking | 40 | 0.458 | [0.364, 0.552] | >20 | [16, >20] | 0.675 |
| Naive, thinking | 40 | 0.465 | [0.375, 0.556] | 20 | [16, >20] | 0.800 |
| EIG depth 1 | 40 | 0.742 | [0.659, 0.816] | 9 | [6, 14] | 0.950 |
| EIG depth 2 | 40 | 0.770 | [0.695, 0.835] | 8 | [5, 11] | 0.975 |
| EIG depth 3 | 40 | 0.644 | [0.556, 0.724] | 12 | [9, 15] | 0.900 |

## Comparisons

Positive Accuracy-AUC deltas favor the first method. Negative Q@80 deltas mean it reaches 80% in fewer questions.

| Comparison | Pairing | AUC delta | 95% CI | Q@80 delta | 95% CI | W/T/L |
|---|---|---:|---:|---:|---:|---:|
| depth 1 EIG - naive non-thinking (paired 40) | paired n=40 | +0.285 | [0.181, 0.390] | -12 | [-15.0, -6.0] | 30/2/8 |
| depth 1 EIG - naive thinking (paired 40) | paired n=40 | +0.277 | [0.191, 0.365] | -11 | [-14.0, -4.0] | 32/3/5 |
| depth 2 EIG - depth 1 EIG (unpaired 40) | unpaired 40 vs 40 | +0.028 | [-0.076, 0.133] | -1 | [-7.0, 3.0] | - |
| depth 2 EIG - depth 1 EIG (paired seed 12345, n=10) | paired n=10 | -0.040 | [-0.105, 0.025] | +5 | [-1.0, 5.0] | 3/1/6 |
| depth 3 EIG - depth 2 EIG (paired 40) | paired n=40 | -0.126 | [-0.224, -0.035] | +4 | [0.0, 8.0] | 15/1/24 |
| depth 3 EIG - depth 1 EIG (paired seed 12345, n=10) | paired n=10 | -0.225 | [-0.450, -0.030] | +15 | [-5.0, 16.0] | 1/0/9 |

## Why The Apparent Depth-2 Gain Does Not Hold

Depth 1 used one 40-trial stream at seed 12345. Depths 2 and 3 used four independently restarted 10-trial blocks at seeds 12345-12348. Only the first ten depth-1 trials are exactly paired with the seed-12345 blocks; the remaining 30 depth-1 trials do not share targets and randomized prior orders with the deeper runs.

The full, unpaired depth-2 estimate is 0.770 versus 0.742 for depth 1, a delta of +0.028. But on the only direct pair, depth 2 is 0.670 versus 0.710 for depth 1, a delta of -0.040 with W/T/L 3/1/6.

The depth-2 block AUCs are 12345: 0.670, 12346: 0.825, 12347: 0.765, 12348: 0.820. This large block spread explains how the unpaired aggregate can look favorable while the matched block reverses.

Depth 3 does not rescue the trend. Its paired 40-trial AUC delta versus depth 2 is -0.126 with W/T/L 15/1/24. Its block AUCs are 12345: 0.485, 12346: 0.755, 12347: 0.600, 12348: 0.735.

## Comparability Audit

The EIG arms match on the scientifically relevant configuration: Gemma 4 E4B non-thinking questioner, Gemma 4 31B non-thinking answerer, categorical belief, exponential-rank prior (rate 0.12), no belief generation or filtering, guess threshold 0.99, 40 Monte Carlo samples, generation temperatures 1.3/1.0, answer temperature 0.7, and the same 40-animal pool.

Operationally, depth 1 used `batched_block_size=1000` in one 40-trial run; depths 2 and 3 used `batched_block_size=2000` in four 10-trial runs. This does not itself change the intended policy, but the seed-block design prevents a paired full-sample depth-1 comparison.

Code provenance is weaker than configuration provenance. The logs do not record Git commits. Depth-1 and depth-2 logs are from May 4; depth-3 logs are from May 18-20, after commit `9c2261f` rewrote recursive forward search. Therefore depth 3 is configuration-matched but not demonstrably implementation-identical to the earlier runs.

## Bottom Line

1. One-step EIG is the stable positive result in this bank: it substantially outperforms both naive baselines on 40 exactly paired trials.
2. The small depth-2 advantage in the old aggregate is not credible causal evidence for planning depth because the direct paired subset reverses it.
3. Depth 3 is worse than depth 2 on all 40 paired trials in aggregate, with more trial-level losses than wins.
4. Do not rerun Paprika or build another animals variant. Proceed to the pre-registered MediQ non-myopic test, while retaining this animals result as one-step BED transfer evidence.

## Reproduction

```bash
python scripts/analyze_animals_depth_reanalysis.py
pytest tests/test_analyze_animals_depth_reanalysis.py -q
```

The JSON artifact records source SHA-256 hashes, recovered binary trial matrices, target/prior pairing metadata, curves, intervals, and comparison diagnostics. Raw logs remain untracked.
