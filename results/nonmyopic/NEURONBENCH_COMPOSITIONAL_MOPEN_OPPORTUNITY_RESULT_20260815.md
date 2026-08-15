# NeuronBench Compositional M-Open Opportunity Result

Date: 2026-08-15 (Europe/London)

Status: **failed closed**

## Provenance

- Frozen protocol commit: `8f8a9add`
- Pushed implementation commit: `a859068b`
- Official NeuronBench commit: `c354622458c460b419cab821d482c879f0578377`
- Protocol SHA-256: `e693587d96f5e557089d3438430cc0ee3a6406d59b69dd44187be1a589525ba5`
- Result SHA-256: `7af6760f47a93556e6dd8b04250f3c3cf2e9e7a50898ca10186daf129eae7980`
- Response-bank SHA-256: `34ba78da58dc8016b855e9b6b3ead4ef79bce0c3732dafb37802a18472a560f4`
- Canonical response payload SHA-256: `3bdab504c9290cd3551985872843a5cab59c6d17432dd73c68cdfd405e9b5c4c`
- Model/API calls: `0`
- Cost: `$0`

The exact source adapter built 22 executable models, all 15 two-current truths,
the released nine-action design pool, and 28 common held-out query protocols.
All source, schema, finiteness, normalization, and planned-versus-explicit-replay
checks passed.

## Primary Result

| Policy | Root protocol index | Expected held-out MSE | Median truth MSE | Maximum truth MSE |
|---|---:|---:|---:|---:|
| dynamic d1 | 6 | 1.137226000 | 0.012182593 | 15.360128032 |
| dynamic d2 | 7 | 0.334016652 | 0.012182593 | 3.161487181 |
| dynamic d3 | 4 | 0.156647377 | 0.012582600 | 0.944638941 |
| full-support d1 | 6 | 0.117149147 | 0.012358032 | 0.639684319 |
| plain fixed-support d1 | 0 | 69.797619048 | - | - |
| history-blind d3 | 0 | 40.804507990 | - | - |
| random-action dynamic | - | 2.570703576 | - | - |

The roots are respectively paired-long-pulse, depolarising-conditioning, and
hyperpolarising-conditioning protocols. Both successive mean reductions are
large: 70.63% for d2 versus d1 and 53.10% for d3 versus d2. Root actions and
reachable policy decisions also differ, and every planned prior risk exactly
matches independent uniform truth replay.

## Why The Gate Failed

| Frozen condition | Result |
|---|---|
| d2 mean reduction at least 5% | pass |
| d3 mean reduction at least 5% | pass |
| d2 paired wins exceed losses | fail: 2 wins / 9 ties / 4 losses |
| d3 paired wins exceed losses | fail: 5 wins / 4 ties / 6 losses |
| d3 beats d1 on at least 12/15 truths | fail: 3 wins / 6 ties / 6 losses |
| d3 within 10% of full-support d1 | fail: 33.71% higher MSE |
| d2 and d3 behaviorally distinct | pass |
| planned risk equals truth replay | pass |

The mean ladder is driven by tail repair, not a broad paired improvement. d1's
15.36 error on `na_fatigue+ca_rebound` falls to 0.017 at d2. d2 then incurs a
3.16 error on `h_sag+d_type`; d3 reduces that to 0.945 while worsening several
smaller rebound/D-current cells. Median loss is effectively flat across depth.

This is not a global identifiability failure. The full nine-action signatures
identify all 15 pair truths, and the 28-query signatures are also unique. The
dynamic support contains the true pair for 14/15 truths at every depth. The
miss changes from `na_fatigue+ca_rebound` at d1 to `ca_rebound+d_type` at d2 and
d3. Thus the remaining failure is path-dependent support/risk allocation, not
an absent executable truth family.

## Architectural Reading

The compositional extension successfully creates a real horizon-dependent
decision problem: deeper policies select different roots and sharply reduce
the expected tail. It does not establish the frozen paper claim because raw
count MSE lets one catastrophic cell dominate the expectation while most
truths are tied or mildly worse.

For a new prospective environment, use a scale-balanced proper predictive
score (standardized count NLL or per-query normalized MSE), report mean and
tail risk separately, and require multi-bank conservative action adoption so a
depth increase cannot trade many modest regressions for one rescued outlier.
Keep the useful architecture: a small executable mechanism grammar, adaptive
support refresh, exact numerical likelihoods, a local particle scenario tree,
and an LLM restricted to typed model edits. Exact enumeration should remain an
offline verifier; this run required roughly half an hour and exposed why the
runtime planner needs bounded local trees and cacheable proposal banks.

Per the frozen protocol, this exact deterministic formulation is closed. No
LLM mechanism-edit gate, stochastic endpoint, efficacy claim, or paper headline
is authorized from this result.
