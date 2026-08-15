# Number Game DeepSeek Stratified Semantic V5 Router Terminal Result

Date: 2026-08-15 (Europe/London)

Status: **transport and semantic null; exact V5 route closed**

## Provenance

- Pushed implementation commit: `fd8b8f27`
- Execution binding SHA-256:
  `37f7f72ae48c1b4108fc98f248da27770127ee8a3883a86367fc3c5af7dd2a59`
- Frozen model: `deepseek/deepseek-v4-flash-0731`, nonreasoning
- Frozen seeds opened: `202608210000..202608210063`
- HTTP attempts / accepted responses: `64 / 64`
- Retries, reasoning tokens, forced exits: `0 / 0 / 0`
- Actual local and posted cost: `$0.002245333`
- Authenticated closing credits / usage / balance:
  `$245.000000000 / $220.348811737 / $24.651188263`

The router transport repair reached the exact model. All 64 transport records
reported the requested and returned model as the frozen DeepSeek revision.
However, only 57 records had a clean `stop`; seven had finish reason `error`.
The independent verifier therefore correctly failed its clean-transport replay
gate. It passed request identities, raw hash, semantic diagnostics replay, and
outcome privacy.

## Semantic Result

The first, empty-history group stopped the run before the two conditioned
history groups.

| Metric | Frozen floor | Observed |
| --- | ---: | ---: |
| valid particles | 48 | 18 |
| unique signatures | 28 | 13 |
| unique extensions | 24 | 18 |
| even-half valid/signatures/extensions | 24/24/20 | 8/8/8 |
| odd-half valid/signatures/extensions | 24/24/20 | 10/10/10 |
| maximum extension multiplicity | at most 4 | 1 |

Rejections were 35 signature mismatches, seven malformed JSON responses, two
shape failures, and two invalid expressions. The low multiplicity shows that
accepted outputs were diverse, but the model did not obey the requested
five-anchor semantic partitions reliably enough to construct the required
particle bank.

## Authority And Interpretation

This is not evidence about non-myopic policy efficacy. The run never opened a
conditioned proposal group, complete semantic bank, Number Game canonical
target, grammar endpoint, policy tree, development cohort, or confirmation
cohort. It authorizes nothing and must not be retried with changed providers,
seeds, prompts, or thresholds.

Together with the Qwen V3 null, this closes arbitrary five-bit signature
obedience as the primary route. The next architecture should follow the MDA
lesson: ask an LLM for residual-conditioned executable mechanisms in a natural
domain vocabulary, then let numerical likelihoods, SMC, and terminal-risk
planning own the belief and decision calculations. The strongest existing
substrate is the already verified monotonic ChemBench dynamic-support ladder.

## Banked Artifacts

- Failure SHA-256:
  `4d90234fd179c78e7b8f6ed7f9de10eb46fd05627850db179f8b18b07e4d7bc2`
- Reconciled ledger SHA-256:
  `46772d88b5037a622492889c91b05bf926a0c69313f2d8c3f4f2ff7c1d68deee`
- Label-free result SHA-256:
  `851c986a472e4d86e244c01f2012d42c17c1d60afcb1dbc749aad9127486fdd2`
- Verification SHA-256:
  `dd1388a3ab899227dd8f744dca0f7453e9c585d69295e7b5b638b1489bc1219a`
- Private raw-response SHA-256:
  `b6d830a259f174dd8ea20ed73f1eb923692f9ecfa66bab7d23e6025c516ee9dc`
