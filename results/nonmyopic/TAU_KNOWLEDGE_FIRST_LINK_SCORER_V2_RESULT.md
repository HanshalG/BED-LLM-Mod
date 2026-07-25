# tau-Knowledge First-Link Scorer V2 Result

## Decision

The overall preregistered confirmation gate failed because the sealed split had
less oracle-strength structural opportunity than required. However, every
frozen scorer-efficacy gate passed, and the task-level root-ranking improvement
is statistically supported. The defensible result is a qualified positive
first-link finding, not an end-to-end or all-gates success.

## Serving

The fresh two-task smoke completed exactly four calls with zero reasoning
tokens, retries, forced exits, or malformed responses and cost `$0.07014750`.
Both isolated scorer views produced varying canonical scores on both cases.
Descriptively, full-tree root-pair accuracy was `0.875` versus `0.625` myopic,
with one root-endpoint win and no losses.

Public smoke SHA-256:
`c5560d9f171c79515798d15109bd71612b6ec0a4e7a79f63b785b114505827e8`.

Private smoke SHA-256:
`175494b1cb768f0eacc509de800a8b1125dd598e6ae12df3209e0876a0c0fbdc`.

## Confirmation

The untouched 20-task confirmation completed exactly 160 calls with zero
reasoning tokens, retries, forced exits, or malformed responses and cost
`$1.69483500`.

The frozen structural-opportunity gates failed:

- oracle-strength non-myopic gap occurred on `2/20` tasks, below `4/20`; and
- mean gap was `0.10` documents, below `0.20`.

All frozen scorer-efficacy gates passed:

- 113 endpoint-distinct root pairs, above 50;
- non-myopic pairwise accuracy `0.7788`, above `0.60`;
- myopic pairwise accuracy `0.6858`, a gain of `0.0929`, above `0.05`;
- root selections differed on `9/20`, above four;
- root policy won on four tasks, lost on one, and tied on 15;
- total first-link endpoint advantage was `+5` documents, above two; and
- on the two structural-gap tasks, one non-myopic root was oracle-optimal.

The first-link policy retrieved 44 required documents under oracle
continuations, versus 39 for the isolated myopic policy. It selected an
oracle-optimal root on 15/20 tasks versus 12/20 myopic. The hidden-truth
oracle-strength greedy control reached 48 and the unrestricted oracle reached
50; these are descriptive, non-deployable controls.

Public confirmation SHA-256:
`dfbf597f8405b109d61d90206606962b5fbda439c8b81f086a9592c07fa247d1`.

Private confirmation SHA-256:
`9bc32d32d53355ff0a5c09e5163c7a11efc0f53994668872406b3db83f9946d9`.

## Uncertainty

A deterministic zero-call analysis used exact task-level randomization and
100,000 task bootstraps with seed `24336`.

- Root-ranking gain: exact paired label-swap one-sided `p=0.03979`;
  bootstrap 95% interval `[0.00435, 0.18692]`.
- Mean endpoint advantage: exact sign-flip one-sided `p=0.125`;
  bootstrap 95% interval `[-0.05, 0.60]` documents per task.

Thus the ranking result is supported at the task level, while the `+5` endpoint
gain is directional and underpowered.

Analysis artifact SHA-256:
`c4e9fceeae318b45a991484ac9dd4805aa9a9a16e3cbc8934bb1f4716a4c2ba3`.

## Mechanism

The result cleanly separates the two links. Full-tree semantic evidence improves
which first root is chosen, including recoveries on foreign-ATM policy, card
disputes, and multi-account card freezing. But the scorer's chosen followups
realized only 34 required documents, ten fewer than the 44 available beneath
its selected roots, with losses on 9/20 tasks.

This directly supports the diagnosis that continuation choice is a noisy proxy
for first-link quality. A next experiment should preserve the selected first
root and evaluate a separately frozen receding-horizon continuation policy,
rather than rerunning or retuning this confirmation.

## Budget

The project ledger is `$72.88624466` spent with `$32.49855803` headroom. The
live account has `$57.49855804` remaining, or `$32.49855804` above the
protected `$25` Monday reserve. OatML was not used.
