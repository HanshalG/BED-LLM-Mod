# Bongard Matched Realized-Updater Implementation Audit

Date: 2026-08-08
Status: prospective, implemented, independently replayable, zero model calls

## Question

Does seeing the realized first answer when the VLM regenerates its intermediate
belief support lead to a better second query and lower endpoint predictive error?

The new `history_blind_update_matched_first` control holds fixed the dynamic
policy's root belief, first query, realized first label, endpoint, terminal
regeneration algorithm, terminal request seed, and terminal dispatch batch. Its
intermediate support is the paired same-seed support generated without the first
answer; that support is analytically updated by the realized label before it
selects query two. Dynamic planning instead selects query two from the support
regenerated with the answer in context.

This identifies the link from answer-conditioned intermediate support to the
next action and endpoint performance under a common terminal updater. It does
not identify a general advantage of the terminal updater itself.

## Prospective Gates

Development requires at least 24/64 changed final histories, at least 24 robust
changed second actions, changes in every block, at least 3% relative Brier gain,
paired bootstrap probability of improvement at least 0.80, and non-worse mean
log loss. Confirmation requires the analogous 36/96 path counts, every block,
at least 3% relative Brier gain, a paired-tree 95% bootstrap interval wholly
below zero, and non-worse mean log loss.

No gate was selected after seeing a Bongard mechanics, development, confirmation,
or endpoint response. Those response paths remain unopened.

## Frozen Chain

- Amendment SHA-256: `dfa981153687004c8fb2c1195879d0774a281ca6c231c55d85495f2ac622178b`
- Integrity amendment SHA-256: `1f0da098fbed4968a3594761194f661a0ebf49b12c477383d7c90bfa4989abf9`
- Development V17 SHA-256: `7564ced7755f17be13f254067f013130b4beb277313fc51de16b43611a608676`
- Confirmation V14 SHA-256: `0d9c6f52ea05aa93e40bf7aa61c8ebc6323f50a6454624d6f6c49ca946d3924a`
- Naive V8 SHA-256: `25db6fd3241d8ffaa4989ffaadfe6bb2bec111d7f1e3d6936dd9e39fd333d454`
- Mechanics implementation SHA-256: `453dfca0294265b1c997d871e41ec8f5e28fc1fbe378a2179381a65cbaae89d0`
- Development analyzer SHA-256: `03f06f89bc675897bf44ee82b69567f5217d7ab3d4bcd208996fe6fb623afc7a`
- Confirmation analyzer SHA-256: `d20aca14b8b254851caf785cac3dcf4a82a2e0b41c327506991e77ad676ad794`

The task identities, partitions, model seeds, root/branch requests, and scientific
endpoints are unchanged. The control adds at most one distinct terminal request
per task. Development is bounded by 704 accepted responses, 719 attempts, and
`$2.876` precharged exposure per 16-task block; its same-day naive block brings
the maximum to `$3.004`. Confirmation is bounded by 1,056 accepted responses,
1,078 attempts, and `$4.312` per 24-task block under the account-wide `$5` day.

## Verification

- All `tests/test_bongard_openworld*.py`: 166 passed.
- Focused matched-control and frozen-protocol suite: 82 passed.
- Python compilation passed for all modified runners and verifiers.
- Development V16, Confirmation V13, Naive V7, confirmation execution, paper
  bound files, and the banked-smoke replay certificate all independently passed.
- Authenticated Aug 10 preflight returned `ready_without_paid_calls`; all paid
  paths were absent, model calls and files written were zero.
- Live OpenRouter snapshot remained credits `$245.00`, usage `$220.121013787`,
  balance `$24.878986213`; the reported additional `$30` was not yet posted.

An adversarial follow-up found that confirmation replay reconstructed the
matched policy but its block-level validity conjunction did not explicitly name
the shared-first exactness gate. Before any response, the integrity amendment
added that gate and strengthened the shared verifier to reject altered
second-score support, argmax, or selection margin. Development and mechanics
already carried the shared-first gate. The repair changes no scientific design,
request, seed, action, endpoint, threshold, or budget ceiling.

No scientific result is implied by this audit. It only makes the prospective
test executable and fixes the interpretation before responses are observed.
