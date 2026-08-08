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
- Development V16 SHA-256: `7ed91de5698e2d0e9a5a2dbbeb83c70362c4567719b7a02f2266912eacbdd5b7`
- Confirmation V13 SHA-256: `2debe466a051b11581dfc5a9f7840a549be506862f692407ce4b5111fa9348f7`
- Naive V7 SHA-256: `3b4be9ed0bb2f5eea35573a8d25438672677ce46dd58db14a55de9324efe40be`
- Mechanics implementation SHA-256: `40d40f6850910467e6299a26c0ca98e4c81eb5e8091c642a8f59f1cb1efc01c9`
- Development analyzer SHA-256: `595ae3590ac67c07ad7f6551bf36720d8f903d681ed7b0a07707b311bb2ee4b1`
- Confirmation analyzer SHA-256: `418df164b36cdeb6c199e87c566603ad66e34f7648480e4f99110cc5ab0d927e`

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

No scientific result is implied by this audit. It only makes the prospective
test executable and fixes the interpretation before responses are observed.
