# Number Game GPT-5.4 Mini External-Canonical Replay64

Date: 2026-07-29

## Question

Do the fixed cross-fitted depth-three policies from two independent
GPT-5.4-Mini planning studies outperform their equally cross-fitted
depth-two and myopic policies on a deterministic concept bank specified
outside this project?

The LLM remains load-bearing in every initial and answer-conditioned belief
support and in policy selection. The endpoint is the complete fixed bank of 33
concepts listed by Tenenbaum and Griffiths (2001), compiled to exact Boolean
extensions on integers 0--100. No endpoint LLM or semantic judge is used.

## Frozen Sources

Study A:

- result SHA-256:
  `1081da1e8381b88f7cd3fcd905b5ed539047863bcc087d686e509ed8f82d794a`
- trees SHA-256:
  `cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9`

Study B:

- result SHA-256:
  `25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8`
- trees SHA-256:
  `197dcfe3cb48eeb9a4b656d0f9b20ef2e0b45ac8d1df0ec4bd9358e07af18802`

Each source contains 32 trees with disjoint planning seeds. All roots are
fixed before this replay. Score all 33 concepts with equal concept weight,
equal tree weight, and one deterministic endpoint bank per tree. Naturally
extend each published 1--100 predicate to this project's 0--100 domain; omit
or reweight no concept.

## Primary Gates

All must pass:

- exactly 64 hash-bound trees and 33 unique canonical targets;
- depth-three and depth-two roots differ on at least 24 trees;
- both 32-tree studies directionally favor depth three;
- pooled depth-three Brier improves at least 1% over depth two;
- stratified tree-bootstrap 95% Brier-difference interval is below zero;
- depth three wins at least 24 trees versus depth two;
- pooled depth-three Brier improves at least 5% over myopic;
- its stratified interval versus myopic is below zero; and
- depth three wins at least 32 trees versus myopic.

The bootstrap resamples 32 trees independently within each source study using
the already implemented 20,000-draw stratified procedure.

## Diagnostics

Report without adding them to the primary conjunction:

- Hamming and exact-extension coverage versus depth two;
- fixed-support depth three, PTS, and uniform-random controls;
- depth-three versus depth-two rank fidelity;
- novel-target fields from the original source structure, noting that all
  canonical targets are external to policy selection.

This is a prospective combination of already-open fixed policies and an
already-open external bank, not a fresh-tree confirmation. Status language
must preserve that boundary even if every gate passes.

## Accounting

- model calls: `0`;
- OpenRouter cost: `$0`;
- OatML, Slurm, SSH, and cluster use: `0`.
