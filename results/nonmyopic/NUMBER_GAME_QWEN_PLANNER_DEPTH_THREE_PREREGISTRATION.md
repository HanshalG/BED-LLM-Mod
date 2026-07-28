# Number Game Qwen Planner Depth-Three Preregistration

Date frozen: 2026-07-29, before any Qwen history-conditioned planning
response.

## Claim

Can a third model family generate the open-ended, path-dependent planning
beliefs and still yield a monotonic depth-three improvement?

`qwen/qwen3.7-plus` replaces GPT-5.4 Mini as the generator of the initial
support and both history-conditioned belief refreshes. Gemini 2.5 Flash
independently generates eight validation supports and sixteen endpoint
supports per tree. All models use reasoning disabled.

This is stronger than the completed Qwen cross-judge result: Qwen is now
load-bearing in policy construction rather than only target generation.

## Fixed Method

The method is unchanged from the fresh GPT-planner replication:

- domain 0--100 and the same executable safe-rule grammar;
- eight candidate roots from the initial generated support;
- retain answer-consistent parent hypotheses at both refreshes;
- generate fresh hypotheses after every first- and second-query history;
- score depth-three and depth-two roots on eight independent validation
  supports with equal draw weight;
- execute each frozen three-query policy against sixteen independent Gemini
  endpoint supports;
- compare posterior-predictive Brier, best-rule Hamming, exact-extension
  coverage, novel targets, rank fidelity, and the existing controls.

No endpoint can alter a support, candidate, score, or root.

## Full-History Mechanics Gate

The already completed ten-call Qwen prior-support gate establishes generic
structured serving only. This new gate exercises one complete planning tree:

- tree seed 33000;
- Gemini target seed 33100;
- eight validation seeds 33200--33207;
- fifteen additional endpoint seeds 33300--33314;
- exactly 73 accepted calls;
- temperature 0.7, strict schema, unchanged parser and prompts;
- first- and second-refresh retention enabled;
- hard cost cap `$0.20`.

The gate passes only if every condition holds:

- exact request/attempt accounting, at most eight retries, zero reasoning
  tokens, zero forced exits, and cost within cap;
- initial support has at least 16 valid rules;
- all eight validation and all sixteen endpoint supports have at least 16
  valid rules;
- the tree has at least 128 novel endpoint hypotheses in total; and
- every retained first branch has at least eight rules and every retained
  second branch has at least four.

The tree is scored for end-to-end mechanical validation, but its efficacy is
descriptive and is explicitly forbidden from authorizing, changing, or
canceling the formal design. Only the mechanics conjunction authorizes it.
Failure closes this exact Qwen planning route without prompt, parser, seed,
or threshold repair.

## Conditional Powered Study

Only after the mechanics gate passes, run 32 wholly fresh trees:

- Qwen planning seeds 34000--34031;
- Gemini target seeds 34100--34131;
- eight Gemini validation draws per tree, seeds 34200--34455;
- fifteen extra Gemini endpoint draws per tree, seeds 34500--34979;
- exactly 2,336 accepted responses;
- equal draw weight within tree and equal tree weight;
- 50,000 whole-tree bootstrap replicates;
- hard cost cap `$4.50`, with at least `$3.80` live balance before launch.

The formal result passes only if every mechanics gate and every scientific
gate holds:

- depth-three and depth-two roots differ on at least 12/32 trees;
- depth three improves Brier by at least 1%, its paired whole-tree 95%
  interval is strictly below zero, and it wins at least 12 trees;
- Hamming and coverage do not regress;
- novel-target Brier, Hamming, and coverage do not regress;
- depth three beats myopic EIG and fixed-support depth three by at least 5%
  Brier, and PTS by at least 3%, with every interval below zero; and
- depth-three source-to-endpoint Brier Spearman correlation is at least 0.7
  and exceeds depth two by at least 0.15.

This design deliberately omits the prior auxiliary requirement that
cross-fitting beat the in-sample depth-three selector by 2%. That threshold
does not test monotonic depth or planner-family robustness and caused the
otherwise successful fresh GPT replication to be labeled conjunctively null.
The in-sample comparison remains reported descriptively.

There is one mechanics gate and one conditional formal run, with no seed
replacement, response deletion, policy repair, source pooling, or threshold
change after outcomes.
