# InfoQuest Target-Alignment Audit Preregistration

Frozen after closing the cross-family likelihood route and before computing any
all-candidate target-gain aggregate.

## Status and Inputs

This is a deterministic, zero-LLM, post hoc development audit. It binds:

- V3 public/private SHA-256
  `0cfbe3d1590001af508d351d131d12d747f2fcadf0448e2e7868809f40d968f7`
  and
  `134a9659dc5f86df3dba6e18fdc8c72a79f3524b166d2cd52d91a7fe88747e1e`;
- cached-answer diagnostic public/private SHA-256
  `9855e57d3a00afcd074e15812aa14bb987ede1bc8b09d97a9382ae8b901451f2`
  and
  `e5a128d92fa48693e3b6f28b216d3b8c314bb3c67081ad1ba6cdb485e932dc9a`;
- the existing common-history and fixture bindings already enforced by those
  loaders.

No model, provider, cluster, or semantic text output is used.

## Target-Gain Proxy

Each fixture's cached-answer checklist judgment contains five immediate
information bits for every root. For a current root and candidate root, target
gain is the number of candidate immediate bits not already present in the
current root. This gives all four candidate actions in all 30 root/world cells.

The proxy is validated against the already-judged dynamic and fixed selected
paths: the bitwise union of current and selected-root immediate bits is compared
with the directly judged two-turn bits. All 300 selected-path bits are checked.
Agreement below `.90` fails the audit.

For both V3 dynamic and fixed supports, the audit reports:

- mean within-cell Spearman correlation between all four predicted EIG scores
  and target gains, excluding cells with constant target gain;
- top-choice target gain and oracle regret;
- number of target-optimal choices;
- dynamic-versus-fixed target-gain wins/ties/losses.

## Frozen Gates

All must pass:

1. exact 6 fixtures, 30 cells, and 120 candidate actions;
2. selected-path additive bit agreement at least `.90`;
3. at least 15/30 cells have nonzero candidate target-gain spread;
4. oracle target gain exceeds fixed selected gain by at least `.15` on average;
5. mean dynamic-support EIG/target-gain Spearman at least `.20`;
6. dynamic-support mean oracle regret no worse than fixed support;
7. zero LLM calls and `$0` cost.

Opportunity gates 2--4 establish that the candidate bank and proxy are usable.
Gates 5--6 establish target alignment. If opportunity passes but alignment
fails, further InfoQuest work must change the target representation rather than
partition prompting or planning depth. If opportunity fails, this exact
InfoQuest action bank closes.

The reserved pre-Monday operational allowance remains `$2.08203240`. OatML
jobs: `0`.
