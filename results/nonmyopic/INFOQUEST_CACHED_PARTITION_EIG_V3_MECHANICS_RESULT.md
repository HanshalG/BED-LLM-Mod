# InfoQuest Cached-Partition EIG V3 Mechanics Result

The frozen V3 disclosed-world mechanics run completed cleanly, but failed its
scientific gates. Exact EIG over GPT-5.4's regenerated semantic support and
answer partitions did not select more useful follow-up questions.

## Protocol

- run ID: `infoquest-cached-partition-v3-mechanics-20260726T033658Z`;
- exact 126 physical requests and 126 HTTP attempts;
- all 60 compact partition, 60 simulator, and 6 checklist responses parsed;
- zero retries, reasoning tokens, and forced exits;
- cost `$0.4447171`, below the `$0.85` cap;
- no response was repaired, reparsed, reissued, imputed, or dropped.

## Frozen Endpoints

- dynamic supports changed in `30/30` cells with mean novel fraction `1.0`;
- at least two actions had positive predicted EIG in `24/30` dynamic and
  `25/30` fixed cells;
- dynamic and fixed exact-EIG choices differed in `12/30` cells;
- every fixture used at least two dynamic action labels;
- mean realized incremental checklist gain was `.20` dynamic versus `.40`
  fixed;
- mean dynamic-minus-fixed gain was `-.20`;
- paired outcomes were `1/22/7` wins/ties/losses;
- `0/6` fixtures had positive mean dynamic-minus-fixed gain.

The transport, support-change, candidate-informativeness, action-difference,
and diversity gates passed. The dynamic-gain, paired-win, positive-fixture,
and dynamic-over-fixed efficacy gates failed.

## Post Hoc First-Link Diagnosis

Using only the public cell metrics:

- selected dynamic EIG versus realized dynamic gain:
  Spearman `rho=.125`, `p=.510`;
- selected fixed EIG versus realized fixed gain:
  `rho=.169`, `p=.372`;
- predicted dynamic-minus-fixed score difference versus realized gain
  difference: `rho=-.015`, `p=.936`;
- among the 12 cells where the selected action changed, outcomes were
  `1/7/4` and mean dynamic-minus-fixed gain was `-.25`.

These correlations are exploratory, not preregistered endpoints. They locate
the failure at the ranking link: the LLM-generated semantic likelihood
partitions provide substantial predicted entropy variation but do not rank
questions by realized checklist information.

The public `MECHANICS.json` SHA-256 is
`0cfbe3d1590001af508d351d131d12d747f2fcadf0448e2e7868809f40d968f7`.
The private raw-response SHA-256 is
`134a9659dc5f86df3dba6e18fdc8c72a79f3524b166d2cd52d91a7fe88747e1e`.

This is clean adverse development evidence. V3 is closed without a fresh
confirmation, policy run, depth sweep, or rerun.
