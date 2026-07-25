# SWE-QA-Pro Non-Myopic Code-Search Opportunity Result

## Decision

The frozen zero-call opportunity gate failed. This exact SWE-QA-Pro
file-retrieval construction is closed before any LLM call and before parsing
any development or holdout row.

## Results

The audit reproduced all 120 opportunity rows and all 26 exact repository
commits. Ninety-two tasks exposed at least two answer-cited evidence files and
were usable; their mean evidence set had `3.3261` files. Root search was
diverse, with `10.6304` distinct top-one files on average, and immediate search
fully saturated only `14/92` tasks (`15.22%`).

Depth-two retrieval was broadly useful:

- `64/92` usable tasks recovered at least one additional evidence file;
- pair gains were one file on 43 tasks, two files on 19, and three files on 2;
- the mechanics, usability, saturation, pair-gain, and strict-gap-magnitude
  gates all passed.

The forced non-myopic tradeoff was too sparse:

- strict opportunities: `7`, below the frozen `12`;
- strict opportunities with at least three evidence files: `6`, below `8`;
- repositories represented: `6`, below `8`; and
- strict opportunity rate: `7.61%` of usable tasks.

Every strict event gained exactly one file over the greedy root plus that
root's own best continuation. Their mean normalized gap was `.3214`, above
the frozen `.15`, but prevalence failed the conjunction.

## Interpretation

SWE-QA-Pro does contain real path-dependent code search. The seven strict
examples span PennyLane, FitBenchmarking, Xarray, Sanic, SQLFluff, and yt-dlp,
and include cases where an initially irrelevant documentation, test, or
configuration result exposes a symbol that unlocks the implementation files.

That mechanism is not common enough under the frozen interface. Although a
second search helps on `64/92` tasks, on 57 of those 64 an immediately useful
root can use its own best observation-conditioned continuation to match the
best pair. The code-search tree is therefore sequential but mostly
order-commutative at horizon two. Paying an LLM to score it would mainly test
retrieval competence, not a forced non-myopic decision.

No root/query threshold, evidence parser, retrieval width, corpus filter, or
cohort was changed. The 40 development and 100 holdout rows remain outside all
question/answer computations. There is no threshold repair, selected seven-row
smoke, or paid follow-up.

## Budget

- API calls: `0`.
- OpenRouter spend: `$0`.
- Protected `$25` reserve affected: no.
- OatML use: none.

## Artifacts

- Preregistration:
  `results/nonmyopic/SWE_QA_PRO_NONMYOPIC_OPPORTUNITY_PREREGISTRATION.md`
- Audit:
  `results/nonmyopic/swe_qa_pro_nonmyopic_opportunity/AUDIT.json`
- Audit SHA-256:
  `947233181cc3070eaa591c9de6735defd38b8549f20ba22c72cc6498cc097284`
- Official code commit:
  `93ac6a4f3af3fe3f86580f62142f47e97e2cc897`
- Official dataset revision:
  `596892dac60b6f500f01a7dc2becb9f66593b7b7`

The paper remains unchanged: this is a prospective environment-screening
null, while the current tau-Knowledge result remains the strongest LLM-native
positive.
