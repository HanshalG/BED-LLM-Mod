# HoVer Path-Dependent Retrieval Opportunity Result

## Outcome

**Pass, exactly at the robust boundary.** The frozen 400-row audit passes every
integrity, prevalence, dynamic-range, and first-action gate without model calls.
The broad root-planning opportunity is substantial; the strongest
immediate-sacrifice opportunity is real but rare.

Analysis SHA-256:
`e2e53c8b616c8178af0473c15945318019e87f909ec7881bb313fafad2463586`.

## Results

| Metric | Result | Frozen gate |
| --- | ---: | ---: |
| Resolved top-100 candidate sets | 400 / 400 | at least 390 |
| Nonempty title-transition graphs | 398 / 400 | at least 300 |
| Reachable and root-dynamic rows | 360 / 400 | at least 250 |
| Rank-sensitive root opportunities | 80 / 400 | at least 40 |
| Three-hop rank-sensitive | 45 / 200 | at least 15 |
| Four-hop rank-sensitive | 35 / 200 | at least 15 |
| Robust immediate-sacrifice opportunities | 8 / 400 | at least 8 |
| Best depth 3 strictly above depth 2 | 53 / 400 | at least 20 |
| Mean rank-sensitive root gain | +0.220 documents | at least +0.10 |
| Mean robust root gain | +0.020 documents | at least +0.02 |

All input hashes and the opportunity split hash reproduce. The run made zero
model calls, spent `$0`, and used no OatML resource.

## Interpretation

The useful comparison is stronger than greedy-versus-oracle continuation. The
myopic root receives its own best possible depth-3 tail. In the robust version,
*every* root tied for best immediate supporting-document coverage receives an
oracle tail, and the best of them is the baseline. Eight tasks still require a
different root that gives up one immediately supporting document and retrieves
two supporting documents later.

The broader signal is less fragile: 80 tasks change the official-rank myopic
root and gain exact support coverage, with mean gain `+0.22`. Depth itself is
load-bearing on 53 tasks because no two-document path matches the best
three-document path.

This establishes an exact non-myopic first-link substrate. It does **not**
establish that an LLM can recognize the valuable roots, generate useful
path-dependent beliefs, or beat a compute-matched myopic scorer.

## Boundary Audit

Because the robust count and mean land exactly on their frozen thresholds, all
eight robust paths were inspected after the aggregate result:

- every edge is an exact, semantically valid title mention in the source
  article;
- no robust path is caused by a short-title or parenthetical-alias collision;
- the eight rows correspond to six unique underlying Hotpot IDs and six unique
  title-path signatures;
- two path signatures each occur in both a supported and a not-supported claim
  variant.

The result is therefore not a parser artifact, but the effective independent
mechanism count is six rather than eight. Any later paper claim must call this
a narrow opportunity population and report the duplicated source structures.

## Decision

The full frozen gate authorizes a separately preregistered, low-cost
LLM-native mechanics smoke. It does not authorize a scale experiment.

The smoke should:

- make the LLM generate initial semantic evidence-chain hypotheses;
- regenerate those hypotheses separately after each candidate root document;
- score immediate evidence value and future support/readiness separately;
- compare full immediate-plus-future scoring with myopic, compute-matched,
  shuffled-future, and random controls;
- use exact `V3(root)` as the primary first-link endpoint, keeping continuation
  execution secondary; and
- use nonreasoning for the proposed method, reserving thinking for the naive
  baseline.

Development and holdout values remain sealed until that protocol is committed.
