# LongVidSearch Four-Hop Tradeoff Confirmation V2 Result

## Decision

The prospective V2 confirmation **passes every frozen gate**. This establishes
a replicated four-hop semantic first-action opportunity and authorizes only a
separately preregistered 10-call nonreasoning OpenRouter serving smoke capped
at `$0.20`.

It is not yet an LLM policy result.

The public confirmation SHA-256 is
`895e448c047ca7afa924393da0a4f637a21735c8a770336fa17db0a72fdd4081`.

## Frozen-Gate Results

| Gate | Result | Pass |
|---|---:|:---:|
| Retrieval-complete tasks | 40/40 | yes |
| Answer-scorable tasks | 40/40 | yes |
| Diverse-root tasks | 40/40 | yes |
| Depth-four gain tasks | 39/40 | yes |
| Mean oracle four-clip coverage | 0.6563 | yes |
| Mean coverage gain | 0.4250 | yes |
| Eligible strict tradeoffs | 10/40 | yes |
| Eligible strict total gap | 10 clips | yes |
| Mean eligible strict sacrifice | 0.2039 | yes |

## Replication

The opportunity and confirmation blocks agree closely:

| Metric | Opportunity V1 | Confirmation V2 |
|---|---:|---:|
| Tasks | 40 | 40 |
| Strict tradeoffs | 10 | 10 |
| Strict total gap | 11 | 10 |
| Mean strict sacrifice | 0.2650 | 0.2039 |
| Depth-four gain | 39 | 39 |
| Mean oracle coverage | 0.6500 | 0.6563 |
| Mean coverage gain | 0.4125 | 0.4250 |

V2 does not pool these blocks to pass. The untouched confirmation independently
passes all thresholds.

## Category Replication

| Category | Tasks | Strict | Gap | Same root | Mean oracle coverage | Mean gain |
|---|---:|---:|---:|---:|---:|---:|
| Causal Inference | 16 | 4 | 4 | 12 | 0.6563 | 0.4219 |
| Global Summary | 10 | 4 | 4 | 6 | 0.7250 | 0.5000 |
| State Mutation | 14 | 2 | 2 | 12 | 0.6071 | 0.3750 |

The effect is not confined to one category, though it is most prevalent in
Global Summary.

## Scientific Scope

This confirms that, on one quarter of fresh four-hop tasks, the
answer-supported greedy first retrieval is not the root with highest attainable
four-search necessary-clip coverage. The oracle sacrifices about 20 percentage
points of immediate answer-token coverage on average to gain a necessary clip
later.

The result is classical and structural. BM25 and hidden evidence identify the
opportunity; no LLM has generated beliefs, likelihoods, queries, or policy
values yet. Any LLM-native claim requires a frozen policy with:

- semantic initial support or query generation;
- observation-conditioned support regeneration;
- likelihood/value scoring over complete four-step paths;
- a compute-matched myopic control;
- a random-strategy control; and
- exact necessary-clip endpoints hidden until policy freeze.

## Freshness And Budget

The 22 four-hop reserve videos, 20 three-hop development videos, and 22
caption-only fresh videos remain caption-unopened.

Calls: `0`. Cost: `$0`. OatML/Slurm: `0`.

