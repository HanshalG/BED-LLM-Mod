# LongVid Contrastive Path-Belief Mechanics Preregistration

## Status

Frozen before any response from this interface. This is a disclosed-
development first-link gate, not a reserve or headline result.

The previous entropy-ranking interface is closed and remains unchanged. It
used rows `955`, `1802`, `540`, `479`, `1332`, and `1068` and stopped on a
Unicode anchor before any path, score, or endpoint existed. This experiment
is materially different: it uses four other strict rows, provider-enforced
flat JSON, and contrastive belief-trajectory ranking rather than entropy.

## Bound Sources And Tasks

- Official LongVidSearch code commit:
  `4aa5620e06ca1bc5cc7cfce2b3e3ed0a5ae82d4e`.
- QA SHA-256:
  `370711f1299202cd40d559440859476bf3a2d2ca74540dad5226786558ee123b`.
- caption SHA-256:
  `0f2ce94265b7050eaee5de239a760c1a0f0762c1754cbb3bd7b45d00774b6c68`.
- four-hop opportunity artifact:
  `3cb882b1facbef2ababe2f8be68529423773c362a1eb4402b68d0bb240cc3ce0`.
- support-serving artifact:
  `0e594dbdaa748160e9006830992cca000ccfc7badec8cdde3b6d7ca56d412239`.

The last four unused strict opportunity rows and blinded root order are:

| Row | Candidate root indices |
|---:|---|
| 2156 | 1, 0 |
| 2062 | 2, 0 |
| 1689 | 11, 12 |
| 1648 | 0, 10 |

Layout seed is `270741`; layout SHA-256 is
`abf7091e39ac9116ac33e7e3b8fe88c41d81c2a602bc1e3493c2f1d0edf6242d`.
The model never receives greedy/oracle labels, answers, evidence-slice IDs,
coverage, or endpoint-derived values.

## LLM-Native State And Policy

Model: `openai/gpt-5.4` through OpenRouter, nonreasoning, temperature `0`,
zero retries.

For each task:

1. Generate one six-hypothesis semantic support over possible four-clip
   evidence chains.
2. Execute both fixed blinded root queries with deterministic BM25.
3. After each caption, regenerate all six hypotheses, integer weights,
   observation anchors, and next searches.
4. At steps two through four, execute the highest-weight next search while
   excluding retrieved captions.
5. Give a myopic contrastive scorer only the first regenerated support and
   first query from each candidate.
6. Give a final contrastive scorer the four regenerated supports and four
   queries from each candidate.

Both scorers make one nonreasoning structured call per task. They see only
the question plus the LLM's belief/query trajectory, not raw captions. Thus
the regenerated semantic state is the information bottleneck rather than
decorative logging.

The support response is a strict flat 24-field JSON object. Unicode is valid
inside anchor strings; grounding is checked after token normalization. This
is not a permissive reparse of the prior failed output.

## Delayed Endpoint

Necessary-clip IDs remain unloaded until:

- all `44` responses are checkpointed and parsed;
- all eight four-search paths are complete;
- all immediate and final choices are frozen; and
- a pre-endpoint checkpoint is written.

Then each path receives exact necessary-clip coverage. No call occurs after
endpoint loading.

## Frozen Gates

All are conjunctive:

- exactly `44` physical requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all strict structured outputs parse;
- all eight paths contain four distinct captions;
- at least `28/32` refresh supports change;
- all `32` refreshes contain at least four grounded anchors;
- at least three tasks have different realized candidate coverage;
- final pairwise accuracy is at least `.75`;
- final pairwise accuracy strictly exceeds immediate accuracy;
- immediate and final choices differ on at least two tasks;
- at least one choice change strictly improves coverage;
- final selected coverage exceeds immediate selected coverage by at least two
  necessary clips;
- final selected coverage strictly exceeds seeded-random coverage; and
- exact adapter cost is at most `$0.75`.

Failure closes this contrastive development interface. No task subset,
threshold change, reissue, repair, alternate model, or rerun is allowed.
Passage authorizes only a separately frozen experiment on the untouched
four-hop reserve videos.

## Budget

Projected spend is `$0.40`, with a hard `$0.75` cap. A live authenticated
OpenRouter balance check is required immediately before launch. The account
must retain at least `$25` through Monday, 3 August 2026. OpenRouter only;
OatML, Slurm, and cluster use are prohibited.
