# BrowseComp-Plus Semantic Mechanics v1 Result

## Decision

Version 1 fails closed before future scoring or endpoint analysis. The exact
200-character query interface is closed and will not be repaired or reused.

## Exact Failure

- Run: `browsecomp-plus-semantic-mechanics-20260725T194339Z`.
- Initial responses parsed: `5 / 5`.
- Root-refresh responses parsed: `27 / 30`.
- Three otherwise valid `A01` lines had query lengths
  `210`, `239`, and `319`, above the frozen `200`-character ceiling.
- Physical requests / HTTP attempts: `35 / 35`.
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`.
- Future scorer calls: `0`.
- Terminal belief calls: `0`.
- Scientific endpoints: not evaluated.
- Cost: `$0.1810625`.
- Public failure SHA-256:
  `d08ba9737d2051809d88c6bd827b9821d0bf931fd1fb7e31a7727402a90927cf`.
- Private raw SHA-256:
  `76544d1478d920900c60c4e998af5637bccbb478d00a8de33150ccce04f2cd90`.
- OatML use: none.

## Source-Derived Interface Check

The already-open opportunity trajectories contain `2,884` official GPT-5
BM25 search actions. Their query lengths have median `70`, p99 `180`, and
maximum `382` characters; `17` exceed 200 and none exceed 400. This check
uses no new task block or endpoint.

That native action distribution prospectively justifies one final v2 with a
400-character query ceiling. It is a new interface run with fresh responses,
not a repair or continuation of v1. All scientific mechanics and thresholds
remain unchanged. Any v2 failure closes the route.
