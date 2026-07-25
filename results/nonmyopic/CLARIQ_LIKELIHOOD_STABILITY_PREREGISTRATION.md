# ClariQ GPT-5.4 Likelihood-Stability Gate

## Purpose

Test whether a non-thinking GPT-5.4 produces stable, informative semantic
facet-to-answer likelihood maps for fixed human-authored ClariQ questions. This is
the serving gate for a route where ClariQ supplies action diversity and retrieval
endpoints while the LLM supplies the likelihood model.

## Frozen Source

- Official ClariQ repository at commit
  `46885a544581a0af8aff0681d29e4971807e2912`.
- `data/dev.tsv` SHA-256
  `68d2a5f87eab73721979b5f45f64099a9b2f080db1d0ce4b979d9daa4249906e`.
- Dev topic `133`, initial request `all men are created equal`.
- Five external facet descriptions, `F0134` through `F0138`.
- Four fixed human-authored questions:
  `Q00796`, `Q01384`, `Q03514`, and `Q03741`.

## Frozen Protocol

- GPT-5.4 through OpenRouter, explicitly non-thinking, temperature `0`.
- One classification returns exactly five `Y`/`N`/`U` characters in facet order.
- Ten independent requests in fixed order:
  questions 0-3 once, questions 0-3 again, then question 0 twice more.
- Thus every question has an exact repeat and question 0 has four identical
  requests in total.
- No question generation, repair, reissue, answer normalization, or consensus.
- No hidden facet, target, answer transition, retrieval endpoint, or test split is
  loaded.
- Seed `24390`; projected cost `$0.15`; hard run cap `$0.50`.
- OpenRouter only; OatML is not used.

## Frozen Gates

All must pass:

- exactly 10 physical requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all 10 five-character maps parse;
- every repeated question agrees exactly;
- all four copies of question 0 agree exactly;
- every base map is informative and has entropy at least `.30` nats;
- at least three unique base maps;
- maximum minus minimum base-map entropy is at least `.10` nats;
- total adapter cost is at most `$0.50`.

Failure closes this exact ClariQ likelihood interface. Passage authorizes a
separately frozen paired efficacy experiment on external ClariQ transitions and
NDCG endpoints; it does not itself establish policy efficacy.

The source-backed deterministic end-to-end fixture passes all gates with four
unique base maps and entropy range `.172609` nats.
