# Bamboogle Semantic-BED Manifest Preregistration

## Purpose

Freeze a fresh multi-hop retrieval partition before reading any Bamboogle
question or answer. The intended experiment tests non-myopic planning over an
LLM's own generated answer hypotheses and observation-conditioned search
queries. This manifest only creates the value-blind boundary; it does not
authorize a policy result.

## Sources

- InfoReasoner repository:
  `https://github.com/dl-m9/InfoReasoner` at
  `ae79767d43de201963b0d012b00d935845bd3bdd`.
- InfoReasoner evaluates output-aware semantic information gain on Bamboogle
  as one of four multi-hop QA datasets.
- Dataset:
  `RUC-NLPIR/FlashRAG_datasets@bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`,
  `bamboogle/test.jsonl`.
- Source SHA-256:
  `c9703dae6bb1ceb9e2df77be45da28cb12aa040d2f471507890a461296968f3f`.
- Original Bamboogle release:
  `https://github.com/ofirpress/self-ask` at
  `559e56ec1ba8dc93cf63c4148ec1f4717caf4b60`.
- The source contains exactly 125 rows with keys `id`, `question`, and
  `golden_answers`. Only IDs and schema were read before this preregistration.

## Content-Blind Split

Sort the 125 string IDs, shuffle once with seed `24404`, and take contiguous
blocks:

| Split | Size | Ordered ID SHA-256 |
| --- | ---: | --- |
| Mechanics | 5 | `f2a9e1d7587e22d16127dae8ed15dd8c58fcc9b867ce1a1a1fc13f9e01912b7e` |
| Opportunity | 20 | `6abfc7dc790564fa09cc6fb282851beece36306c682e038e177e39b546aa2e94` |
| Development | 20 | `01837e7e950110c5da05440862d443092a1edc9ce14b0032c8942d5bbfedb8a6` |
| Holdout | 80 | `c2da4d412ae4d7c14c3aca06033f61ba5ec1078789ab70a39811eeeed3f85856` |

The public manifest may emit IDs, ordered hashes, and per-record hashes. It
must not emit question or answer text.

## Access Boundary

- After this preregistration and manifest implementation are committed, only
  the five mechanics questions and answers may be inspected.
- Opportunity values remain sealed until a complete cached-search transition,
  target-blind belief representation, controls, and prospective gates are
  frozen.
- Development values remain sealed through the mechanics and opportunity
  gates.
- Holdout values remain sealed through all method and serving development.
- Gold answers are evaluation endpoints. They may never appear in generation,
  search, belief-update, scoring, or selection prompts.

## Intended Mechanics Gate

The mechanics stage will answer three questions before scaling:

1. Does a no-search frontier-model baseline already saturate these questions?
2. Do retrieved observations materially change the model's sampled semantic
   answer support?
3. Does an observation-conditioned second query reach information unavailable
   to a width-matched nonadaptive or one-step policy?

The environment transition will use a fixed, cached search interface. Myopic,
non-myopic, width-matched, and random policies must share generated roots,
cached observations, sampling budgets, and stopping. Primary policy endpoints
will be exact normalized answer accuracy and gold-answer support recall;
semantic entropy is diagnostic and target-blind.

## Stop Rules

- Stop if the source hash, 125-row schema, split sizes, or ordered hashes fail.
- Stop if mechanics no-search accuracy is saturated or generated answer
  support does not change after evidence.
- Stop if the second search step is effectively order-commutative or provides
  no delayed evidence opportunity over a width-matched control.
- A mechanics pass authorizes only a separately committed opportunity
  protocol. An opportunity pass authorizes only low-cost development.
- No OpenRouter call or OatML job is authorized by this manifest.
