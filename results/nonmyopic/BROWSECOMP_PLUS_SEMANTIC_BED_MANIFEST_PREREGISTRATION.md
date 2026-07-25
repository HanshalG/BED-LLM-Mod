# BrowseComp-Plus Semantic-BED Manifest Preregistration

## Motivation

BrowseComp-Plus is a fresh, fixed-corpus deep-research benchmark derived from
BrowseComp. Its official release separates documents needed as evidence from
documents that semantically contain the final answer and includes hard
negative documents. This creates a more credible LLM-native sequential design
surface than public-Wikipedia tasks: semantic queries reveal bounded text
observations, multiple pieces of evidence are usually required, and exact
document endpoints remain available.

The source is the Apache-2.0 official repository at commit
`046949032b0328319cc9a02663a759ec601d9402`. The six encrypted Parquet shards,
evidence qrels, and gold qrels are hash-bound in the manifest script.

## Content-Blind Split

Before decrypting any question, answer, or document text, use only task IDs
and the number of official evidence qrels. Gold document IDs and evidence
document IDs are not emitted.

- Seed: `24407`.
- Evidence bins: low `1..4`, mid `5..7`, high `8+`.
- Mechanics: `5` tasks, bin counts `1/2/2`.
- Opportunity: `120` tasks, bin counts `33/51/36`.
- Development: `40` tasks, bin counts `11/17/12`.
- Holdout: `665` tasks, bin counts `186/283/196`.

The manifest must reproduce all `830` unique task IDs, the official file
hashes, gold-as-subset-of-evidence invariant, exact split sizes, and frozen
ordered hashes. It emits no question, answer, document, URL, or document ID.

## Access Rule

After the manifest is generated and committed, only the five mechanics rows
may be decrypted to design a target-blind zero-cost opportunity audit. The
opportunity 120 may be opened only after that audit is frozen in code and
prose. Development and holdout remain sealed.

No OpenRouter or OatML call is authorized by this manifest. Any paid mechanics
requires a separate preregistration, a passing zero-cost opportunity gate, a
sub-`$1` cap, and the existing `$25` protected OpenRouter floor through Monday
2026-07-27.
