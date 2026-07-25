# HoVer Semantic-BED Manifest Preregistration

## Purpose

Freeze a fresh many-hop fact-retrieval partition before reading any selected
development claim, label, or supporting fact. The intended experiment tests
non-myopic planning over an LLM's own generated semantic evidence hypotheses
and observation-conditioned retrieval queries. HoVer supplies an exact
external endpoint: retrieval of at least one supporting fact from every
required document. This manifest creates only a value-blind boundary and does
not authorize model calls or a policy result.

## Source

- Official repository: `https://github.com/hover-nlp/hover` at
  `39b84697f196308f398a251a7aea9b82ae0f0562`.
- Official development file:
  `data/hover/hover_dev_release_v1.1.json`.
- File SHA-256:
  `67c14858f2d7fcdb96b6fe3d538ffcd6f76e3ba594aa2c0cd4359f601101e89d`.
- The file has 4,000 rows with keys `claim`, `hpqa_id`, `label`,
  `num_hops`, `supporting_facts`, and `uid`.
- Exactly 1,835 rows have three hops and 1,039 have four hops. Only schema,
  UIDs, and hop counts were read before this preregistration. Two-hop rows are
  excluded prospectively because they offer less room for delayed retrieval
  value.

The HoVer authors designed later evidence documents to have reduced direct
semantic overlap with the claim, and the official metric requires at least one
supporting fact from every required document. Those properties make immediate
support coverage and incremental future support coverage separately
measurable.

## Content-Blind Split

For each of hops 3 and 4 independently, sort UIDs, shuffle once with
`random.Random(24405 + num_hops)`, and assign three mechanics rows, 200
opportunity rows, 20 development rows, and all remaining rows to holdout.
Concatenate the three-hop block before the four-hop block in each split.

| Split | Size | Ordered UID SHA-256 |
| --- | ---: | --- |
| Mechanics | 6 | `5ff4651736dd93dd91d793fa084ded2afd9f7b7a739020094a42be114aa17f4c` |
| Opportunity | 400 | `230ba25090172484250812ae0c24b089b2f3c888c7406702a47d2cd18a570fd0` |
| Development | 40 | `2eac3af917f9647919286ef37c1f691f0d95248d3e26e9cd353900ec3d0351f9` |
| Holdout | 2,428 | `050dc22e6700618f8929d301cfb7528f2b95bf77759077584d433064770cac6f` |

The manifest may emit UIDs, hop counts, ordered hashes, and per-record hashes.
It must not emit claim text, labels, or supporting facts.

## Access Boundary

- Commit this preregistration, manifest builder, and tests before generating
  the manifest or reading any selected row values.
- After the manifest reproduces, inspect only the six mechanics rows to freeze
  the exact retrieval transition and opportunity statistic.
- Run a separately preregistered zero-call audit on only the 400 opportunity
  rows.
- Development values remain sealed through the opportunity audit and all
  serving/mechanics work.
- Holdout values remain sealed through method development and the final
  experiment.
- Labels and supporting facts are evaluation endpoints. They may not enter
  hypothesis generation, query generation, belief updates, scoring, or
  selection prompts.

## Intended Opportunity Test

The mechanics inspection will determine a target-blind deterministic retrieval
interface over the official Wikipedia snapshot. The opportunity audit must
then quantify, without model calls:

1. how often a first document exposes a later required document that is poorly
   retrievable from the claim alone;
2. the exact immediate supporting-document gain of each root action;
3. the exact incremental future gain of its best legal continuation; and
4. whether the root maximizing immediate gain differs often enough from the
   root maximizing total two-step gain.

The audit must freeze prevalence, dynamic-range, and greedy-gap thresholds
before opening opportunity values. A pass may authorize only a small,
separately preregistered LLM-native support-regeneration smoke with paired
myopic, non-myopic, compute-matched, and random controls.

## Stop Rules

- Stop if the source hash, row count, schema, hop counts, split sizes, or
  ordered hashes fail.
- Stop if mechanics cannot define an exact target-blind transition using the
  official corpus.
- Stop if the opportunity audit finds insufficient root-choice variation or
  delayed supporting-document gain.
- Do not relax thresholds, select a favorable subset, or inspect development
  or holdout after a failed gate.
- No OpenRouter call or OatML job is authorized by this manifest.
