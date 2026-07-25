# ClariQ Topic-Level Train Opportunity Preregistration

## Status

Frozen before loading the ClariQ train synthetic conversation graph or
multi-turn retrieval endpoint. Only the `topic_id` column of `train.tsv` has
been used to construct the split.

This is a zero-call structural gate. Failure closes the corrected topic-level
ClariQ route before any new LLM likelihood estimation.

## Motivation

The first ClariQ audit treated each facet-specific empty-history context as a
decision unit. That overstates what a policy can condition on: all facets of a
topic share the same observable initial request. A valid BED policy must select
one first question under a belief over those alternative facets.

A corrected analysis on the already disclosed dev data uses a uniform prior
over facets and one root per topic. It finds:

- 35 usable topics;
- greedy and depth-two roots differ on 24;
- 21 have a positive depth-two terminal gain;
- 14 have gain at least `.005` NDCG@20;
- mean gain `.007261`; and
- maximum gain `.035343`.

The fresh train audit tests whether that opportunity transfers.

## Source

- Official repository: `https://github.com/aliannejadi/ClariQ`.
- Commit: `46885a544581a0af8aff0681d29e4971807e2912`.
- `train.tsv` SHA-256:
  `65d3da13b2d6ea77e7eaa45290894ffc162a5bd000e7640decd1b0a272a6e9d1`.
- `train_synthetic.pkl.tar.gz` SHA-256:
  `04c7312fddb79494696b6d3b965c1892d9ff6767456f7200c48b018ab785e90c`.
- multi-turn train evaluation parts:
  `fb9562b7ac1810b3ee74a04eea2936242929169a451ca7591e5c67af5d5982e2`
  and
  `903ad459772c716734c2f1fccd2610148d1d7414c7d5dcd9360a4bafe8e6a306`.
- Concatenated multi-turn archive SHA-256:
  `4611b0d2f551da4cfddf13fbaddac1d0cac5b388f1d9c806d9528836bd4cc509`.

## Frozen Split

Sort the 187 integer topic IDs, shuffle once with Python `random.Random(24399)`,
and assign:

- first 93: opportunity, ID-list SHA-256
  `52e8cfe007e7204800fde86eed429715c25f02274c87d63f1a23db91adeca52b`;
- next 31: development, ID-list SHA-256
  `9bc6d027d463ad5dc357be0bb299c22d25bb29376d20f67bff97875fab48612e`;
- remaining 63: holdout, ID-list SHA-256
  `b4e1809fcaa35708a46472da0793d12c0ea20a58a9c7abe63b4e1d7619f2e711`.

Only opportunity topic rows, contexts, and evaluation entries may be indexed or
reported by this audit.

## Topic-Level Decision Rule

For each opportunity topic:

1. alternative user worlds are its human-authored facets, with a uniform prior;
2. legal roots are human-authored questions available for every facet at empty
   history;
3. root immediate utility is mean official `NDCG20.with_answer` over facets;
4. each facet supplies its exact human answer and exact successor context;
5. facets producing an identical answer share one observation branch;
6. within each branch, one common second question maximizes mean official
   terminal utility;
7. root depth-two value is expected terminal utility over the prior;
8. greedy selects maximum immediate utility and depth two selects maximum
   terminal utility; and
9. exact ties choose the lexicographically smallest question ID.

The root question cannot be repeated at the second step. Topics need at least
two facets and two valid roots.

## Frozen Gates

All must pass:

- exact split sizes `93/31/63`, disjoint and exhaustive;
- at least 50 usable opportunity topics;
- greedy and depth-two roots differ on at least 30 topics;
- at least 25 topics have a strictly positive terminal gain;
- at least 15 topics have terminal gain at least `.005`;
- mean terminal gain is at least `.005`; and
- maximum terminal gain is at least `.02`.

Failure closes the route without topic selection, prior, utility, tie,
threshold, action, or transition repair.

## Conditional Next Stage

Passage authorizes only a separately preregistered development serving/ranking
gate. That method may replace the failed single-call deterministic likelihood
with a proper multisample semantic estimator, but must:

- use fresh development topics;
- keep facets, answers, successor utilities, and selected endpoint hidden until
  all likelihood samples and policy scores freeze;
- compare depth two with myopic, matched-width myopic, exact-human-likelihood,
  and seeded-random controls;
- use the same human question bank for every method;
- fail closed on serving errors; and
- cost at most `$0.50`.

No train holdout policy run is authorized by this document.

## Budget

The opportunity audit uses zero API calls, zero OpenRouter spend, and no OatML.
