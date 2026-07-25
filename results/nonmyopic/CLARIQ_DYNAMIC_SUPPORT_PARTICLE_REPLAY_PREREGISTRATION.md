# ClariQ Dynamic-Support Joint-Particle Replay Preregistration

## Status and Purpose

This is a zero-call post hoc diagnostic on the disclosed failed V2 mechanics
tree. It cannot rescue V2 or support an efficacy claim. No score or endpoint from
the failed tree has been inspected before freezing this analysis.

V2 failed because two particles shared the same normalized intent text while
predicting different future response profiles. The diagnostic defines one
support particle prospectively as:

```text
(normalized natural-language intent, full predicted response-code profile)
```

Identical intent text with a different response profile is therefore allowed as
a distinct uncertainty particle. An exact duplicate pair remains invalid.

Raw failed-tree SHA:
`752718c675f3e2c9d7f89dcb94b2a26286e563c9e68a1ea0c8237d8ff78bab9e`.

## Frozen Replay

- Strictly reparse all one initial plus 90 branch responses.
- Keep every mass, code, and line unchanged.
- Compute myopic, fixed-support depth two, dynamic-support depth two, and
  answer-link-shuffled dynamic scores.
- Freeze all scores and roots before loading the already-open topic-`60`
  oracle-tail NDCG@20 endpoints.
- Load no development or holdout content/endpoint.

## Authorization Gates

A fresh, separately preregistered joint-particle mechanics run is worth buying
only if all of the following pass:

- all 91 supports parse under joint identity;
- zero exact joint-particle collisions and at most five text-only collisions;
- at least 80 changed branch supports;
- at least 75 profile-diverse and 75 positive-continuation branches;
- myopic and dynamic-future score ranges at least `.05` nats;
- a dynamic/fixed score difference at least `.02` nats;
- dynamic changes the root from myopic;
- dynamic all-root terminal-NDCG Spearman at least `.20` and at least `.02`
  above both myopic and fixed support;
- dynamic selected terminal NDCG strictly beats both myopic and fixed support;
  and
- dynamic is nonworse than answer-link-shuffled dynamic.

Failure closes the ClariQ dynamic-support route with no more parser, model,
topic, or threshold attempts. Passage authorizes only a fresh disclosed
mechanics task under the joint-particle definition, not development `148`.

## Budget

This replay uses zero model calls and `$0`. OpenRouter remains the sole
authorized future compute path; OatML/cluster use is prohibited. The stricter
project headroom is `$12.335724790776695`, and `$25` remains protected through
Monday.
