# Zendo Particle-Multiset Belief Smoke Preregistration

Date frozen: 2026-07-25, before any response for this interface.

## Motivation and scope

The preregistered `phi` final-readiness smoke stopped before scoring because one
of eight complete branch responses repeated one AST. This fresh development
smoke does not repair, deduplicate, reissue, or evaluate that response. It uses
the previously untouched official rule `mu`, a new seed, and entirely fresh
responses.

The single representational change is mathematically substantive: each generated
population is a **particle multiset**, not a set of unique support elements.
Repeated executable ASTs remain separate particles and retain multiplicity in the
uniform particle prior and every posterior update. This is ordinary particle
filter semantics and lets the generator express extra mass on a rule without
silently deleting or replacing a sample.

The scientific claim is unchanged: can target-blind final-readiness scoring select
a root that sacrifices immediate EIG because its outcome-conditioned regenerated
belief supports better two-step identification of the hidden rule?

## Frozen source and task

- Official repository:
  `https://github.com/topwasu/doing-experiments-and-revising-rules`
- Commit: `af07590c4f4f617a79791e173460e5a4322b727f`
- Case-file SHA-256:
  `6440c543ff491af13606b79c57384281fae4e8bc205366e67be06d1c81fbacbc`
- Task: `mu`; no prior paid Zendo call used this task.
- Seed: `24370`.
- Initial positive-scene SHA-256:
  `1a521da141ca4ec0179847734e70c0ccc3e6f73968fe3e70038521ccc5649759`
- Target-blind 256-scene pool SHA-256:
  `80df2912f62d4f8f553ee78ba91fe86742532aaeb6a03738b23ef3e0d8f10abc`

The official `mu` predicate is hidden from every model prompt and from experiment
selection. It is evaluated only after all branch populations and readiness scores
are frozen.

Pre-response mechanics amendment: the general Zendo helper's official-scene audit
cannot represent one `mu` scene with more than six blocks under the frozen
generator DSL. The endpoint therefore uses exactly 512 deterministic random legal
scenes from seed `24370 + 1009 * RULE_ORDER.index("mu")`, all within the same
one-to-six-block language. No official scene beyond the initial positive example
is included. Its canonical SHA-256 is
`34baa977b1bfcbba82cb514e96488c7c1f1933bc1fc48b86c9e5f96e6433bc99`.
This amendment was made after a zero-call fake-adapter test and before any `mu`
model response or hidden endpoint evaluation.

## Frozen interface

- Interface: `zendo-particle-multiset-belief-1`.
- Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.
- Exactly 10 physical requests:
  1. one initial 12-particle executable rule population;
  2. eight isolated 12-particle branch refreshes for four roots by two labels;
  3. one blinded four-root final-readiness score.
- Particle IDs, count, fields, AST grammar, and JSON object remain strict.
- Duplicate ASTs are preserved as separate particles; there is no deduplication.
- No response repair, coercion, replacement, or scientific retry.
- OpenRouter only; no OatML.
- Projected cost `$0.25`; hard cap `$0.75`; preserve the global `$25` reserve
  and the through-Monday spend cap.

The target-blind compiler and controls are exactly the preregistered algorithm
from the prior final-readiness smoke:

- create 256 deterministic legal scenes;
- select four distinct informative prediction signatures nearest `1.00`, `0.75`,
  `0.50`, and `0.25` of maximum initial EIG;
- regenerate particles independently after each hypothetical root label;
- compute each branch's exact best continuation in the same pool;
- score complete pathways without hidden truth or immediate root EIG;
- reveal actual labels only after the score parses;
- evaluate posterior-weighted behavioral truth agreement over the frozen
  512-scene random legal audit bank after both realized observations.

Myopic and fixed-support depth-two controls share the four roots and the entire
continuation bank. Fixed support also preserves initial particle multiplicity.

## Frozen gates

Mechanics must all pass:

- exactly 10 adapter requests and 10 HTTP attempts;
- zero retries, reasoning tokens, forced exits, repairs, and parse failures;
- every initial/branch population has exactly 12 valid AST particles and at least
  eight unique AST values;
- at least six initial behavioral signatures;
- four distinct informative root signatures, each minority-label probability at
  least `0.10`;
- at least four behaviorally distinct refreshed branch populations;
- every exact continuation has positive finite EIG;
- scorer values vary and have a unique maximum;
- cost at most `$0.75`.

Scientific gates must all pass:

- model root differs from myopic;
- model sacrifices at least `0.01` nats immediate EIG;
- realized endpoint range is at least `0.10`;
- model exceeds myopic realized weighted truth agreement by at least `0.05`;
- model exceeds fixed-support depth two by at least `0.03`;
- score versus realized endpoint Spearman correlation is at least `0.30`.

A full pass authorizes only a separately preregistered multi-rule confirmation.
Failure closes this exact multiset interface. No rule, seed, threshold, parser,
particle interpretation, prompt, or pool will change after outcomes.
