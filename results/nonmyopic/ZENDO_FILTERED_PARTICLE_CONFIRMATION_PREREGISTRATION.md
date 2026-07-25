# Zendo Filtered-Particle Confirmation Preregistration

Date frozen: 2026-07-25, before any response for this interface.

## Amendment and disclosure

The free-form seven-rule confirmation made seven initial calls but stopped before
branches, scorers, policy selection, or hidden endpoints when one of 36 stored
particles had an invalid cross-field AST. Both provider-enforced schema routes
then failed before generation. The scientific hypothesis remains unmeasured.

This is a prospectively changed particle-validation interface, not a repair or
continuation of those responses. All 84 responses are generated fresh. Prior
responses are not supplied to the model or reused. The same seven public tasks
have now received initial-support calls, so this is iterative development rather
than a pristine one-shot confirmation; that limitation will be reported.

## Single serving change

Every response must still be one strict JSON object with exactly 12 ordered
particle rows, exact IDs/fields, and nonempty rule text. Each AST is validated
independently:

- valid ASTs enter the particle multiset unchanged;
- invalid AST rows are **filtered out**;
- invalid rows are never normalized, repaired, deduplicated, replaced, or
  reissued;
- every population must retain at least eight valid particles and at least eight
  unique valid ASTs;
- the aggregate invalid-particle fraction across all 63 generated populations
  must be at most `0.15`.

This treats malformed generations as failed particle samples, which is standard
for a generative particle proposal. Multiplicity among valid duplicate ASTs is
still preserved.

Top-level JSON errors, wrong row counts/IDs/fields, empty descriptions, fewer than
eight valid particles, or later scorer parse errors remain fail-closed.

## Frozen scientific protocol

Everything else is identical to
`ZENDO_PARTICLE_MULTISET_CONFIRMATION_PREREGISTRATION.md`:

- interface: `zendo-filtered-particle-confirmation-1`;
- source commit and case hash unchanged;
- tasks: `upsilon`, `iota`, `kappa`, `omega`, `nu`, `xi`, `psi`;
- base seed `24371`, stride `7919`;
- committed scene/audit/random manifest unchanged;
- model `openai/gpt-5.4`, temperature zero, explicit non-reasoning;
- exact 84 calls: 7 initial, 56 branch refresh, 21 aligned/root-only/shuffled
  scorers;
- target-blind 256-scene pools, four EIG-spanning roots, exact branch
  continuations, and 512-scene hidden behavioral endpoints;
- aligned, root-only compute match, cyclic shuffled future, exact myopic,
  fixed-support depth two, and deterministic random controls;
- hidden predicates evaluated only after every population and score freezes;
- no response repair, replacement, scientific retry, or partial favorable subset;
- projected cost `$1.10`, hard cap `$1.50`;
- OpenRouter only, no OatML, preserve the `$25` reserve.

## Frozen gates

All prior mechanics gates remain, plus:

- every population has at least eight valid particles;
- aggregate invalid-particle fraction at most `0.15`.

All prior scientific gates remain byte-for-byte in code:

- endpoint range at least `0.10` on at least 4/7;
- aligned differs from myopic on at least 4/7;
- immediate sacrifice at least `0.01` on at least 3/7;
- aligned versus myopic mean gain at least `0.04`, at least four wins, at most
  two losses, exact task sign-flip `p <= 0.10`;
- aligned versus fixed/root-only/shuffled mean at least `0.03`, at least four
  wins, at most two losses for each;
- aligned versus random mean at least `0.04` and at least four wins;
- mean aligned score-endpoint Spearman at least `0.25`;
- aligned Spearman exceeds root-only and shuffled by at least `0.15`;
- aligned pairwise accuracy exceeds root-only and shuffled by at least `0.05`.

Failure closes this filtered-particle interface. No further Zendo parser or
serving amendment will be made on these tasks.
