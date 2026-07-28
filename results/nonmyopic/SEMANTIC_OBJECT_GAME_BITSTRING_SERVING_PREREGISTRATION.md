# Semantic Object Game Bitstring Serving Preregistration

Date: 2026-07-29

## Motivation And Boundary

The semantic planning side of array-schema V2 completed cleanly, but Gemini
rejected the nested membership-array schema before validation. V2 remains
failed closed and will never be recovered or scored.

This is a transport-distinct interface with fresh seeds. It represents each
complete 32-object extension as a fixed 32-character `0`/`1` string in the
published object order. This removes the large nested enum array while
preserving the same semantic object universe and exact membership information.

No V2 response is reused. No V2 endpoint exists, so the change is not informed
by efficacy.

## Two-Call Serving Gate

- interface: `semantic-object-game-depth-three-bitstring-1`;
- planning model: `openai/gpt-5.4-mini`;
- target model: `google/gemini-2.5-flash`;
- reasoning: disabled;
- temperature: `0.7`;
- planning seed: `36300`;
- target seed: `36301`;
- one initial 16-concept response from each model;
- exact two accepted requests;
- hard cost cap: `$0.08`.

The provider schema constrains the bit string to length 32. The unchanged local
semantic checks require only `0` and `1`, 3--29 positive members, distinct full
extensions, nonempty names, and nonempty coherent descriptions.

Every serving condition must pass:

1. exact two accepted requests and exact transport accounting;
2. zero reasoning tokens and zero forced exits;
3. cost at most `$0.08`;
4. at least 12 valid unique planning concepts; and
5. at least 12 valid unique target concepts.

The gate opens no scientific endpoint and computes no policy.

## Conditional Fresh Mechanics

Only a full serving pass authorizes one fresh mechanics tree:

- planning seed `36400`;
- validation seeds `36500--36503`;
- endpoint seeds `36600--36607`;
- exact 49 accepted requests;
- hard cost cap `$0.75`;
- same 32 objects, 16 concepts, six roots, complete-history twice-refreshed
  support, retained consistent parents, cross-fitted Brier root selection, and
  descriptive-only endpoint as the original mechanics preregistration.

All original mechanics gates remain unchanged: minimum support sizes, finite
six-root risks, nonzero depth-three risk range, and different depth-two and
depth-three selected roots. The mechanics run must hash-bind the passing serving
artifact.

Any serving failure closes the bitstring interface before mechanics. Any
mechanics failure closes it without repair, response reissue, partial endpoint,
or another representation.

OpenRouter only. OatML, Slurm, SSH, and cluster resources are not used.
