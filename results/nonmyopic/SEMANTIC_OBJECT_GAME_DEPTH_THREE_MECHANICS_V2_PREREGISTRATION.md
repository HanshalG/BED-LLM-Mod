# Semantic Object Game Depth-Three Mechanics V2 Amendment

Date: 2026-07-29

V1 failed before any response or charge because Azure structured output rejects
the JSON Schema keyword `uniqueItems` on arrays.

V2 changes exactly one transport detail:

- remove `uniqueItems: true` from the provider-facing `members` array schema.

The strict local parser remains unchanged and rejects any member list whose set
size differs from its list size. Therefore the accepted response language and
all scientific semantics are identical to V1.

Everything else remains frozen by
`SEMANTIC_OBJECT_GAME_DEPTH_THREE_MECHANICS_PREREGISTRATION.md`, including:

- object universe, models, prompts, proposal count, and complete membership
  requirement;
- seeds, temperature, support retention, tree construction, and cross-fitting;
- exact 49 accepted requests;
- all transport, support, depth-activity, and `$0.75` gates;
- no semantic repair, response reissue, dropped support, or partial result; and
- descriptive-only status of the one-tree endpoint.

Any further provider-schema or response failure closes this exact interface.
