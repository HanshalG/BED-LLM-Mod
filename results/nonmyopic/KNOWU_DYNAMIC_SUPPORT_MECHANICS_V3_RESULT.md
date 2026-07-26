# KnowU Dynamic-Support Mechanics V3 Result

Date: 2026-07-26

Status: **failed closed before user simulation or scientific endpoint**

The six initial GPT-5.4 policy calls completed, but the parser reported
non-distinct dimensions. Raw responses were retained under SHA-256
`3f9a1512b4514920a77086db0177c4d788915b3d3c4167e9ddb2181e7c47b11f`.

Inspection showed the outputs were semantically distinct. The parser's
ASCII-only normalizer mapped every Chinese dimension and hypothesis to the
empty string. It would also reject the full-width Chinese question mark used
in two otherwise valid questions. This is an implementation-format failure,
not an observed support-expansion outcome.

Recorded usage:

- accepted requests and HTTP attempts: 6
- prompt tokens: 5,173
- completion tokens: 1,830
- reasoning tokens, retries, and forced exits: 0
- cost: $0.0403825

No profile-conditioned answer, refreshed support, semantic truth judgment, or
scientific gate was evaluated.

V4 may make only a Unicode parser amendment and continue from the exact cached
six responses. It must not regenerate the initial support, alter prompts, or
inspect any unavailable outcome. Composite V3+V4 accounting must still equal
the preregistered 42 requests.
