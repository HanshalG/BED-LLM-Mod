# KnowU Dynamic-Support Mechanics V2 Preregistration

Date frozen: 2026-07-26

## Amendment

V1 serving failed at zero cost because OpenRouter exposed no GPT-5.4 chat
endpoint accepting the `response_format=json_schema` parameter. V2 changes
only the response transport:

- remove the unsupported provider JSON-schema parameter;
- request an exact, flat JSON object in the prompt;
- retain the same exact-key, type, range, length, uniqueness, atomicity, and
  threshold parsers with no fallback;
- retain zero repair, reissue, and retry.

The private fixture SHA-256 remains
`2cb9d0e34e3e13aee896b5d4f2c6bf0c470d1a92096a23128f6d8cc26905eff8`.

All scientific and budget criteria in
`KNOWU_DYNAMIC_SUPPORT_MECHANICS_PREREGISTRATION.md` remain unchanged:
GPT-5.4 nonthinking, six worlds, four hypotheses, four atomic roots, 24
physical single-question refreshes, semantic truth threshold 70, exact
10-request serving gate, exact 42-request mechanics gate, $0.10 serving cap,
$0.75 mechanics cap, and no OatML.

If prompt-only serving fails, do not run mechanics. If it passes, run mechanics
once without changing prompts or criteria.
