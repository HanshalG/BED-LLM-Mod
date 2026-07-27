# DiscoveryWorld Deterministic Replay Preregistration

Frozen: 2026-07-27, before running the patched eight-theme replay.

## Purpose

Test whether a minimal deterministic wrapper makes the pinned official
DiscoveryWorld simulator suitable for paired counterfactual policy evaluation.
This is a mechanics gate only. It does not test an LLM policy or scientific
efficacy.

## Source

- Repository: `https://github.com/allenai/discoveryworld`
- Required commit:
  `fd591323920be0d3786ef350955de1945aa571e5`
- Difficulty: `Challenge`
- Scenario seed: `0`
- Themes: all eight official scientific themes
- Replay length: five actions after the initial observation

## Frozen Wrapper

Before each replay:

1. seed Python's global RNG and NumPy's RNG from a SHA-256 hash of the
   interface version, theme, difficulty, and scenario seed;
2. patch `Object.__init__` so each object's private RNG is seeded from a
   SHA-256 hash of the interface version, world seed, and object UUID; and
3. instantiate a fresh official `DiscoveryWorldAPI`.

The action schedule cycles through sorted official teleport locations. If a
theme has no teleport locations, it uses the fixed direction sequence north,
east, south, west. Each action is followed by one official world tick and one
observation.

## Frozen Gate

For each theme, two fresh replays must agree exactly on:

- process seed;
- action sequence;
- action results;
- all six canonical UI observation hashes;
- all six oracle scorecard hashes;
- final normalized score;
- completion flag; and
- successful-completion flag.

The gate passes only if all eight themes pass. A partial pass is a failure; no
theme subset, changed action schedule, additional seed, or relaxed comparison
is permitted.

## Consequence

- **Pass:** authorize one separately frozen, zero-call source opportunity audit
  for a native scientific task. No OpenRouter call is authorized.
- **Failure:** close this wrapper. Do not repair after inspecting which theme or
  field diverged.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
