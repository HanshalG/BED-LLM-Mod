# Bamboogle Cached-Search Semantic-Belief Mechanics Result

## Decision

Version 1 failed closed at the initial response boundary. No retrieval action,
belief refresh, policy selection, or scientific endpoint was evaluated.

## Exact Failure

- Initial model requests: `5/5`.
- HTTP attempts: `5`.
- Parsed strict JSON objects: `4/5`.
- Model retries / reasoning tokens / forced exits: `0 / 0 / 0`.
- Logical / physical Wikipedia requests: `0 / 0`.
- Cost: `$0.0220475`.
- Public failure artifact SHA-256:
  `23e3c6fbc27837ff514ce32bac3ed8aa64ffac645a3bf3a4edd98270257f0262`.
- Private raw checkpoint SHA-256:
  `2178d519df780e861ce2784b380b4f1b843fdb1e673984b86590ab13644fbed8`.
- OatML use: none.

The rejected response contained one valid 24-field JSON object followed by one
extra closing brace. The other four responses were exact objects. The
preregistered parser accepted neither trailing text nor object extraction, so
the run stopped before any search.

## Interpretation

This is an interface-serving failure, not evidence for or against adaptive
semantic search. It leaves every scientific mechanics endpoint missing.
Version 1 will not be repaired or rerun.

The existing OpenRouter adapter has provider-enforced strict JSON Schema
support that was not used in version 1. A new version may use it only through
a separately committed serving gate that freezes the full contingent v2
protocol before making another call. That is a transport change; the
scientific prompts, entropy policies, endpoint metrics, and pass thresholds
must remain unchanged.
