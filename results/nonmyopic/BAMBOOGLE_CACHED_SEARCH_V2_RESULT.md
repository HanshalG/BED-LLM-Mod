# Bamboogle Cached-Search v2 Result

## Decision

The structured-serving prerequisite failed, so contingent mechanics v2 was
not run. The Bamboogle cached-search interface is closed for the current
project; there will be no third serving attempt.

## Exact Failure

- Structured logical model requests completed: `0`.
- OpenRouter HTTP attempts: `5`.
- HTTP result: five provider-routing `404` responses.
- Provider message:
  `No endpoints found that can handle the requested parameters.`
- Adapter model cost: `$0.00`.
- Model retries / reasoning tokens / forced exits: `0 / 0 / 0`.
- Wikipedia requests: `0`.
- Scientific endpoints: none.
- Public failure artifact SHA-256:
  `707cd4da8d6e2636a1a75244929da9152b0ccd88b5b753e7345917348938a45d`.
- Private raw checkpoint SHA-256:
  `99b7421a6a93a1b1a2bdcf56e8190ccd0af9e8094bddd72ac8e1d10b17aeacd3`.
- OatML use: none.

The adapter's nested cumulative tracker fields include the prior v1 run
because both commands used the same run identifier. The authoritative
per-adapter fields for this invocation are zero physical model requests and
zero adapter cost; the five HTTP attempts were rejected before inference.

## Interpretation

OpenRouter currently has no `openai/gpt-5.4` endpoint satisfying the strict
JSON Schema parameter requirement. This is a transport capability failure,
not a semantic-search result. Under the contingent preregistration, Phase A
failure closes Phase B, so no response was reused and none of the 65 model or
60 retrieval actions occurred.

Bamboogle still appears unsaturated from the no-search screen, but this exact
mechanics route has generated no evidence about non-myopic policy quality.
Future work would need a new environment or a substantially different,
prospectively justified interface rather than another parser/provider retry.
