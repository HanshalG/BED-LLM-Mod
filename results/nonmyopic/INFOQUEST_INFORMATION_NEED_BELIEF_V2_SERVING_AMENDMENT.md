# InfoQuest Information-Need Belief V2 Serving Amendment

Frozen after V1 dry validation and before any live information-need response.

V1 was preregistered in commit `59bc74f`. This prospective amendment changes
only the synthetic transport qualification to match the project's standard
ten-call serving smoke:

- interface version:
  `infoquest-information-need-belief-1` to
  `infoquest-information-need-belief-2`;
- expected serving requests and HTTP attempts: `2` to `10`;
- serving projected cost: `$0.02` to `$0.06`;
- serving hard cap: `$0.03` to `$0.08`.

All ten synthetic responses must parse, contain five distinct needs, and have
nonconstant action scores. Zero retry, reasoning-token, and forced-exit gates
remain unchanged.

The LLM-owned five-need representation, exact response schema, GPT-5.4
non-reasoning model, temperature, expected-resolved-mass formula, cached
histories, hidden target labels, 30-call mechanics protocol, `$0.35` mechanics
cap, all scientific thresholds, no-repair rule, and development-only
interpretation are unchanged.

The V1 two-call dry artifact and the V2 ten-call oracle dry artifact are
transport tests only and are excluded from scientific evidence. No live
response existed when this amendment was frozen.

The combined V2 serving-plus-mechanics hard cap is `$0.43`, leaving at least
`$1.65203240` of the stricter operational allowance through Monday. No OatML
job is used.
