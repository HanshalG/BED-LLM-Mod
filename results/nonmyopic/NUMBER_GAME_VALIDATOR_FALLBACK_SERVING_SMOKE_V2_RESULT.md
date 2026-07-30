# Number Game Validator Fallback Serving Smoke V2 Result

Date: 2026-07-30

## Status

**Passed every frozen gate.** The corrected evaluator exercised ten fresh
Gemini 2.5 Flash validation-support requests through the deterministic
provider-error fallback adapter.

## Results

- exactly 10 accepted requests and 10 HTTP attempts;
- zero retries, provider-error retries, or fallback transitions;
- all ten responses passed the base strict-parser contract;
- valid unique support sizes:
  `22, 22, 23, 23, 22, 23, 23, 23, 23, 23`;
- zero reasoning tokens and forced exits;
- cost `$0.022474`, below the frozen `$0.04` cap.

No scientific tree, target, candidate root, or efficacy endpoint was used.
The full resilient96 V2 runner is authorized only after binding the exact
public result hash below.

## Artifacts

- Public `RESULT.json` SHA-256:
  `26b9bcf95b72e89d081163d844bbbb06f368e7c1b8b3119c875c2b31940f1dcc`
- Private raw responses SHA-256:
  `52b27b3f9c498819becd98c496783c1dd73d832129bcff964e29a2c8c65bb91a`
