# HoVer Support-Regeneration Serving Result

## Outcome

**Pass.** The endpoint-free flat-text interface completes every frozen serving
gate on the first already-open HoVer mechanics task.

- Public result SHA-256:
  `e44acf8b3a93302d8ab964681c18b0513a4b0db994d5c891c919d112ae412983`.
- Private raw response SHA-256:
  `6f5eb4a842fb279d11ab7fd10f579283246fef2f4fbe225c6f4f239f6804119d`.
- Exact physical requests / HTTP attempts: `10 / 10`.
- Transport retries: `0`.
- Reasoning tokens / forced exits: `0 / 0`.
- Responses parsed without repair: `10 / 10`.
- Regenerated states distinct from initial: `8 / 8`.
- Pairwise-distinct regenerated states: `8 / 8`.
- Future-uplift scores: `[92, 96, 91, 83, 90, 82, 86, 78]`.
- Cost: `$0.1249525`, below the frozen `$0.15` cap.
- Endpoint loaded: `false`.
- OatML jobs: `0`.

The future vector is nonconstant but compressed toward high values. This is a
mechanics observation, not a post-hoc gate change. The contingent two-task
mechanics stage remains authorized under its unchanged score-fidelity and
external-endpoint gates.

The first shell invocation failed before importing project dependencies because
the system Python lacked NumPy. It made no API request and emitted no artifact.
The exact command was restarted once using the existing
`20_questions_env`; no prompt, data, response, or gate changed.
