# InfoQuest Cached-Trajectory V2 Polarity Amendment

Recorded after the first V2 artifact was written and before any replay.

## Bug

Every structural, missingness, and substantive gate in the first V2 artifact
is true. The artifact also correctly records:

```json
"semantic_content_emitted": false
```

However, the internal gate dictionary used the same negative-polarity entry:

```json
"semantic_content_emitted": false
```

The top-level verdict was computed as `all(gates.values())`, so this correctly
redacted artifact received a false top-level verdict solely because `false`
was passed to `all`.

Original artifact SHA-256:
`ca24275121e366314719155aa97fb072f8e862e7c9fe2d27e97be6c3dcb69015`.

## Frozen Correction

Change only the internal gate key/value to:

```json
"semantic_content_not_emitted": true
```

Keep the separate descriptive output field
`"semantic_content_emitted": false`.

The replay must use the same code, source files, hashes, split, records,
metrics, thresholds, tokenization, and controls. Apart from:

1. the corrected gate key/value;
2. the top-level `passed` field;

the replay artifact must be byte-equivalent after canonical JSON comparison.
Any other difference fails the correction and leaves V2 without a passed
verdict.

This is a deterministic reporting correction. It does not authorize changing
or recomputing any scientific endpoint.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
