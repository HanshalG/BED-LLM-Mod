# ClariQ Multisample Likelihood Development V2 Manifest Amendment

## Status

Frozen after running the preregistered structural selector and before any
OpenRouter request or selected-topic utility-value access.

## Frozen Manifest

- Manifest:
  `results/nonmyopic/clariq_multisample_likelihood_v2_manifest/MANIFEST.json`
- SHA-256:
  `97af5ebc228dfff94e5c270f1dd0f10b862a8d5627a26bcb36266ac3a8d1ce73`
- Evaluation dictionary key presence inspected: yes.
- Evaluation utility values read by selector: no.
- API calls: `0`.

The first three eligible fresh development topics are:

| Topic | Facets | Valid roots | Initial request |
|---|---:|---:|---|
| `136` | 3 | 14 | Tell me about american military university. |
| `125` | 3 | 13 | butter and margarine |
| `149` | 3 | 14 | Tell me about uplift at yellowstone national park |

Their exact sorted valid-root question objects and facet descriptions are
stored in the hash-locked manifest. No other question can enter V2.

## Exact Request Count

There are `41` manifest roots. Five independent semantic likelihood samples
per root produce exactly `205` physical requests and HTTP attempts. The V2
runner rejects any manifest hash, topic order, root count, or request count
change.

All models, prompts, sampling, smoothing, policies, controls, endpoint timing,
scientific gates, `$0.50` cap, and no-OatML rule remain byte-for-byte or
semantically unchanged from the V2 preregistration.
