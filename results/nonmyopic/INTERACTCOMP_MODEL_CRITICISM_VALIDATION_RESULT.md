# InteractComp Model-Criticism Validation V1 Result

Date: 2026-07-25

## Outcome

**V1 failed closed on classification grammar before auxiliary support, hidden
context, or any scientific endpoint. The exact V1 screen and interface are
closed.**

The prospective screen successfully found six collapsed current supports among
the 16 untouched tasks. Root generation also completed. One of 48
current-particle classification responses was `YYY Y`, containing an internal
space instead of the frozen exact four-character grammar. The strict parser
rejected it.

## Execution

| Stage | Completed calls |
|---|---:|
| Initial particles for 16 screen tasks | 128 |
| Four roots for six enrolled tasks | 24 |
| Current-particle classifications | 48 |
| Auxiliary generation/validation/classification | 0 |
| True responder | 0 |
| Realized refreshes | 0 |
| **Total** | **200** |

The six enrolled dataset indices were `141, 97, 84, 21, 123, 59`
(benchmark IDs `142, 98, 85, 22, 124, 60`) with exact normalized current-support
unique counts `3, 3, 4, 1, 2, 4`. This confirms that the prospective collapsed
support criterion was feasible without target-based selection.

The single malformed classification was response 7 in the enrolled
classification batch. It had the intended four labels separated by one space,
but the preregistered parser allowed no internal whitespace. There was no
transport retry, repair, whitespace normalization, response replacement, or
partial scientific analysis.

## Endpoint Integrity

The private checkpoint contains only:

- screen particles;
- enrolled indices and support-diversity counts;
- generated roots; and
- current-particle classifications.

It contains no auxiliary population, hidden context, true response, refreshed
support, target answer, or endpoint. The GPT-5.4 responder made zero requests.
All 16 screened task contexts and answers remain sealed.

Per preregistration, V1 receives no same-interface rerun. A distinct V2 may use
an entirely fresh screen and prospectively remove ASCII whitespace before
applying the same four-label grammar. Before a V2 formal run, the unreached
auxiliary-generation, semantic-validation, and classification path should pass
a small target-free serving smoke on already-open task inputs.

## Integrity And Cost

- Preregistered commit: `7907dc9`.
- Run ID: `interactcomp-model-criticism-validation-20260725T111000Z`.
- Model used: `openai/gpt-5.4-mini`, non-thinking.
- Requests/attempts: `200/200`.
- Retries/reasoning tokens/forced exits: `0/0/0`.
- Cost: `$0.09922275`.
- Private raw SHA-256:
  `691130ba00f29936ba1bb250b84b6e7a0e5692dd62e48c94ec80c5cf3c696c06`.
- Public failure SHA-256:
  `f05af626a555b6e7b41f9394271d0f167145e68841b193bb0f36cc105f79cb19`.
- Project-ledger spend after failure: `$86.37242356920763`.
- Monday local allowance remaining: `$14.770881249999789`.
- Authenticated OpenRouter endpoint still reported `$44.111601884` remaining,
  or `$19.111601884` above the protected `$25` reserve; the stricter local
  ledger controls while provider accounting lags.
- OatML resources used: none.
