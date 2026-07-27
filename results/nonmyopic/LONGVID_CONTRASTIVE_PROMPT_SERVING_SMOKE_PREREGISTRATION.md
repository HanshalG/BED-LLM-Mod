# LongVid Contrastive Prompt-Only Serving Smoke Preregistration

## Status

Frozen before any response. This is a transport-only synthetic smoke after
the strict-schema provider-routing failure. It contains no LongVid question,
caption, answer, evidence ID, policy score, or endpoint.

## Protocol

Use `openai/gpt-5.4` through OpenRouter with reasoning disabled,
temperature `0`, concurrency `8`, and zero retries.

The public synthetic fixture makes exactly:

- one initial six-hypothesis evidence-chain support call;
- eight independent observation-conditioned support-refresh calls; and
- one contrastive four-step belief-trajectory rank call.

Total: exactly `10` physical requests and HTTP attempts.

The output route is ordinary chat with an exact prompt-only flat JSON
template. Parsing remains strict: one JSON object, exact fields, six distinct
hypotheses and queries, nonuniform integer weights, and no repair, extraction,
reissue, or fallback. Unicode anchors are accepted as strings and grounding
is checked after the same tokenizer used by retrieval.

## Frozen Gates

All must pass:

- exactly `10` requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all ten outputs parse exactly;
- all eight refreshed supports differ from the initial support;
- all eight refreshed supports are mutually distinct;
- every refresh has at least four observation-grounded anchor/query pairs;
- the contrastive choice and confidence parse;
- cost is at most `$0.20`.

Failure closes prompt-only contrastive LongVid transport. Passage authorizes
only a separately frozen development mechanics gate on new tasks; it is not
scientific evidence.

## Budget

Projected cost is `$0.08`; hard cap is `$0.20`. Recheck the authenticated
balance before launch and retain at least `$25` through Monday, 3 August
2026. OpenRouter only; no OatML, Slurm, or cluster use.
