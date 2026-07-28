# Number Game Depth-Three Development V1 Result

Date: 2026-07-28

## Verdict

**Transport failure; scientific endpoint unmeasured.**

The frozen Gemini-planner/GPT-target run completed four of eight trees. On
tree five, one paid normal-stop Gemini response contained 3,843 completion
tokens but was not exact JSON. The parser failed closed. Partial trees were
not aggregated or inspected for target endpoint effects.

The run had 249 successful calls before failure. A separate one-call attempt
to require a provider endpoint that explicitly supports the JSON-schema
parameter returned OpenRouter HTTP 404 before a model response: no such
Gemini endpoint was available under strict parameter routing.

Raw partial-response SHA-256:
`dfe3b96d61214902d0d443a3ee698090070c6b69aaffd12ae419ed82e2b1fb9a`

V2 changes only model assignment and seeds: GPT-5.4 Mini, which previously
returned 576/576 exact structured planning responses, becomes the planning
generator; Gemini becomes the independent target generator. The scientific
method and promotion rule remain unchanged.
