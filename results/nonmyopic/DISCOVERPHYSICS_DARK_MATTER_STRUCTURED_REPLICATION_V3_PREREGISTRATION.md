# DiscoverPhysics Default-Routing Structured Replication V3

## Status

Frozen before any V3 model response. V2 is closed at its uncharged discarded
preflight and is not repaired, rerun, or rescored.

## Prospective Transport Change

V2 combined a strict `response_format` with
`provider.require_parameters=true`. Its inherited generic payload also sent
sampling fields not advertised by any live GPT-5.4 endpoint, so OpenRouter
filtered all providers and returned `404` before accepting a response.

V3 changes exactly one payload field:

```json
{"provider":{"require_parameters":false}}
```

This is OpenRouter's documented default. Every currently listed GPT-5.4
endpoint advertises `response_format` and `structured_outputs`; default routing
allows a provider to ignore unrelated unsupported sampling knobs rather than
discarding the route. Strict JSON Schema remains present and mandatory in the
request and is still checked again by the frozen semantic parser/compiler.

No prompt, model, schema, semantic field, parser, compiler, scientific
hyperparameter, seed, control, or threshold changes.

## Exact Run

Use:

- model `openai/gpt-5.4`;
- reasoning disabled;
- requested temperature `0`;
- the exact V2 initial and branch-conditioned prompts;
- the exact V2 strict initial and refresh JSON Schemas;
- no clipping, coercion, extraction, partial support, semantic retry, or
  manual edit;
- bounded logged transport retries only;
- one discarded initial-support route preflight; then
- one scientific initial support and eight scientific branch refreshes.

The exact accepted-response target remains `10`: one discarded preflight plus
nine scientific calls. The V2 attempt accepted zero responses, so this does
not reuse model output.

## Phase A Gates

All V2 gates remain unchanged:

- exactly 10 accepted responses;
- `http_attempts == requests + retry_count`;
- zero reasoning tokens and zero forced exits/finals;
- all ten responses pass strict schema and semantic compilation;
- all eight refreshes differ from the scientific initial support;
- at least three roots have branch-distinct supports;
- center B branches choose distinct continuations;
- every refresh keeps at least two regions;
- cost at most `$0.25`;
- immediate EIG selects D;
- retained-support lookahead selects B; and
- B reduces internal retained-support risk by at least `10%` versus D.

Failure closes V3 before the physical endpoint. No alternate provider order,
schema relaxation, ordinary-JSON fallback, or second V3 tree is allowed.

## Untouched Endpoint

Only after Phase A passes and the model/policy files are written and hashed:

- map seeds `24700--24715`;
- noise seed `24716`;
- bootstrap seed `24717`;
- 384 maps, 96 per region;
- NE/NW/SW/SE prior `.4/.3/.2/.1`;
- `.95` initial-support and `.05` routed-refresh mass;
- exact full-history official-simulator likelihoods;
- 8 root and 4 continuation samples per map; and
- retained B, myopic D, random A, and same-root fixed B.

V2 opened none of these seeds.

## Scientific Gates

All must pass:

- at least `10%` retained-B MSE reduction versus myopic D;
- positive paired-bootstrap lower bound for `MSE(D)-MSE(B)`;
- at least `5%` retained-B reduction versus random A;
- at least `1%` retained-B reduction versus same-root fixed B;
- positive paired-bootstrap lower bound for
  `MSE(fixed B)-MSE(retained B)`; and
- at least `5%` retained-union nearest-support risk reduction.

Any failure is a replication null. No seed, subset, weight, threshold, prompt,
schema, provider field, or model response may be changed after the result.

## Claim Scope

A pass would replicate the prior LLM-generated path-dependent support gain
with a fresh support tree and fresh physical endpoint under a robust structured
transport. It would establish an LLM-native hypothesis-space contribution, not
LLM likelihood estimation, and would remain one physical domain with a
deterministic semantic-to-executable compiler.

## Accounting

- Projected OpenRouter cost: `$0.14`
- Hard run cap: `$0.25`
- Authenticated balance before V3: `$8.990154844`
- Reserve: none
- OatML, Slurm, SSH, and cluster use: prohibited
