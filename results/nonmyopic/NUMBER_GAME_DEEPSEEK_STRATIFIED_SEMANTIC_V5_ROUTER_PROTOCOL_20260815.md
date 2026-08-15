# Number Game DeepSeek Stratified Semantic V5 Router Protocol

Date frozen: 2026-08-15 (Europe/London), after V4 transport closure and before
any V5 response, HTTP attempt, run artifact, or ledger component.

## Scope

V4 produced no accepted response and no semantic evidence because its exact
GMICloud provider order returned `404 No endpoints found`. V5 is a transport-only
successor. It retains every scientific choice from the V4 protocol unchanged:

- exact model `deepseek/deepseek-v4-flash-0731`, reasoning disabled;
- seeds `202608210000..202608210191`;
- temperature `0.6`, strict `{name, expression}` JSON, maximum 220 tokens;
- three 64-call groups at histories empty, `2=YES`, and `2=YES,3=NO`;
- two calls per each of 32 five-anchor signatures;
- the identical DSL, prompt, parser, semantic floors, parity-half gates,
  multiplicity gate, checkpoint ordering, endpoint privacy, and `$0.20` stage
  cap.

Seed reuse is permitted only because V4 accepted and banked zero responses.
There is no response-conditioned change.

## Routing Change

Remove the provider `order` and `allow_fallbacks=false` fields. Send only
OpenRouter `provider.require_parameters=true`, so the router may choose any
active exact-model endpoint supporting all requested parameters. The payload
still contains the exact seed, strict response format, and explicit reasoning
disable. Exact returned model, seed set, prompt hashes, stop finishes, and zero
reasoning/retry remain gated.

Before every block, read the exact endpoint catalog and define eligible rows as
active endpoints supporting `seed`, `reasoning`, and either `response_format`
or `structured_outputs`. Reserve exposure using the componentwise maximum price
across every eligible row, not the cheapest row or eventual provider. At freeze,
the maxima are `$0.20/M` prompt and `$0.50/M` completion tokens. Any higher
eligible price, missing capability, malformed row, or empty eligible set stops
the run; it cannot increase the ceiling.

The exact serving provider is reported descriptively if available but has no
scientific gate authority. Provider change cannot rescue malformed responses.

## Budget And Authority

Use fresh V5 binding, run, ledger, result, and failure paths. The Europe/London
Aug 15 account boundary remains opening cumulative usage `$220.339269126`; the
posted V3 cost `$0.007297280` and unrelated use count. V4 cost was zero. The
daily cap remains `$5.00`, V5 stage cap `$0.20`, concurrency at most 64, and
zero retries.

Every V4 transport, replay, semantic-obedience, diversity, privacy, and failure
gate remains exact. A pass authorizes only a separately frozen fresh full
mechanics protocol. A semantic null, transport failure, malformed artifact,
budget loss, or replay failure authorizes nothing. There is no retry, provider
pin experiment, seed substitution, prompt or threshold change, favorable-subset
scoring, or response reuse.
