# Range-Gated Rock Cached-h3 OpenRouter S0 Failure

The separately preregistered OpenRouter provider replication **fails before cell
acceptance and stops without rerun**.

At cell 0, Gemma 4 26B used all 4,352 completion tokens, including 4,129 reasoning
tokens. Before the response could be returned to the fixed-tail provider, token
logging raised:

```text
AttributeError: 'str' object has no attribute 'parent'
```

The new CLI had assigned `runtime_config.log_path` as a string, while the shared
logger requires a `Path`. Consequently:

- zero cells were accepted;
- no raw plan entered the policy interface;
- no route-quality or trajectory endpoint exists;
- the adapter made one request costing `$0.00153156`; and
- project spend became `$38.52285523`, leaving `$71.47714477`.

This is an implementation failure, not evidence about 26B spatial policy quality.
Per the frozen OpenRouter S0 rule, the provider-replication line is not repaired or
rerun. The same bug was fixed prospectively for the still-pending, response-free
direct-vLLM cluster line under a separate pre-response amendment.
