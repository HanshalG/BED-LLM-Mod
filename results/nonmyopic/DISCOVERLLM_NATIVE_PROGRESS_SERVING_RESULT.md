# DiscoverLLM Native-Progress Serving Result

## Verdict

The exact native-progress serving interface **fails closed at follow-up
transition parsing**. The first five stages completed and parsed, but the
64-cell follow-up transition response contained one forbidden state:

```text
R2_O3_W4|R|P
```

An artifact cell may be `S`, `N`, or `T` under the frozen grammar, not `P`.
No follow-up feedback, terminal likelihood tier, policy score, or hidden-world
endpoint was generated.

The exact task/interface is closed without coercion or rerun. A root-stage
audit also shows that the generic action bank did not activate the intended
causal mechanism: all eight dialog cells were `D|N` and all eight artifact
cells were `R|N`, so there were zero probes and zero root advances.

## Integrity

- Run: `discoverllm-native-progress-serving-20260727T230000Z`
- Model: `openai/gpt-5.4`, temperature zero, no reasoning
- Completed logical requests / HTTP attempts / retries: `6 / 6 / 0`
- Prompt / completion / reasoning tokens: `41,093 / 3,300 / 0`
- Cost: `$0.1522325`
- Forced exits: `0`
- Private raw SHA-256:
  `2323d59ce760a7def523b80da74dd059d1928e53068efcbce462fd9b2d3ebbbb`

The invalid value was the only malformed transition among 64 follow-up cells,
but the preregistration forbids partial analysis or repair. The more important
scientific issue is upstream: a generic 90-word artifact attempt did not
fully satisfy every leaf of any source root.

## Decision

Close the exact generic action-bank route. A scientifically distinct successor
may use a fresh task and hypothesis-targeted shared actions:

- a contrastive dialog question that explicitly probes the four candidate
  priorities but cannot advance; and
- four LLM-written artifact alternatives, one per candidate priority, which
  can exploit DiscoverLLM's native best-alternative rule to satisfy and unlock
  roots.

Its transition interface must report only state-relevant deltas
(`advance`, `stay-clear`, `stay-vague`, `terminal`) rather than incompatible
classification/outcome pairs.

Authenticated post-run balance: `$32.663980094`. OpenRouter only. OatML
jobs: `0`.
