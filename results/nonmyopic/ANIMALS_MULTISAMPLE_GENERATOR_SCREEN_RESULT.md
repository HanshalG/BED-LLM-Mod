# Animals Multi-Sample Generator Screen Result

Status: **development gate passed; scorer development authorized**.

All 20 inspected states and 120 fixed counterfactual branches completed.
Four independent diverse Gemma 4 26B generation calls per branch recovered the
hidden truth in `8/20` branch unions (`40%`), exactly meeting the frozen gate.
The references using one list per branch were Gemma `3/20` (`15%`) and
GPT-5.4 Mini `1/20` (`5%`).

All eight union-covered states had nonzero candidate coverage. Generation
produced broad merged pools, commonly 20-25 unique structurally clean names per
branch after four nominal 16-name calls, before the unchanged validator and
history filter.

This isolates the useful change: independent hypothesis-space sampling, not a
larger semantic model. It makes open-world recovery sufficiently frequent for
a target-blind branch-content scorer development gate, but is not itself a
policy result.

Serving used 11,399 requests, 1,678,925 prompt + 174,450 completion tokens,
zero reasoning, and `$0.21086924`. Project spend is `$43.39492682` of `$110`.

Artifacts are in
`results/nonmyopic/animals_multisample_generator_recall/gemma26b_seed24279/`.
