# Animals Branch-Generator Recall Screen Result

Status: **screen failed; model-only repair rejected**.

All 20 inspected states and 120 fixed counterfactual branches completed under
the production belief-update pipeline. Non-reasoning GPT-5.4 Mini recovered
the hidden target in only `1/20` branch unions (`5%`), versus the fixed Gemma 4
26B reference of `3/20` (`15%`). The preregistered requirement was at least
`8/20`.

Only one state had any nonzero candidate truth coverage. The driver log shows
many GPT generations parsing to a single raw animal followed by minimum-support
retries. This may partly reflect an interface/model mismatch, but the frozen
screen tests the current production prompt and parser; it cannot be repaired or
relabeled after observing the endpoint.

The result rejects a model-only substitution as a way to make open-world
branch recovery identifiable. A distinct future line would need to change the
hypothesis-generation interface itself and qualify its breadth before any
policy scoring.

Serving used 6,916 requests, 948,533 prompt + 104,101 completion tokens, zero
reasoning tokens, and `$1.17985425`. Project spend is `$43.18405758` of `$110`.

Artifacts are in
`results/nonmyopic/animals_branch_generator_recall/gpt54mini_seed24279/`.
