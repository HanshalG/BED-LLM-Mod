# Cleveland Heart Workup 26B Serving Smoke V2 Preregistration

Registered 2026-07-23 after S0 v1 failed on local-index scope and before any v2 model
response was observed.

V2 changes only the serving prompt and correction message:

- state prominently that each root has its own local menu and indices cannot be
  carried across roots;
- append `FINAL_OUTPUT_LIMITS` after the full root payload, giving each root's exact
  branch count and inclusive local integer range;
- repeat the local-menu rule in the single bounded correction message.

Everything else is frozen: non-thinking `google/gemma-4-26B-A4B-it`, direct vLLM on
an A100 in `msc,llm` excluding `oat12`, K4 fixed roots, temperature zero, 128-token
cap, one retry, exact parser, and zero rollout/scoring LLM calls. V2 uses fresh S0 seed
`24138` and a separate output directory. It passes only if all ten cells resolve, all
policies are complete/legal, the workup root and workup branch follow-ups remain
representable, and usage has zero reasoning tokens and forced exits.

If v2 fails, stop the Heart indexed interface rather than making another repair. If
it passes, run the already-frozen S1 proposal gate unchanged at seed `24137`; S0
responses and choices are not proposal-quality evidence.
