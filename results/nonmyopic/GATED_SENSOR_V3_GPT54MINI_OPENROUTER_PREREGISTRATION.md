# Gated Sensor v3 GPT-5.4 Mini OpenRouter Gate

Registered 2026-07-22 after the user added OpenRouter credit and before any v3
GPT-5.4 Mini response. The authenticated credit endpoint reported `$80.00` total
credits and `$40.208810965` lifetime usage. The project ledger remains capped at
`$40.00` and currently records `$30.59453806`, leaving enough for this gate without
raising the project cap.

## Stage S0: Serving Smoke

- Non-thinking `openai/gpt-5.4-mini`, temperature zero, indexed v3, K4.
- Two paired trials, four rounds, seed `24111`, 200 bootstrap replicates, trial
  concurrency two.
- Expected physical requests: ten accepted cells before any retry. Hard run cap:
  `$0.25`; expected cost below `$0.10`.
- Passes only with a completed result, zero invalid responses, zero reasoning tokens,
  zero forced exits, all mechanics true, and zero rollout-scoring LLM calls.
- Entropy/truth and continuation-quality values are descriptive and cannot qualify
  the interface.

## Stage S1: Proposal-Quality Gate

Stage S1 is authorized only if S0 passes. It uses the same model, prompt, K4 roots,
temperature, and exact verifier with four paired trials, eight rounds, fresh seed
`24112`, 2,000 bootstrap replicates, and trial concurrency four. It is a smoke-level
proposal diagnostic, not a policy endpoint.

The pre-existing v3 qualification rule is unchanged. Over 28 nonterminal h2 states:

- mean exact same-root continuation efficiency must be at least `0.90`; and
- it must exceed matched-random continuations evaluated on the exact same
  LLM-reached beliefs by at least `+0.05`.

All actions must be legal, every registered mechanic must pass, and the run must have
zero reasoning tokens, forced exits, and rollout-scoring LLM calls. Root coverage,
proposal/exhaustive-d2 fraction, reached-policy entropy AUC, and truth-log AUC are
reported but cannot rescue a failed continuation gate. No threshold, seed, or model
replacement is allowed after responses.

Passing S1 authorizes a separately preregistered fresh 30-pair confirmation with
entropy AUC primary, truth-log AUC corroborating, shared d1, exhaustive d1,
matched-random strategies, and exhaustive d2. S0/S1 trials are never pooled into that
confirmation.
