# Strategy-Prior Successor: DeepSeek V4 Pro Serving Registration

Registered on 2026-07-18 after, and because of, the Qwen 397B serving-gate failure.
This is the allowed model swap within the distinct successor capability probe; it does
not alter or retry the original L3 registration.

`deepseek/deepseek-v4-pro` is selected from the current OpenRouter catalog because it
supports provider-native `reasoning_effort`, has a 1,048,576-token context window, and
is a stronger reasoning candidate than the original Gemma generator while remaining
inside the successor's hard cost envelope. The smoke uses `reasoning_effort: medium`
and `max_tokens: 2048`, leaving the provider room to reason and still return the strict
JSON cell.

The mandatory gate remains exactly ten no-repair parse-and-execute cells: five L1 and
five L3. It has a `$0.05` hard cap. Only if all ten succeed with no forced exit may the
frozen 30-paired-trial L3 successor run begin.

For the formal run, the historical 1,373-call volume times 2,048 maximum completion
tokens at the catalog completion price of `$0.87/M`, plus the original 498,924 prompt
tokens at `$0.435/M`, gives a conservative approximately `$2.66` cost bound. The run
ledger independently enforces the `$3.00` cap.
