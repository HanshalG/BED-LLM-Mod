# Strategy-Prior Successor: Nemotron 3 Super 120B Serving Registration

Registered on 2026-07-18 after Qwen, DeepSeek, and gpt-oss each failed the strict
no-repair serving gate for distinct documented reasons. This is a permitted model swap
inside the separate successor capability probe; it does not alter the closed original
L3 result.

`nvidia/nemotron-3-super-120b-a12b` is selected from the current OpenRouter catalog
because it is a 120B reasoning model that explicitly reports `supports_max_tokens:
true` for reasoning. Requests use `reasoning.max_tokens: 768` and total
`max_tokens: 1536`, reserving an equal final-response budget. The mandatory gate is
unchanged: five L1 plus five L3 no-repair parse-and-execute cells, all successful and
with zero forced exits.

At 1,373 completed-run reference calls, the total 1,536-token ceiling at the catalog
completion price of `$0.455/M`, plus historical prompt volume at `$0.21/M`, is about
`$1.06`; the independent formal ledger cap remains `$3.00`. Only a smoke pass may
launch the frozen 30-paired-trial L3 comparison.
