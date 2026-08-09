# Bongard OpenWorld August 10 V2 Readiness Addendum

Checked: 2026-08-09 13:10 Europe/London, before any August 10 model
response, candidate label, endpoint label, or terminal artifact was opened.

Status: **ready_without_paid_calls**.

This prospective addendum repairs only the obsolete downstream bindings in the
earlier current-HEAD readiness record. That record remains immutable at JSON
SHA-256 `6ea9ec149f7b62187cf9f966be9f1d6b5cea7ec1fb5bada198b75bbbc01c185f`
and Markdown SHA-256
`093acc3a79cdd08ad3818dc904daca4f41a85f78ad2ef36f867d70b33281eb93`.
Its paid wrapper, model, prompt, tasks, seeds, response contract, attempt limits,
budget, and scientific gates remain in force.

## V2 Handoff

The paid August 10 wrapper is unchanged at SHA-256
`adf0cede0c14e1ac96206461371f2f53f434f5b748327f9cf93ae0e7f521f9a5`.
The sole mechanics analysis handoff is now postprocess V2:

- base protocol `f7d19ef3a8a48478aa30d4b63541e0c680f74641de70ed3120b25f203d888f5e`;
- V2 amendment `0e0a443b033b4ea5dcbe426dbf1820de9baa029ff2d78794d87e329f52cda66f`;
- implementation `883707739185f91fc7d60fe12661896e3a62c690b406fc993e5ccbdeffd69ce0`;
- focused regression `077a6f556ca53cc640d4dcf6d6ae3abaf7632ca218eda15246dfe5f1570e72c9`.

After a wrapper-bound mechanics pass, V2 runs the frozen classical suite, path
mediation, and compute-matched audit in that order. It accepts an interrupted
compute checkpoint only after exact canonical reconstruction. A failure binds
only the validated earlier prefix and can never resume. A successful composite
already contains the mechanics compute audit, so a separate audit afterward is
forbidden. Development and confirmation retain their own stage-specific audit
and paper-wrapper sequence.

## Fresh Read-Only Check

The production preflight again returned interface
`bongard-openworld-luna-aug10-execute-4` and
`ready_without_paid_calls`. Wrapper, serving, mechanics, daily-ledger, and V2
postprocess paths are all absent. It verified four mechanics tasks, 56 images,
ten serving cases, and 8,821,758 serialized prompt bytes. The canonical
preflight SHA-256 was
`0da1cf74f05cd1237f966aeadc2d073d588460ecb68049b7683d47e56c95073a`.

The live catalog still exposes `openai/gpt-5.6-luna` at `$0.10/M` input and
`$0.60/M` output. Authenticated credits, usage, and balance were
`$245.000000000`, `$220.121013787`, and `$24.878986213`; the reported `$30`
top-up remains unposted and is not counted. The hard Europe/London daily cap is
`$5.00`, of which the finite mechanics chain can authorize at most `$2.00`.

This check made zero model calls, wrote no execution artifact, accessed no
candidate or scientific endpoint label, and cost `$0`. Focused postprocess
tests pass `28/28`, the new readiness contract passes `3/3`, the combined
focused set passes `11/11`, and the full Bongard family passes `239/239`.

## August 10 Command Order

Source `.env`, run the fresh same-day preflight, and invoke the frozen wrapper
once only if it remains ready. Bank any terminal result or failure. Then invoke
postprocess V2 exactly once on that terminal artifact. This addendum authorizes
no early call, rerun, repair, development stage, endpoint opening, or claim.
