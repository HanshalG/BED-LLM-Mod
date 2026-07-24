# Animals CA-BED Aligned V13 Result

Date: 2026-07-24

## Outcome

The final Animals interface failed closed during serving smoke, before the
first complete tree and before any endpoint.

- Run ID: `animals-cabed-aligned-v13-smoke-20260724T211233Z`
- Requests: 17
- Reasoning tokens: 0
- Cost: $0.00296441
- Formal targets used: 0

The failed branch requested five follow-up candidates. Gemma first returned
only `Does it live in the water?`. The existing bounded top-up then requested
exactly four new questions while displaying that candidate, but Gemma repeated
only `Does it live in the water?`. One valid candidate remained versus the
required three.

No menu was regenerated or repaired. Per the V13 preregistration there is no
V14. The Animals CA-BED OpenRouter line is closed.

## Interpretation

Target-blind semantic batching was valid and inexpensive, and aligned table
observations remove likelihood/answer mismatch. The remaining free-form
shrinking-menu generator is not reliable enough to support a sealed 24-state
comparison with Gemma 4 26B through this interface.
