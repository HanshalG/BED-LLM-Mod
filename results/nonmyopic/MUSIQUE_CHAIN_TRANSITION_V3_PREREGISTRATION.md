# MuSiQue Indexed Chain-Transition V3 Preregistration

Date: 2026-07-24
Seed: `24314`
Status at registration: no V3 model response has been requested.

## Rationale

The V2 scientific gate never started. Its 12 initial support calls completed, but one
row proposed `We Were Soldiers` as a first title even though that string was a question
entity and not one of the 20 available document titles. The run stopped before any
document opening, refresh, branch metric, or endpoint. It cost `$0.00334004`, used zero
reasoning tokens, and has raw SHA-256
`76cb79eaef3fcd87abb10622e2ef4352252cbec2b1aeb4f3bc6414e941c574b9`.
V2 is closed and none of its generated supports are reused.

V3 makes one interface change: the same 20 titles are displayed beside opaque IDs
`d01` through `d20`, and the model emits IDs rather than copying titles. The parser
maps valid IDs back to titles exactly. It does not repair, coerce, or substitute an
invalid ID. This changes serialization and action legality only.

## Fresh split

All 14 V1/V2 rows are excluded before sampling. The remaining eligibility rule,
official MuSiQue artifact and SHA-256, model, temperature, chain count, six-action
structure, deterministic opened-document observation, target, metrics, gates, request
counts, and budget are unchanged from
`MUSIQUE_CHAIN_TRANSITION_GATE_PREREGISTRATION.md`.

Frozen smoke IDs:

1. `2hop__78756_198548`
2. `2hop__329676_119915`

Frozen formal IDs:

1. `2hop__267938_92763`
2. `2hop__58168_1783`
3. `2hop__85931_108632`
4. `2hop__747306_72813`
5. `2hop__133102_417697`
6. `2hop__462179_643013`
7. `2hop__785711_63853`
8. `2hop__96414_47902`
9. `2hop__6736_6733`
10. `2hop__739909_807845`
11. `2hop__132472_684936`
12. `2hop__499003_853511`

V3 smoke remains exactly 14 accepted calls and formal remains exactly 84. A smoke
failure closes the indexed interface. A formal serving failure closes V3 with no
recovery. A completed formal gate is judged by the original frozen scientific
conjunction; no threshold or metric changes are permitted.

## Smoke implementation recovery

The first V3 smoke execution stopped after the two initial responses and before any
opened-document branch. The model emitted distinct IDs `d06`, `d18`, and `d20`, but
all three source paragraphs share the display title `Smokey Bear`. A leftover V2
parser check compared mapped title strings rather than the preregistered document IDs
and incorrectly rejected these valid distinct documents. Cost was `$0.00064286`, zero
reasoning tokens, and the initial checkpoint SHA-256 is
`a71110359dab7ffa729bdd3c365fbc5db676baa5aaf6ccf9d35ef9dc292ad15a`.

Before any branch response, freeze a hash-locked implementation recovery:

1. change only the same-document check from mapped-title equality to document-ID
   equality, as the V3 specification requires;
2. reuse exactly the two checkpointed initial responses;
3. keep the same run ID so ledger usage includes the original two calls;
4. issue only the 12 missing branch refreshes; and
5. require the original exact combined count of 14 and all unchanged gates.

No response is regenerated, repaired, or substituted. Any recovery failure closes V3.
