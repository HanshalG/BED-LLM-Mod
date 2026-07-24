# MuSiQue Answer-Belief Bridge Gate Preregistration

Date: 2026-07-24
Seed: `24327`
Status: preregistered before any response.

## Question

Does a two-document bridge create non-myopic value in the LLM's own
path-dependent belief over final answers?

This is not a repair of the closed chain-support V3 score. V3 treated ordered
document chains as the latent support, so its immediate entropy score rewarded a
root merely for receiving more generated chains. This gate instead makes the
final answer the uncertain variable. The LLM regenerates an eight-answer
probability distribution after each opened document.

## Frozen design

- Dataset: hash-verified MuSiQue answerable dev v1.0.
- Fresh target-blind draw: seed `24327`, excluding every V1--V3 row.
- Serving smoke: first two frozen rows.
- Opportunity screen: next six frozen rows.
- Reserve only: final six frozen rows.
- Generator: `openai/gpt-5.4`, no reasoning, temperature 0.
- Equivalence judge: `openai/gpt-5.4-mini`, no reasoning, temperature 0.
- Gold answers and supporting-document labels are hidden from the generator.
- The judge sees the gold answer only after every belief is generated.
- Each row's 20 paragraphs receive a deterministic random permutation of opaque
  document IDs.
- Initial state: one eight-answer probability belief and six first-document
  proposals.
- Each first branch: regenerate the eight-answer belief from the opened document
  and propose four unopened second documents.
- Each ordered pair: regenerate the belief from both documents in order.
- The first generated pair is replayed once with the identical prompt.
- Candidate order is shuffled after generation before any deterministic tie
  break.
- Private raw responses are checkpointed after every phase.

The complete tree costs exactly 33 physical requests per row: one initial call,
six first branches, 24 second branches, one replay, and one equivalence call.
The smoke therefore requires 66 calls and the screen 198.

## Frozen opportunity analysis

For each state, truth probability is the sum of probabilities assigned to
judge-equivalent answer strings, floored at `1e-4` only when none match.

- The realized myopic first action maximizes truth probability after one
  document.
- Its continuation maximizes final truth probability among that first action's
  four proposed second documents.
- The two-step oracle maximizes final truth probability over all 24 ordered
  pairs.
- The gold pair is the dataset's ordered decomposition pair under the shuffled
  document IDs.

The six-row opportunity conjunction requires:

1. all 198 calls complete with zero reasoning tokens;
2. mean initial truth probability at most `.25`;
3. the gold first document appears for at least 4/6 rows;
4. the gold second document is proposed after that root for at least 4/6;
5. the gold ordered pair is the two-step oracle on at least 2/6;
6. the oracle first action differs from realized myopic on at least 2/6;
7. the gold root differs from realized myopic on at least 2/6;
8. pair gain over the best one-step truth probability is at least `.10` on
   at least 3/6 and has mean at least `.08`;
9. oracle gain over the realized-myopic continuation is at least `.10` on at
   least 2/6 and has mean at least `.05`;
10. replay truth-probability gap has mean at most `.10` and maximum at most
    `.25`.

Failure of any gate closes this exact answer-belief opportunity design. It does
not authorize a planner, reserve-row evaluation, threshold change, row subset,
candidate-count change, or prompt repair.

## Budget

- Smoke projected/hard cap: `$0.50` / `$1.50`.
- Opportunity projected/hard cap: `$1.50` / `$4.00`.
- Both are inside the project ceiling while preserving the protected `$25`
  OpenRouter reserve through Monday.
