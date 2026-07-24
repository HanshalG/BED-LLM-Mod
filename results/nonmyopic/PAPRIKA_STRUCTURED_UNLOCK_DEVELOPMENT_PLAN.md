# Paprika Structured-Hypothesis Unlock Development

Date: 2026-07-24

Status: preregistered development gate. This is not a policy comparison and does not
reuse Paprika's invalid resolution endpoint.

## Claim Under Test

The private Paprika solution is a structured cause-and-remedy hypothesis. The test asks
whether a diagnostic observation can make the LLM regenerate a differential that
semantically recovers this hidden solution, and whether different diagnostic questions
lead to meaningfully different recovery. This is the prerequisite for non-myopic value
over path-dependent LLM belief dynamics.

## Frozen Sample And Models

- Pinned Paprika customer-service data, train split.
- Twelve indices selected without replacement by NumPy seed `24287`:
  `92, 157, 199, 250, 398, 514, 556, 574, 584, 597, 602, 618`.
- Non-thinking `google/gemma-4-26b-a4b-it` generates eight initial hypotheses, four
  diagnostic-only questions, private-solution-conditioned customer replies, and eight
  refreshed hypotheses per realized branch.
- Non-thinking `openai/gpt-5.4-mini` judges semantic solution coverage only after all
  hypotheses have been generated.
- OpenRouter run cap: `$2.00`; projected cost reservation: `$0.75`.

The private solution is excluded from every initial-generation, question-generation,
refinement, and filtering prompt. It appears only in the customer simulator and final
measurement judge. No terminal-resolution judgment is used.

## Endpoint

For each initial and refreshed support, the judge scores the best hypothesis in `[0,1]`.
Coverage requires a score of at least `0.80`, with the same underlying cause and a
compatible remedy. Shared symptoms, generic checks, or plausible alternative fixes do
not count.

Primary mechanism quantities:

- initial solution omissions;
- initially omitted solutions recovered by at least one candidate;
- oracle best refreshed-match gain over the initial support;
- within-task spread across the four diagnostic candidates.

## Frozen Pass Gate

All conditions must hold:

1. all 12 tasks and 48 realized candidate branches complete;
2. at least 6/12 solutions are absent from initial supports;
3. at least 3 initially omitted solutions are recovered by some diagnostic candidate;
4. mean oracle best-match gain is at least `0.10`;
5. at least 4/12 tasks have candidate score spread at least `0.15`.

Failure closes this Paprika structured-unlock line before any target-blind ranker or
policy run. Passage authorizes development of a target-blind expected belief-quality
scorer on these inspected train tasks, followed by a separately frozen holdout.
