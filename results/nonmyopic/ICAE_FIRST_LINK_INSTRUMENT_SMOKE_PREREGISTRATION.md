# ICAE Model-Aware First-Link Instrument Smoke

Date: 2026-07-29

## Purpose

Test every load-bearing component required for non-myopic planning over the
LLM's own path-dependent belief regeneration before a paired policy
measurement.

The exact-response controller has already passed serving. This smoke adds:

1. LLM-generated semantic hypothesis particles;
2. LLM-generated question likelihoods over those particles;
3. positive and fallback counterfactual observations;
4. history-conditioned support regeneration;
5. model-aware scoring of which initial hypotheses survive each branch;
6. realized exact-controller execution; and
7. an independent hidden-requirement coverage endpoint.

## Frozen Task And Models

Exclude the four mechanics tasks opened by prior ICAE smokes. Select the
remaining task minimizing `SHA256("50700:<alias>")`:

- `realcode@259` (Java).

Models:

- GPT-5.4 non-thinking, seed `50800`: initial support, six hypothetical
  positive answers, positive/fallback branch refreshes for the first question,
  and realized-history refresh;
- GPT-5.4 Mini non-thinking, seed `50900`: 12-by-6 semantic likelihood
  matrix, branch-retention scoring, exact-controller trigger matching, and two
  independent endpoint evaluations.

## Exact Ten Calls

1. initial 12-hypothesis/six-question support;
2. six hypothetical positive answers;
3. complete semantic likelihood matrix;
4. first-question positive-branch refresh;
5. first-question fallback-branch refresh;
6. branch-retention score;
7. realized exact-controller semantic match;
8. realized-history support refresh;
9. hidden-requirement coverage endpoint;
10. exact repeat of the coverage endpoint.

No response repair, retry, reissue, normalization, task/model/seed
substitution, development data, or executable coding endpoint is allowed.

## Gates

All must pass:

- exact 10 accepted requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all schemas and cardinalities parse exactly;
- at least three distinct semantic likelihood values;
- positive and fallback histories each change at least four questions and
  produce different hypothesis supports;
- branch-retention counts differ by at least one of 12 initial hypotheses;
- the realized first question matches at least one released trigger;
- realized history changes at least four questions;
- duplicate hidden-requirement coverage judgments are exact;
- coverage is unsaturated in `[0.20, 0.90]`; and
- cost is at most `$0.40`.

Passage authorizes only a separately frozen paired mechanics measurement.
Failure closes this exact first-link instrument.

## Accounting

- projected cost: `$0.18`;
- hard cap: `$0.40`;
- OpenRouter only;
- no OatML, Slurm, SSH, or cluster use.
