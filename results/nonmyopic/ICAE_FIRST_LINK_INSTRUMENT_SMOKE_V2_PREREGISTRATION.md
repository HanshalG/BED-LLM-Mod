# ICAE Model-Aware First-Link Instrument Smoke V2

Date: 2026-07-29

## Purpose

Prospectively rerun the complete ICAE first-link instrument after V1 failed
on a representation rule. V1 returned known unique trigger IDs outside
catalog order and stopped after call seven. Its realized refresh and hidden
endpoint were never opened, and V1 remains failed without normalization or
rescoring.

V2 changes only the trigger-match codec: `matched_ids` is an unordered unique
set. The parser validates every ID and canonicalizes the set to released
catalog order before the exact-response controller joins stored responses.
The response text, scientific gates, and exact ten-call budget are unchanged.

## Frozen Task And Models

Exclude all five mechanics tasks opened by earlier ICAE smokes. Select the
remaining task minimizing `SHA256("51000:<alias>")`:

- `realcode@023` (Ruby).

Models:

- GPT-5.4 non-thinking, seed `51100`: initial support, hypothetical positive
  answers, branch refreshes, and realized-history refresh;
- GPT-5.4 Mini non-thinking, seed `51200`: likelihood matrix,
  branch-retention scoring, set-valued trigger matching, and endpoint
  evaluation.

## Exact Ten Calls

1. initial 12-hypothesis/six-question support;
2. six hypothetical positive answers;
3. complete 12-by-6 semantic likelihood matrix;
4. first-question positive-branch refresh;
5. first-question fallback-branch refresh;
6. branch-retention score;
7. realized exact-controller semantic match;
8. realized-history support refresh;
9. hidden-requirement coverage endpoint;
10. exact repeat of the coverage endpoint.

No response repair, retry, reissue, task/model/seed substitution,
development data, or executable coding endpoint is allowed. Raw outputs are
checkpointed privately immediately after each accepted response and before
schema parsing.

## Gates

All V1 scientific gates remain frozen:

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
Failure closes this V2 instrument.

## Accounting

- projected cost: `$0.18`;
- hard cap: `$0.40`;
- OpenRouter only;
- no OatML, Slurm, SSH, or cluster use.
