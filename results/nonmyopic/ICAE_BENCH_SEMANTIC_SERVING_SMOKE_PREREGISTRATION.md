# ICAE-Bench Exact-10 Semantic Serving Smoke

Date: 2026-07-29

## Authorization

This smoke is authorized only by the hash-bound zero-call opportunity audit:

- manifest SHA-256:
  `47ab7f2fcf985433389f53873fa7fe8ccbebd88dd8390d654de93083bad84f5f`;
- opportunity audit SHA-256:
  `f63a313adb8bc3bd1090fe4542d1b38b4c2e63abca30c559447a70a6e6348552`.

No development, confirmation, retained, coding-agent, or executable endpoint
data may be opened.

## Frozen Interface

Select two of the 12 mechanics aliases by
`SHA256("50100:<alias>")`, yielding:

- `realcode@044` (JavaScript);
- `realcode@276` (PHP).

Models:

- support/question planner: `openai/gpt-5.4`, seed `50200`;
- official-style Oracle: `google/gemini-3.1-flash-lite`, seed `50300`.

Both are non-thinking at temperature zero. No retry, response repair, parser
normalization, reissue, model swap, seed change, or task substitution is
allowed.

The exact ten requests are:

1. two initial 12-hypothesis/six-question generations from fuzzy PRDs;
2. two Oracle replies to each selected first question;
3. two fresh-session exact replays of those replies;
4. two Oracle replies to one frozen unrelated generic question; and
5. two history-conditioned support/question regenerations after the real
   first answer.

The first question is selected by response order, not hidden data.

## Gates

All must pass:

- exactly 10 accepted requests and 10 HTTP attempts;
- zero transport/provider retries, reasoning tokens, and forced exits;
- all initial/followup supports strictly parse with 12 unique hypotheses and
  six unique questions;
- both first questions trigger at least one hidden requirement and do not
  fall back;
- both fresh-session Oracle replies replay exactly;
- both unrelated questions return the task's exact frozen fallback with zero
  triggers;
- at least four of six followup questions differ from the initial questions
  on each task;
- each followup support uses at least one content token introduced by the
  real answer and absent from the fuzzy PRD/initial question support; and
- reported cost is at most `$0.25`.

Passage authorizes only a separately frozen mechanics first-link experiment.
It does not establish BED efficacy. Failure closes this exact interface.

## Budget

- projected cost: `$0.10`;
- hard run cap: `$0.25`;
- OpenRouter only;
- no OatML, Slurm, SSH, or cluster use.
