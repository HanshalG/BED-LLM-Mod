# InfoQuest Support-Causal-Link Preregistration

Frozen after the zero-cost cached-trajectory V2 opportunity pass and before any
new OpenRouter response.

## Question

On disclosed InfoQuest mechanics cases, does a query-induced regenerated
support that better covers the hidden context lead to more next-turn checklist
discovery than immediate progress, a fixed support continuation, an
answer-shuffled support score, and random root choice?

This is the first causal-link mechanics gate. It does not define a deployable
ex-ante planner and cannot establish policy efficacy. The support judge sees
the hidden context only to measure whether the LLM's regenerated belief
contains the truth.

## Frozen Data

- disclosed IDs: `{0, 1, 4}`;
- two official hidden settings per ID: six task/world fixtures;
- exact pinned `settings.jsonl` and official Falcon mt30 run-0 hashes;
- private six-world fixture SHA-256
  `63248345976e884b9fac127033415fe68bacb3a0e32a32e498285851719251cc`;
- no V1/V2 opportunity, development, or holdout record;
- uniform five-root comparison within each fixture.

The target-blind policy sees only each ambiguous seed message. Hidden settings,
personas, constraints, solutions, and checklist items are available only to
the user simulator and independent endpoint judges.

## Frozen Model Roles

All calls are temperature zero and non-reasoning.

| Role | Model |
| --- | --- |
| Initial support, regenerated support, dynamic/fixed follow-up | `openai/gpt-5.4` |
| Official-prompt hidden-user simulator | `google/gemini-2.5-flash` |
| Hidden-context support judge | `google/gemma-4-26b-a4b-it` |
| Checklist endpoint judge | `openai/gpt-5.4-mini` |

Gemini 2.5 Flash is frozen because the released benchmark's Gemini 2.0 Flash
model is no longer listed by OpenRouter. No model fallback or swap is allowed
after responses begin.

## Frozen Procedure

For each of three ambiguous seeds, GPT-5.4 generates:

- eight distinct concrete latent-context hypotheses;
- five distinct atomic clarification roots.

For each of 30 root/world cells:

1. the hidden-user simulator answers the root under the exact released system
   prompt;
2. GPT-5.4 regenerates eight hypotheses from the full seed/root/answer history
   and chooses a dynamic follow-up;
3. a separate GPT-5.4 call keeps the original eight hypotheses fixed and
   chooses a fixed-support follow-up;
4. the same hidden-user simulator answers both follow-ups from their matched
   histories.

For each task/world fixture:

- Gemma4-26B scores best joint hidden-context coverage for the initial support
  and all five refreshed supports on `0..100`, with presence threshold `70`;
- GPT-5.4 Mini marks the five official checklist needs after the root, dynamic
  continuation, and fixed continuation.

Support size is always eight. No hypothesis is silently deduplicated,
replaced, repaired, or dropped. Questions must satisfy the frozen atomic
parser. A single optional outer Markdown JSON/code fence is transport-only and
is removed before exact parsing.

## Controls

- **Immediate/myopic:** checklist count after the root.
- **Fixed support:** follow-up chosen from the unchanged initial eight
  hypotheses.
- **Answer-shuffled:** refreshed truth scores permuted within each task/world by
  the frozen zero-index permutation `[1, 3, 4, 2, 0]`.
- **Random:** mean two-turn dynamic checklist count over all five roots.

Dynamic and myopic root selection use the same realized dynamic continuation
endpoint, isolating root ranking. Dynamic versus fixed continuation is paired
within every root/world cell.

## Serving Gate

Before mechanics, run exactly ten synthetic calls:

- two initial-support calls;
- two root-simulator calls;
- two refresh calls;
- one fixed-follow-up call;
- one follow-up-simulator call;
- one support-judge call;
- one checklist-judge call.

All parsers must pass with exact 10 physical requests/HTTP attempts, zero
retry, reasoning, and forced exit, and cost at most `$0.15`. Failure closes the
mechanics run without repair or reissue.

## Mechanics Accounting

The one mechanics run uses exactly:

- 3 initial-support calls;
- 30 root-simulator calls;
- 30 refresh calls;
- 30 fixed-follow-up calls;
- 60 paired follow-up-simulator calls;
- 6 support-judge calls;
- 6 checklist-judge calls;

for **165 physical requests and HTTP attempts**. Cost cap: `$1.00`. Any parse,
request-count, retry, reasoning, forced-exit, or cap failure stops before
scientific interpretation.

## Conjunctive Scientific Gates

1. at least one of six initial supports omits its hidden context;
2. at least two query/world cells cause truth entry at score `>=70`;
3. at least four fixtures have refreshed-support score range `>=10`;
4. mean within-fixture Spearman of refreshed truth score versus dynamic
   two-turn checklist count is at least `.20`;
5. pooled Spearman is at least `.25`;
6. mean within-fixture refreshed-score Spearman exceeds immediate-count
   Spearman by at least `.10`;
7. it exceeds answer-shuffled Spearman by at least `.15`;
8. refreshed-score-selected roots beat immediate-selected roots by at least
   `.15` checklist items on average and have more wins than losses;
9. refreshed-score-selected roots beat random expected roots by at least
   `.15`;
10. dynamic continuation beats fixed-support continuation by at least `.10`
    checklist items over all 30 paired roots and has more wins than losses.

All gates are conjunctive. A pass authorizes only a separately preregistered
ex-ante ranking-fidelity gate that must estimate support quality without seeing
the hidden truth. A failure localizes whether truth omission/recovery, support
ranking, root selection, or dynamic continuation broke and stops this exact
route without model, threshold, parser, seed, or disclosed-case repair.

OpenRouter maximum new spend before Monday remains `$4`; this line can consume
at most `$1.15` including serving. OatML jobs: `0`.
