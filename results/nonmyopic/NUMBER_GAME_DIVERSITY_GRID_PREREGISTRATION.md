# Number Game Diversity-Grid Planner Preregistration

Date frozen: 2026-07-29, before any diversity-grid model response.

## Motivation And Scope

The powered Gemini planner result is positive, and a 31-tree Qwen planner
replay independently reproduces the primary posterior-predictive Brier
mechanism. DeepSeek and Grok exact-10 screens under the original flat
24-expression interface failed because history-conditioned supports collapsed
into equivalent or contradictory rules. GLM failed strict transport.

This is a disclosed post-hoc interface-development successor, not a repair or
relabeling of any closed run. It tests whether explicit semantic diversity can
make a second planner family transportable while retaining open-ended
executable hypothesis generation.

The new response has four named cells with exactly six rules each:

1. periodic or digit structure;
2. ordered thresholds or ranges;
3. transformed prime, square, or power-of-two structure;
4. Boolean compositions of structurally different ideas.

The LLM still writes each executable expression. A deterministic parser
executes it on all 101 integers, rejects wrong-family rules, explicit equality
or inequality exceptions for observed numbers, contradictions, near-constant
extensions, and duplicate behavior.

## Zero-Call Feasibility Gate

Before model use, construct a deterministic audit bank for each requested
family. The bank is not shown to the model and is not used by the planner. It
only proves that the prompt is satisfiable on the ten frozen serving histories:

- two empty histories;
- `10` and `42`, each with either one label;
- `YES(10), NO(20)` and its reverse;
- `YES(42), YES(75)` and `NO(42), NO(75)`.

Every history must admit at least 16 unique valid extensions in every family
and at least 4,000 across their union. Failure closes the interface before API
use.

## Exact-10 Serving Qualification

Conditional on the zero-call gate, use:

- planner: `deepseek/deepseek-v4-pro`;
- non-thinking, temperature `0.7`, requested seed `48000`;
- exactly one response for each frozen history, ten accepted requests total;
- no content retry, parser repair, response normalization, seed replacement,
  threshold change, or model swap;
- hard cost cap `$0.15`.

All transport gates are conjunctive: exactly ten HTTP and accepted requests,
all strict schemas parse, zero retry, provider error, reasoning token, or
forced exit.

Support gates are also conjunctive:

- both initial supports have at least 20 valid unique rules and at least four
  from every family;
- every conditioned support has at least 12 valid unique rules, at least one
  from every family, and at least two valid compositional rules;
- every support has a remaining-query maximum EIG of at least `0.50` nats.

The smoke is support qualification only and cannot report policy efficacy.
Failure closes this exact diversity-grid model, prompt, seed, histories, and
thresholds.

## Conditional Fresh Formal Study

Only a complete smoke pass authorizes one fresh 32-tree study:

- DeepSeek diversity-grid planner with fresh seeds beginning at `49000`;
- independent GPT-5.4 Mini targets, eight validation supports, and sixteen
  endpoint supports per tree with separate fresh seeds;
- retained consistent parent particles at both refresh depths;
- shared generated trees and independent endpoint draws for cross-fitted depth
  three, equally cross-fitted depth two, myopic EIG, fixed-support depth three,
  PTS, and seeded-random controls;
- exact 2,336 accepted responses and a `$7.50` cap.

The primary proper-score conjunction is frozen before the smoke:

1. all 32 trees and mechanical support thresholds complete;
2. depth-three and depth-two roots differ on at least 12 trees;
3. depth three lowers posterior-predictive Brier by at least 1% versus equally
   cross-fitted depth two, wins at least 12 trees, and its whole-tree 95%
   bootstrap interval is below zero;
4. novel-target Brier does not regress;
5. depth three beats myopic and fixed-support depth three by at least 5%, and
   PTS by at least 3%, with all whole-tree intervals below zero;
6. depth-three source-risk Spearman is at least `0.70` and exceeds depth two by
   at least `0.15`;
7. exact request accounting, zero reasoning/forced exits, bounded retries, and
   budget all pass.

Hamming and exact-extension coverage are reported as secondary diagnostics,
not unrelated conjunction gates for the Brier claim. No partial-tree formal
aggregate, resume after semantic failure, threshold repair, or additional
planner-family screen is allowed.
