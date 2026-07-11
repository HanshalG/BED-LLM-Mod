# Path E Stop-And-Discuss Decision Memo

Date: 2026-07-11

## Decision Point

The terminal-valid 10-task Paprika pilot passed its frozen arbitration gate:
naive-primary arbitration resolved 6/10 tasks versus 4/10 for thinking naive, with
6 wins, 0 losses, 4 ties, and a paired censored-turn delta of -1.2 turns (bootstrap
95% CI [-2.2, -0.4]). The result authorizes discussion of scaling; it does not itself
authorize a launch.

The original Path E lookahead story is closed for this cycle. Full two-step failed its
gate and was too expensive, and faithful one-step EIG also failed against naive. The
surviving method claim is narrower and different: **belief-guided arbitration over
native LLM proposals**.

## Important Attribution Gap

The pilot's thinking-naive baseline generates one action with its native prompt. The
arbitration arm generates three ordered actions in one prompt, treats candidate 0 as
the native default, and may override it using EIG. Consequently, the arm-level pilot
gain can combine two effects:

1. asking the model to produce an ordered three-action set; and
2. selecting among that set with EIG.

The 12 observed overrides, including four that immediately selected the exact remedy,
support the proposed mechanism but do not isolate it causally. A publishable scaled
claim needs a prompt-matched control that executes candidate 0 from the same three-action
generation prompt without belief scoring.

## Recommended Path: Held-Out Paprika Headline

Pre-register and run 50 held-out official Paprika eval tasks (offsets 10--59), five
turns, paired task seeds, terminal-faithfulness-repaired endpoint, with these arms:

| Arm | Purpose |
|---|---|
| `NaivePrimaryArbitration` | Proposed belief-guided selector. |
| Prompt-matched candidate 0 | Causal control for the EIG override; same three proposals, always execute index 0. |
| Thinking naive | Existing adversarial native baseline. |
| Non-thinking naive | Contextual low-cost baseline; retain for continuity. |

Primary comparisons should be arbitration versus prompt-matched candidate 0 and versus
thinking naive. Report paired resolution@5, full resolution curves, paired censored-turn
differences with bootstrap CIs, exact paired binary tests, override rate, remedy-hit rate,
answer-set coverage, endpoint-faithfulness metrics, forced exits, requests, tokens, and
cost. The first 10 pilot tasks must not enter the headline estimate.

The pilot effect is large enough that 50 tasks is a reasonable first powered headline
sample, while 100 tasks should be reserved for a later confirmation if the 50-task
interval remains ambiguous. The official release contains 200 eval tasks, so this split
does not exhaust the benchmark.

## Cost And Throughput

Observed accepted-run cost per pilot task was approximately:

- arbitration: $0.02918;
- thinking naive: $0.01158;
- non-thinking naive: $0.00100.

Treating the prompt-matched candidate-0 control as roughly thinking-naive cost, the
four-arm 50-task nominal projection is about $2.67. A conservative 2x envelope is
$5.34. Current cumulative OpenRouter spend is $10.38197 of the authorized $20, leaving
$9.61803. This fits the account but would reduce the previously requested $6 MediQ
reserve, so the backend choice and reserve policy require explicit approval.

For OpenRouter, use isolated task shards to avoid head-of-line blocking from long
thinking generations. The user permits aggregate concurrency up to 256; target about
240--250 only while 429/timeout rates remain clean. Increasing concurrency does not
shorten individual thinking calls.

## Alternatives

1. **Recommended: approve the 50-task prompt-matched headline.** This is the only path
   that can turn the pilot into a defensible method claim. Pre-registration and the new
   control must be committed before launch.
2. **Freeze Path E as pilot evidence and write a single-environment exploratory paper.**
   This is cheaper but too weak for the immutable definition of done and should avoid a
   causal EIG-selection claim.
3. **Stop Path E and retain the already validated Path B negative-result draft.** This
   preserves the existing workshop package but abandons the external-benchmark goal.

No MediQ integration, selective lookahead revival, or additional rescue variant should
start before this decision is made.
