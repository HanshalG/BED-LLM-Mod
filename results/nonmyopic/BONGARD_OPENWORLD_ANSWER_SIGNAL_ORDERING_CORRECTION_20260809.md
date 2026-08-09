# Bongard Answer-Signal Ordering Correction

Frozen: 2026-08-09 Europe/London, before any August 10 Bongard response,
candidate label, endpoint label, or terminal artifact was opened.

Status: **prospective orchestration correction; changes no paid request or
scientific threshold**.

## Gap

The answer-signal amendment requires a null or malformed audit to block
Development64 and every endpoint-analysis descendant. Final-handoff interface
V2 ran the answer audit before postprocessing, but it still invoked the
endpoint-opening postprocessor when the audit returned `gated_null`. It marked
the final record failed only after those analyses had already opened.

That ordering violates the intended information boundary even though it cannot
change a paid response or scientific score.

## Corrected Order

Final-handoff interface V3 must execute:

1. the frozen paid wrapper;
2. the zero-call answer-signal audit when the wrapper contains an exact
   mechanics pass;
3. the postprocessor only when no answer audit applies or the answer audit
   returns exact `answer_signal_valid`;
4. the random-strategy audit only after an exact passing answer audit and a
   complete postprocessor result.

If the answer audit returns `gated_null`, V3 must bank a terminal
`failed_closed` handoff containing the paid terminal and answer-audit
components only. It must record `postprocess_opened=false`,
`downstream_analyses_opened=false`, and
`random_strategy_audit_opened=false`. Replay must reject any answer-null record
that contains a postprocess or random component.

If answer-audit construction raises because its hash-bound input is malformed,
the exception must propagate before postprocessing. The paid wrapper remains
banked, so a retry can only replay the zero-call boundary and cannot repeat a
paid component.

Paid failures and mechanics nulls still enter the existing disposition-only
postprocessor because no passing mechanics result exists and that path opens no
candidate or endpoint labels.

This correction changes no model, prompt, response schema, task, image, split,
seed, action, policy, endpoint, likelihood, gate threshold, request count,
retry rule, cost cap, paid date, claim tier, or headline rule.
