# Bongard OpenWorld Path-Mediation Protocol

Frozen: 2026-08-09 (Europe/London), before any Bongard mechanics,
development, confirmation, candidate, or endpoint outcome was opened.

Status: **implemented and fail-closed; zero model calls and zero outcomes**.

This protocol adds a downstream mechanism report. It changes no Luna model,
request, prompt, response schema, task, seed, action, endpoint, policy, score,
science gate, result tier, call count, or budget.

## Question

The primary matched realized-updater contrast already asks whether, after the
same first query and realized answer, an answer-conditioned intermediate LLM
belief chooses a better second query than a same-seed history-blind belief
under a common terminal updater. Aggregate changed-action and endpoint gates
test that contrast, but they do not expose the full intermediate chain.

This report makes three links visible for every task:

1. **belief:** how much the answer-conditioned and history-blind intermediate
   supports differ in rules, predictive probabilities, and second-query score
   ordering;
2. **action:** whether the two supports select different second queries and
   whether each robustly prefers its own action; and
3. **endpoint:** the already registered dynamic-minus-matched endpoint Brier
   and log-loss differences under the common terminal updater.

## Exact Replay Boundary

`scripts/bongard_openworld_path_mediation.py` first invokes the existing
stage authorizer. Mechanics requires the complete Aug10 wrapper and mechanics
artifact. Development and confirmation require their exact four independently
replayed blocks and combined result. Only after that authorization does the
instrument read hash-verified raw first-stage responses and the endpoint-bearing
stage result.

It reconstructs every dynamic and history-blind semantic belief with the
original strict parser. For the realized dynamic first query and answer, it
analytically conditions the history-blind weights, recomputes both seven-way
endpoint-PIG score maps, and requires exact equality with the stored maps and
argmax actions. A changed score, action, raw-response hash, task identity, or
authorization fails closed.

## Frozen Metrics

Per task, report:

- canonical-rule Jaccard overlap;
- mean absolute predictive-probability difference over remaining candidates
  and over the two sealed endpoints;
- Spearman correlation between the two second-query score maps;
- each support's score advantage for its own selected action;
- changed and robustly changed second-action indicators;
- whether both supports robustly prefer their own different actions; and
- dynamic-minus-history-blind-matched endpoint Brier and log loss.

Pool counts and means across all tasks. Report Spearman correlations between
endpoint predictive shift or dynamic action gap and realized Brier benefit.
Report fixed 20,000-draw paired task-bootstrap summaries for all tasks, changed
actions, and robust changed actions. An empty selected subset is explicitly
`not_estimable_no_selected_tasks`.

The changed-action subsets and correlations are descriptive. They do not alter
the registered all-task primary comparison, any gate, confirmation
authorization, claim tier, or classical scope. Correlation is not itself a
causal mediation estimate; causal interpretation remains limited to the
prospectively matched policy contrast.

## Stage Use

Run once after each independently verified stage result:

```bash
/opt/anaconda3/bin/python scripts/bongard_openworld_path_mediation.py \
  --stage mechanics \
  --result-path <mechanics-result> \
  --wrapper-result <aug10-wrapper-result> \
  --output-path <mechanics-mediation.json>
```

For development or confirmation, replace `--wrapper-result` with all four
ordered `--block-result` arguments. A failed or incomplete report is banked;
it permits no paid rerun or endpoint reissue.

## Bindings

- implementation SHA-256:
  `1aef1c9eb90757bd31fec4beb077ddf79965e1a42b2715b4f7a6788e57e8b912`;
- tests SHA-256:
  `8d20f4412f24a87776976f9274e4a7b73547d39768b6b0fa75eaf91ff2868c41`.

Focused tests pass `4/4`; the mediation plus development/confirmation replay
set passes `17/17`; the complete Bongard suite passes `213/213`. This protocol
authorizes no paid call, endpoint opening, rerun, claim, or result-dependent
model or policy change.
