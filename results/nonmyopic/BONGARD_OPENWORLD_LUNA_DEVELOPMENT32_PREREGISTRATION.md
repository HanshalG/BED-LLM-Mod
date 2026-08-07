# Bongard-OpenWorld Luna Development-32 Preregistration

Date frozen: 2026-08-06, before any Bongard model request or development
endpoint access.

Before any model request, the shuffled control and action-margin diagnostic
were corrected by
`BONGARD_OPENWORLD_LUNA_SHUFFLED_CONTROL_AMENDMENT.md`. Development interface
`-2` is invalid; the matched history-blind control then advanced the interface
to `-4`. Before any response, terminal common random numbers were frozen in
`BONGARD_OPENWORLD_LUNA_TERMINAL_CRN_AMENDMENT.md`. The terminal task-batch
rule in `BONGARD_OPENWORLD_LUNA_TERMINAL_BATCH_AMENDMENT.md` advances the
current interface to `-6`.
The subsequently frozen
`BONGARD_OPENWORLD_LUNA_CONTRASTIVE_PROMPT_AMENDMENT.md` makes the benchmark's
positive-present/negative-absent rule semantics explicit before any model
response; all policies, tasks, endpoints, seeds, and gates remain unchanged.
The later
`BONGARD_OPENWORLD_LUNA_PATH_DEPENDENT_CLAIM_AMENDMENT.md` advances the
development result interface to `-7` and strengthens only the claim boundary:
the strongest tier must also beat fixed-support depth two.
The later simulated-branch obedience amendment advances the interface to `-8`:
each endpoint-blind block requires class-conditional Brier below `0.25` for the
newly supplied positive and negative simulated branch labels. Calls, policies,
and scientific gates are unchanged.

## Claim And Boundary

This is a prospective development experiment for the LLM-native claim:
path-dependent two-step BED can improve prediction when the LLM itself
generates the semantic hypotheses and visual likelihoods after each possible
history.

The 32-task development partition may open only after the frozen exact-10
serving smoke and four-task full mechanics tree both pass every gate. This
experiment can authorize a later confirmation preregistration. It cannot
authorize confirmation execution, establish a confirmatory result, or open any
of the 64 confirmation or 199 sealed-test tasks.

## Frozen Model And Belief Process

- Model: `openai/gpt-5.6-luna`, non-reasoning, temperature zero.
- Each belief call receives all 14 images under opaque IDs and only the labels
  in that history.
- Each response contains exactly ten distinct free-form visual rules, integer
  history-conditioned belief weights, and 14 calibrated positive-class
  probabilities per rule.
- The model weights are used directly at the history that generated them;
  predictive probabilities, entropy, hypothetical future updates, and EIG are
  analytical Bernoulli calculations over that response. The observed history
  is not multiplied into the model weights a second time.
- All images are visible under opaque IDs, but candidate and endpoint roles are
  evaluator-private and their unobserved labels are absent from every model
  request. Source UID, concept, caption, paths, positions, and label-bearing
  filenames remain hidden.

The LLM therefore owns both the open-ended hypothesis support and visual
likelihoods. Classical code only performs exact inference and planning over
the LLM's emitted belief.

## Staged Endpoint-Blind Execution

Sort the 32 frozen development tasks by opaque task ID and split them without
replacement into four equal blocks:

| Block | Offset | Tasks | Earliest London date | Model seed |
|---|---:|---:|---|---:|
| A | 0 | 8 | 2026-08-11 | 2026081101 |
| B | 8 | 8 | 2026-08-12 | 2026081201 |
| C | 16 | 8 | 2026-08-13 | 2026081301 |
| D | 24 | 8 | 2026-08-14 | 2026081401 |

Each block runs under a fresh account-wide `$5.00` Europe/London ledger and a
`$4.75` block cap. Unspent allowance does not roll over. Concurrency is 24.
The observed cost per request from the passed mechanics result is projected to
the maximum block tree with a 1.5 safety multiplier before authorization.
Every block must verify the same pre-call protocol manifest, which binds all 32
opaque task IDs and source-row hashes, block assignment, dates, model seeds,
request bounds, schemas, and implementation/document hashes. Any between-block
code or protocol change invalidates later execution.

For every task in a block, generate one root, all 16 answer-conditioned
candidate/outcome branches, and 16 paired history-blind branches. This is 33
first-stage requests per task. For every one of the eight
possible first actions, release its actual candidate label, select the best
second action under that regenerated branch, and generate the distinct final
support. Also include any distinct fixed-depth-two or random-policy final
history. Reciprocal action orders may share one final history, so there are
4--10 final calls per task and at most 344 total requests per block.

All distinct terminal histories within a task use one common requested model
seed, while different tasks use different seeds. When dynamic and
history-blind select different final histories, those requests are ordered
adjacently with dynamic first. This reduces avoidable terminal model-seed
variance without adding calls or changing the belief process.

Complete task-level terminal groups are greedily packed into explicit
dispatch batches of at most 24 requests, and no task may cross a batch
boundary. The batch manifest is persisted and replay-gated. Thus a distinct
dynamic/history-blind pair shares both its requested seed and its actual
adapter dispatch invocation.

The all-first-action continuations make ranking fidelity observable: within
each task, compare the frozen dynamic, myopic, fixed, and shuffled root score
rankings against negative realized endpoint Brier for the corresponding
branch-greedy continuation. This tests whether first-link scores rank actions
that actually induce useful final beliefs.

Candidate labels are released only after all root scores are frozen. Each
all-first-action continuation receives only its own first label and selected
second label; each policy receives only labels on its executed path. A block is
represented in memory with endpoint labels removed from the task mapping. The
block result contains no endpoint metrics and has scientific status
`sealed_until_all_blocks_complete`.

Requests are issued in deterministic chunks of at most 24. Every returned
chunk is strict-parsed and checkpointed with its exact case IDs before the next
chunk starts. A partial artifact is fail-closed and may not be rerun in place;
accepted requests are therefore never silently regenerated.

Blocks B, C, and D are mandatory after preceding blocks pass transport and
mechanics checks. No endpoint outcome exists at block time, so a block cannot
be stopped, continued, or altered based on scientific performance. Any partial
or failed block closes that artifact; it is never silently resumed or rerun.

## Frozen Paired Policies

All policies share exactly the same root, branch, and deduplicated final belief
calls for each task:

- `myopic_width`: one-step root EIG, followed by one-step EIG in the realized
  regenerated branch;
- `fixed_depth2`: cumulative two-step EIG on the fixed root support;
- `dynamic_depth2`: first-step EIG plus expected best future EIG under the
  LLM support regenerated separately for each possible first answer;
- `shuffled_dynamic_depth2`: complete expected continuation values are rotated
  across first actions before scoring, exactly preserving compute and their
  quality distribution while breaking action-specific path coupling;
- `history_blind_depth2`: every branch receives a matched fresh semantic draw
  that omits the simulated answer, followed by an analytical update using that
  answer; execution uses the same realized dynamic updater as the other
  generated-support policies;
- `random`: two deterministic seeded candidates without replacement.

`dynamic_depth2` versus `myopic_width` is the original primary paired
comparison. The co-required `dynamic_depth2` versus `history_blind_depth2`
comparison directly tests whether answer-conditioned regeneration, rather than
another semantic draw, provides the useful non-myopic signal. The original
comparison
isolates the non-myopic first-action objective: both policies use the same
query budget and the same realized branch and final-regeneration machinery.
`fixed_depth2` tests whether fixed-support lookahead is enough, while the
shuffled control tests whether action-specific LLM belief dynamics matter.

## Endpoint Opening And Metrics

Only after all four public block results and private raw hashes independently
replay may a local combined analyzer load the two official endpoint labels per
task. Endpoint labels are never sent to a model. The analyzer scores every
policy's regenerated final belief on the same 64 binary endpoint outcomes.

Primary endpoint:

- paired task-level difference in mean endpoint Brier score,
  `dynamic_depth2 - myopic_width` (negative is better).

Secondary endpoints:

- paired task-level endpoint log-loss difference;
- accuracy and mean truth probability;
- paired Brier/log-loss comparisons to fixed, shuffled, and random controls;
- number of tasks where dynamic and myopic produce different final histories;
- within-task Spearman correlation between each root score and realized
  all-first-action endpoint utility;
- root candidate-label Brier, scored only in the combined analysis, against
  the constant-half baseline.

Report policy means, paired mean differences, sample standard deviations,
wins/ties/losses, and deterministic 95% percentile bootstrap intervals using
20,000 task-resampling replicates. Bootstrap seed: `2026081501` plus the frozen
policy/metric offset implemented in the runner.

## Frozen Development Signal

`development_signal` requires every condition below:

1. all four endpoint-blind blocks independently replay and cover exactly 32
   unique development tasks;
2. dynamic and myopic final histories differ on at least 12 tasks and in every
   execution block; at least 12 changes also give the dynamic-selected action
   a `1e-6`-nat advantage over the myopic-selected action under the dynamic
   score, excluding numerical ties;
3. root candidate Brier beats 0.25, dynamic mean endpoint-ranking Spearman is
   positive, and is no worse than myopic score ranking;
4. dynamic mean Brier improves on myopic by at least 3% relatively;
5. the paired bootstrap probability that dynamic improves Brier is at least
   0.80;
6. dynamic mean log loss is no worse than myopic;
7. dynamic mean Brier is no worse than shuffled-dynamic;
8. all endpoint metrics are finite, and confirmation/test remain unopened.

The matched history-blind amendment additionally requires at least 12 changed
final histories across all four blocks, at least 3% relative Brier improvement,
at least 0.80 paired bootstrap probability of improvement, nonworse log loss,
and nonworse ranking fidelity. These are conjunctive with the eight original
conditions.

The strongest path-dependent-support tier additionally requires at least 12
dynamic/fixed final-history differences, at least 12 first-action differences
that clear the existing `1e-6`-nat dynamic-score margin, a difference in every
execution block, at least 3% relative Brier improvement over fixed depth two,
paired bootstrap improvement probability at least 0.80, non-worse log loss,
and non-worse ranking fidelity. Only the conjunction of the policy,
history-blind, and path-dependent-support families authorizes confirmation.

A pass authorizes writing a fresh 64-task confirmation preregistration, not
running it. A null returns development to the four mechanics tasks for one
mechanism change; it does not permit tuning on confirmation data. The complete
paired result is reported even when one or more gates fail.

## Commands

The protocol manifest was generated before any model call with:

```bash
set -a; source .env; set +a

python scripts/bongard_openworld_luna_vlm_development.py manifest \
  --output results/nonmyopic/bongard_openworld_luna_vlm_development32/PROTOCOL_MANIFEST.json
```

After the Aug-10 serving and mechanics passes, run blocks on separate London
days:

```bash
set -a; source .env; set +a

python scripts/bongard_openworld_luna_vlm_development.py block \
  --block-id a \
  --output-dir results/nonmyopic/bongard_openworld_luna_vlm_development32/block-a-20260811 \
  --run-id bongard-openworld-luna-vlm-development32-a-20260811 \
  --mechanics-result results/nonmyopic/bongard_openworld_luna_vlm_mechanics_tree/bongard-openworld-luna-vlm-mechanics-tree-20260810/RESULT.json \
  --protocol-manifest results/nonmyopic/bongard_openworld_luna_vlm_development32/PROTOCOL_MANIFEST.json \
  --daily-ledger results/nonmyopic/openrouter_daily_budget/2026-08-11.json
```

Block B additionally receives `--previous-result` for block A; block C
receives A and B; block D receives A, B, and C. The combined analyzer receives
exactly four `--block-result` paths and writes one prospective development
result.
