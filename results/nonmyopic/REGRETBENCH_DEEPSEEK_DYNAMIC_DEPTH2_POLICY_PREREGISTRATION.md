# RegretBench DeepSeek Dynamic Depth-Two Policy Preregistration

Date frozen: 2026-08-07

## Claim

Test whether non-myopic planning over an LLM's own answer-conditioned belief
regeneration improves two-question identification of a hidden semantic intent.
The primary comparison is:

- `dynamic_depth2`: choose the first question by expected terminal truth-mass
  Brier after answer-conditioned support regeneration and an EIG-selected
  second question;
- `myopic_width`: choose the first question by immediate EIG while consuming
  the exact same generated tree and executing the same dynamic continuation.

The LLM is load-bearing: it generates the semantic hypotheses, questions,
answer likelihoods, and every path-dependent future support. The released
finite CIG is never supplied to the planner and has already been proven to have
zero strict depth-two gain on this cohort.

## Dependencies

This protocol is frozen before any RegretBench model response. It is bound to:

- source result
  `d7a10f15ecf6779520fbb20712c8d43059d8b03168fa904fe72e82ffe87578de`;
- source manifest
  `8a46b40395487aae0857d1a61d6f680beef579414c4580acf09d7e0b30b38e97`;
- support-recovery preregistration
  `32825b97d9c73bbe940d2f0dfff5cb5deebd80f2cfec16812f227659a88a6dfe`;
- development split hash
  `29d33b2fda0be7cc4eea6f4d9d4fe74fe200b5c580632dcb7ba844c3b825af69`.

The exact-10 support-recovery smoke and 64-task support mechanism must both
pass every frozen gate. A null, mechanics failure, partial artifact, hash
change, or unverified predecessor opens zero policy calls.

## Model And Information Boundary

- Model: `deepseek/deepseek-v4-flash-0731` through OpenRouter.
- Reasoning: disabled and excluded for every planner/environment role.
- Temperature: `0.7`; maximum output: `2,200` tokens.
- Concurrency: `128` for the large independent branch block.
- No scientific retry, repair, coercion, continuation, or task replacement.

The model sees only opaque `task_id`, ambiguous `prompt`, and dialogue. It
never sees hidden intents, aliases, descriptions, slots, facets, reference
questions, metadata, true intent, benchmark belief, policy labels, scores, or
endpoints. Generated questions and supports are visible to later model calls
only when they occur in that simulated or realized dialogue.

A separately labelled `naive_thinking` baseline uses
`openai/gpt-5.6-luna` with medium reasoning and reasoning text excluded. It
generates clarification questions directly from visible dialogue and never
sees particles, branch scores, candidates, hidden CIG fields, or endpoints.
Reasoning is not used by any BED policy.

Each strict enriched support contains exactly:

- eight hypotheses, each with an interpretation, concise final answer,
  nonnegative prior weight, and four predicted user replies;
- four ranked single-dimension clarification questions; and
- reply `j` in every hypothesis aligned to question `j`.

All eight `(interpretation, final_answer)` pairs must be distinct after exact
normalization; a duplicate fails this enriched interface. Priors are
renormalized across the eight particles. Replies induce deterministic
likelihood partitions; their weighted outcome entropy is the question's EIG.

## Dynamic Tree

For each development task, one initial call generates the shared support and
four root questions. For every root, every one of the eight initial hypotheses
is treated as a simulated truth. Its aligned predicted reply is appended to
history. Two independent enriched supports are generated from that conditioned
history.

Each conditioned draw is immediately paired with a same-seed history-blind
draw that sees the initial prompt only. The same requested seed is also reused
across all four roots for a fixed `(task, simulated truth, draw)` tuple. This
task-level common-random-number schedule prevents root-specific generator seed
luck from entering the estimated root ranking while leaving each root's prompt
and simulated reply distinct. Thus every task has:

```text
4 roots * 8 simulated truths * 2 draws * 2 arms = 128 branch calls
```

For a branch support and simulated truth:

1. match the simulated truth's final answer to regenerated final answers using
   the frozen conservative lexical matcher;
2. choose the branch's second question by maximum EIG over its regenerated
   support;
3. marginalize over predicted replies within the matched truth group; and
4. calculate expected terminal Brier `(1 - posterior truth-group mass)^2` and
   log loss after that second reply.

No match receives Brier `1` and log loss at the frozen probability floor
`1e-12`. A root's dynamic risk is the initial-prior-weighted mean over all eight
simulated truths and both draws. Lower risk is better. This is a proper
Bayesian rollout over the LLM's own hypotheses, with future states supplied by
the LLM generator rather than a fixed support.

## Policies

All policies share the exact initial support, candidate questions, conditioned
and blind trees, and actual-history calls.

- `dynamic_depth2`: minimum conditioned dynamic terminal Brier.
- `history_blind_depth2`: the identical scorer on matched prompt-only draws.
- `myopic_width`: maximum initial immediate EIG; ignores branch scores.
- `fixed_depth2`: exact two-question terminal truth-group Brier on the initial
  support without regeneration.
- `random`: uniform root from a frozen per-task seed.
- `naive_thinking`: Luna medium reasoning directly asks one question, observes
  the exact environment reply, then directly asks a second question.

Ties use the lowest original question index. `myopic_width` is the primary
compute-matched control. `fixed_depth2` tests whether generic finite-support
lookahead, rather than path-dependent regeneration, explains any gain.
`history_blind_depth2` tests whether branch sampling noise alone explains it.
`naive_thinking` is a stronger-model, unmatched-compute descriptive baseline;
it cannot pass, rescue, veto, or abort the primary experiment. Its serving
smoke is attempted before development, but a smoke failure disables only the
baseline. Formal Luna calls and the DeepSeek endpoint measurements run through
separate adapters sharing the same hard run budget. Any baseline transport,
schema, reasoning, support, or action-coverage failure is banked as an
unavailable baseline while the primary DeepSeek pass/null decision continues.

## Realized Execution And Endpoint

The hidden true intent is sampled uniformly by a frozen local seed only after
all planning responses and root selections are frozen. Each distinct selected
root is executed once and shared by every policy selecting it:

1. the official RegretBench mapper parses the generated first question;
2. the environment returns the exact true slot value or the fixed unsupported
   reply;
3. a fresh enriched support is generated from realized history;
4. its maximum-EIG second question is selected without hidden information;
5. the official mapper and exact environment answer it; and
6. a final enriched support is generated from the complete two-question
   history.

Within a task, every distinct selected root uses the same requested seed for
its first refresh and the same separate requested seed for its final refresh.
Different tasks retain distinct seeds. This realized common-random-number
coupling makes paired policy differences depend on their histories rather than
avoidable root-specific model-seed draws.

The primary endpoint is final truth-mass Brier, where truth mass is the summed
final-support probability of answers matching a hidden answer alias. Secondary
endpoints are final truth log loss, lexical truth coverage, supported-action
rates, and truth mass after question one. Continuous mass prevents a binary
coverage ceiling from carrying the main claim.

All comparisons are paired by task and common hidden truth. Report means,
sample standard deviations, wins/ties/losses, 20,000 paired task-bootstrap
intervals and probability of improvement, plus selection disagreement and
predicted-to-realized Spearman diagnostics.

For `naive_thinking`, the first question is generated before hidden truth
access. After its exact first reply, one DeepSeek enriched support measures
first-step truth mass, while Luna independently chooses question two from
visible dialogue. A final DeepSeek support after the second exact reply supplies
the identical terminal endpoint used for every BED policy.

## Exact-10 Enriched Serving Smoke

Use all four mechanics tasks: four initial enriched supports followed by one
conditioned/blind pair for each of the first three tasks. Exact seeds:

- initial: `202608088000 + task_index`;
- matched branch: `202608088100 + task_index`.

Pass requires exact 10 accepted requests and HTTP attempts; zero retries,
provider retries, reasoning, and forced exits; all strict enriched schemas with
exactly eight unique hypotheses; four unique questions and aligned replies;
at least two informative root questions per initial support; at least one
informative follow-up per branch support; all three selected first questions
officially supported; every privacy audit passing; and cost at most `$0.20`.
No endpoint is opened and no smoke efficacy value may authorize passage.

## Exact-10 Naive-Thinking Smoke

After the enriched smoke passes, Luna runs a separate exact-10 medium-reasoning
transport and semantics diagnostic on the four mechanics tasks:

- four first-question calls, seeds `202608088200..202608088203`;
- four second-question calls after exact environment replies, seeds
  `202608088300..202608088303`; and
- two independent first-question redraws on the first two tasks, seeds
  `202608088400..202608088401`.

Every response is a strict one-question object. Passage requires exact ten
accepted requests and HTTP attempts, zero retries/provider retries/forced
exits/forced finalization, positive reasoning tokens, all ten questions
nonempty and syntactically interrogative, all eight executed questions mapped
to supported official facets, every prompt privacy audit passing, and cost at
most `$0.20`. No answer, coverage, or policy efficacy value enters passage.
Failure disables only the descriptive baseline and does not prevent the
primary policy development run.

## Development Request Schedule

- initial seeds: `202608089000 + task_index`;
- matched branch seed:
  `202608100000 + task_index*16 + hypothesis*2 + draw`, reused across roots;
- hidden truth: `202608130000 + task_index`;
- random root: `202608140000 + task_index`;
- realized first-history seed: `202608110000 + task_index`, reused across roots;
- realized final-history seed: `202608120000 + task_index`, reused across roots;
- bootstrap: `202608150000`.
- naive first question: `202608160000 + task_index`;
- naive second question: `202608170000 + task_index`;
- naive first-history support: `202608180000 + task_index`;
- naive final-history support: `202608190000 + task_index`.

The primary planning block is exact `64 + 8,192 = 8,256` calls. Primary
realized calls are one
first-history and one final-history call per distinct selected root, at most
`512`. An available naive baseline adds `128` Luna calls and its endpoint adds
`128` calls through a separate DeepSeek adapter. The result records primary,
baseline-endpoint, Luna, and combined counts separately. Missing or partial
baseline counts cannot invalidate exact primary accounting. Maximum combined
total remains `9,024`: at most `8,896` DeepSeek and `128` Luna.

## Mechanics Gates

All must pass:

- every predecessor, source, split, seed, prompt, privacy, and response binding;
- exact request/attempt accounting and zero retries/provider retries/reasoning/
  forced exits for the primary DeepSeek adapter;
- every primary initial, simulated-branch, and realized support is strict with exactly
  eight unique hypotheses and aligned four-reply vectors;
- every task has at least two informative initial roots;
- at least 90% of simulated branch supports have an informative follow-up;
- every primary policy has at least 48 supported first actions and 40 supported
  second actions;
- every primary public artifact excludes raw questions, replies, aliases, facets,
  hidden intent indexes, and raw responses; and
- conditioned/blind pairs share exact seeds, all four simulated roots share
  each task/truth/draw seed, realized roots share task-level first/final seeds,
  and seeds remain distinct across task/truth/draw groups; and
- combined spend, including any attempted baseline calls, is at most `$3.50`.

The result separately reports Luna transport/reasoning, baseline DeepSeek
endpoint-support validity, prompt privacy, and the same `48/40` action-coverage
targets as descriptive diagnostics. None enters primary mechanics, science,
status, or confirmation authorization.

## Scientific Gates

All are conjunctive:

1. dynamic and myopic roots differ on at least `16/64` tasks;
2. dynamic and history-blind roots differ on at least `12/64` tasks;
3. dynamic and fixed roots differ on at least `12/64` tasks;
4. conditioned dynamic predicted Brier improves over the myopic-selected root
   by at least `0.01` on average;
5. dynamic minus myopic realized Brier is at most `-0.02`, bootstrap
   probability of improvement is at least `0.90`, and wins exceed losses;
6. dynamic minus history-blind realized Brier is at most `-0.015`, bootstrap
   probability of improvement is at least `0.80`, and wins exceed losses;
7. dynamic minus fixed realized Brier is at most `-0.01`, bootstrap probability
   of improvement is at least `0.80`, and wins exceed losses;
8. dynamic mean log loss is no worse than each of myopic, history-blind, and
   fixed; and
9. on dynamic/myopic changed-root tasks, predicted Brier advantage has Spearman
   correlation at least `0.15` with realized Brier advantage and bootstrap
   probability of positive correlation at least `0.80`.

Failure closes this exact policy interface. A pass authorizes only a separately
frozen confirmation on the untouched 64-task confirmation split. No threshold,
draw count, matcher, seed, policy, endpoint, or favorable subset may change.

## Daily Budget

The enriched smoke cap is `$0.20`, naive smoke cap is `$0.20`, and combined
policy development cap is `$3.50`. On Aug 8
they run only after the Luna naive smoke and RegretBench support-recovery stages,
using the same account-wide opening usage. Combined worst-case daily caps are:

```text
Luna naive smoke          $0.20
support-recovery smoke    $0.20
support-recovery dev      $0.50
enriched policy smoke     $0.20
naive-thinking smoke      $0.20
dynamic policy dev        $3.50
                         ------
total                     $4.80
```

Within this chain, the enriched policy smoke is the only policy smoke that can
gate primary development. The RegretBench naive-thinking smoke is attempted
and banked first, but its failure only removes the descriptive baseline.

Unspent allowance does not roll over. Calls are never added merely to approach
the cap.
