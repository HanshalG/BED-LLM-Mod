# RegretBench DeepSeek Support-Recovery Preregistration

Date frozen: 2026-08-07

## Claim Under Test

Test the first causal link needed for LLM-native non-myopic BED:

> After an LLM-generated clarification question receives an exact cooperative
> answer, the same LLM should regenerate a support that contains the sampled
> true interpretation more often than a matched history-blind redraw.

This is a support-recovery mechanism test, not a policy-efficacy result. It does
not compare depth-two and myopic query selection and cannot open confirmation.

## Frozen Source And Privacy Boundary

The run is bound to:

- source audit result SHA256
  `d7a10f15ecf6779520fbb20712c8d43059d8b03168fa904fe72e82ffe87578de`;
- source protocol manifest SHA256
  `8a46b40395487aae0857d1a61d6f680beef579414c4580acf09d7e0b30b38e97`;
- mechanics ID hash
  `707be5a1d1f86d6a0dc08ee61df77da1b9597093ac557d2e7706fcad8ef3b2f6`;
- development ID hash
  `29d33b2fda0be7cc4eea6f4d9d4fe74fe200b5c580632dcb7ba844c3b825af69`.

The model sees exactly an opaque `task_id`, the ambiguous `prompt`, and a
dialogue list. It never sees hidden intents, descriptions, answer aliases,
slots, facets, reference questions, metadata, the benchmark belief, the true
intent index, or an endpoint. A pre-dispatch privacy audit rejects forbidden
source strings in model payloads unless the same text already occurs in the
visible prompt or realized dialogue.

Raw model responses, generated questions, exact environment replies, and hidden
controls remain under an untracked `private/` directory. Public output contains
only hashes, counts, booleans, aggregate metrics, accounting, and gates.

## Model And Support Interface

- Model: `deepseek/deepseek-v4-flash-0731` through OpenRouter.
- Reasoning: explicitly disabled and excluded.
- Temperature: `0.7`.
- Maximum output: `2,200` tokens.
- No scientific retry, repair, coercion, continuation, or replacement.
- OpenRouter transport retry count: exactly zero.

Every strict response contains exactly eight hypotheses and four ranked
clarification questions. A hypothesis contains a short interpretation, a
concise final answer, and a nonnegative prior weight. The parser normalizes
weights, removes exact duplicate `(interpretation, final_answer)` pairs, and
requires at least four unique hypotheses. It does not use hidden truth to
filter or repair support.

The first ranked question is selected without hidden information. The official
RegretBench `SemanticActionMapper` maps it after selection. If supported, the
environment replies with the sampled true intent's exact slot value for the
mapped facet. If unsupported, it replies with a fixed refusal and the task is
excluded from scientific efficacy while remaining in mechanics accounting.

Truth coverage is a deliberately conservative lexical endpoint. A generated
final answer matches a pipe-separated hidden alias only when their lowercase
alphanumeric normalizations are equal, or when one normalized string contains
the other and the shorter side has at least two tokens and eight characters.
Different equal-length normalizations never match.
The matcher is frozen before model calls and is applied identically to root,
conditioned, and blind supports.

## Matched Construction

For each task:

1. a root request sees the prompt only;
2. the sampled true intent is chosen by a frozen local seed;
3. the first generated question is mapped and answered by the environment;
4. the conditioned refresh sees prompt plus that question and answer;
5. the history-blind refresh sees the original prompt only; and
6. conditioned and blind refreshes use the same requested model seed and are
   dispatched adjacently in the same bounded batch.

The root request is identical for both arms. There is no benchmark-generated
question and no truth insertion into either support.

## Exact-10 Mechanics Smoke

Use all four frozen mechanics tasks. Make four root requests, then conditioned
and blind requests for the first three tasks, for exactly ten accepted requests
and ten HTTP attempts. Seeds:

- truth: `202608081100 + local_task_index`;
- root: `202608083000 + local_task_index`;
- matched refresh: `202608084000 + local_task_index`.

Every smoke gate must pass:

- exact `10` requests and HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all ten responses are strict and have at least four unique hypotheses;
- all four root responses contain exactly four unique questions;
- the first question maps to a supported facet for each of the first three
  tasks;
- the privacy audit passes for every request; and
- cost is at most `$0.20`.

Coverage outcomes are descriptive and cannot authorize the smoke. A pass
authorizes only the frozen development mechanism run below.

## Development Mechanism Run

Use all 64 frozen development tasks. Make exactly `192` accepted requests:
64 root requests followed by 64 adjacent conditioned/blind pairs. Seeds:

- truth: `202608082000 + local_task_index`;
- root: `202608085000 + local_task_index`;
- matched refresh: `202608086000 + local_task_index`.

Bootstrap `20,000` paired task resamples with seed `202608087000`.

Scientific population: tasks whose selected first question maps to a supported
facet. Primary endpoint: mean conditioned-minus-blind truth coverage on that
population. Primary gates are conjunctive:

- at least `48` supported tasks;
- at least `16` supported tasks whose root support misses truth;
- at least `8` supported tasks where conditioned and blind coverage differ;
- conditioned-minus-blind coverage is at least `0.05`;
- bootstrap probability that this difference is positive is at least `0.80`;
- conditioned recoveries exceed conditioned losses; and
- among root-missing supported tasks, conditioned-minus-blind recovery is at
  least `0.10`.

Mechanics gates additionally require exact `192` requests and HTTP attempts,
zero retries/provider retries/reasoning/forced exits, every response strict with
at least four unique hypotheses, every root with four unique questions, all
privacy checks, and total cost at most `$0.50`.

All gates must pass. Failure closes this exact DeepSeek support interface. A
pass authorizes only a separately preregistered development policy experiment;
confirmation remains sealed.

## Budget Discipline

The exact-10 cap is `$0.20`; development cap is `$0.50`. Both must also pass the
calendar-day `$5.00` Europe/London ledger before adapters are created. These
caps reserve the rest of each day for higher-value frozen experiments; no calls
are made merely to consume the allowance.
