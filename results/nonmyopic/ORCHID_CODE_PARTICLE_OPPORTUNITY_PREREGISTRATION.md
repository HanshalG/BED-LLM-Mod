# Orchid Executable-Particle Opportunity Preregistration

Date: 2026-07-25

## Question

Do intentionally ambiguous function requests produce an unsaturated population
of LLM-generated executable programs in which exact clarification inputs have
enough held-out correctness range, and enough myopic ranking error, to justify
path-dependent depth-two BED?

This is an initial-support opportunity gate. It makes no branch-regeneration
calls and cannot support a non-myopic efficacy claim.

## Frozen Sources And Manifest

- Orchid dataset:
  `https://huggingface.co/datasets/SII-YDD/Orchid`.
- Orchid commit: `55ddeb0d3670d22d420e072816cf4f034cf62caf`.
- Orchid HumanEval JSONL SHA-256:
  `412b5caf99df978482af62d4e3ea7dec089cb457b99d47fcd5d9eb56329e0082`.
- Sandboxed execution code:
  `https://github.com/kasia-kobalczyk/active-task-disambiguation`.
- Execution commit: `4c8ecb4d4ffdbffcc611366743fc1e2461037772`.
- Eligibility: at least eight structured tests; every input and output is a
  literal; every relation is exact equality; inspected Orchid indices 0-2 are
  excluded.
- Eligible tasks: 52 of 164.
- Manifest seed: `24374`.
- Seeded eligible-manifest SHA-256:
  `cbc79146077f4d7b8d4d68365b4d816acf45777a15ae3e14f2fe5c62b1e96920`.
- Frozen tasks: indices `97` and `133`, the first two manifest entries.
- Ambiguity view: `Vagueness_prompt` only. No alternate ambiguity type may be
  substituted after results.

The zero-call audit used only test counts, syntax, literal/equality status, and
prompt lengths. It did not generate a program or compute a query endpoint.

## Frozen Query/Holdout Splits

The split seed is `24375 + task_id`; sorted indices are:

| Task | Query indices | Held-out endpoint indices |
|---:|---|---|
| 97 | `2, 5, 6, 7` | `0, 1, 3, 4` |
| 133 | `0, 3, 5, 6, 7, 9` | `1, 2, 4, 8, 10, 11` |

Only query inputs are candidate clarification actions. Held-out inputs are never
used in EIG or root selection. Their outputs define external program
correctness.

## LLM Particle Contract

- Model: `openai/gpt-5.4-mini` through OpenRouter.
- Reasoning disabled; temperature `.7`; output limit 1,500 tokens.
- Eight independent program particles per task.
- Exactly 16 physical requests total.
- Each response must be complete executable Python defining the frozen entry
  point.
- Plain code or one Python code fence is accepted prospectively; prose outside
  a fenced block is ignored only when that Python fence exists.
- Every candidate program is executed in the pinned isolated reliability-guard
  worker on all frozen query and held-out inputs with a one-second per-input
  timeout.
- Parse failures, missing entry points, exceptions, timeouts, or output-width
  failures are filtered as invalid particles; at least six valid particles are
  required per task.
- Duplicate programs and duplicate behaviors are retained as generated belief
  mass.
- No program repair, completion replacement, retry for scientific validity, or
  response reissue is allowed.

## Frozen Scoring Order

Before any target output value loads:

1. generate and parse all 16 programs;
2. execute every valid program on every frozen input;
3. filter invalid particles;
4. freeze each query's output partition and entropy.

Only then re-read the Orchid rows and parse exact target outputs.

For each query:

- immediate EIG is entropy of the generated-particle output partition;
- exact target output selects surviving particles; and
- endpoint is the survivors' mean pass fraction on the held-out tests.

The initial endpoint is mean held-out pass fraction across all valid initial
particles. The myopic query maximizes immediate EIG. The opportunity oracle
maximizes the externally measured posterior held-out endpoint. Ties use frozen
query order.

## Exact Gates

All gates must pass:

1. exactly 16 physical requests and 16 HTTP attempts;
2. zero transport retries, reasoning tokens, and forced exits;
3. at least six valid executable particles per task;
4. initial held-out pass fraction is between `.10` and `.90`, inclusive, on
   each task;
5. at least two queries have EIG at least `.30` nats on each task;
6. query-dependent endpoint range is at least `.10` on each task;
7. oracle query improves over the initial endpoint by at least `.10` on each
   task;
8. oracle and myopic roots differ on at least one task;
9. mean oracle endpoint advantage over myopic is at least `.05`; and
10. adapter cost is at most `$0.50`.

A pass authorizes only a separately frozen path-dependent depth-two development
run on these now-open tasks. A failure closes this exact Orchid route; no
alternate ambiguity type, favorable task subset, threshold relaxation, or
particle rerun is allowed.

## Budget

- Projected cost: `$0.15`.
- Hard run cap: `$0.50`.
- Project-ledger spend before the run: `$86.18245031920742`.
- Monday new-work allowance remaining: `$14.9608545`.
- Last authenticated OpenRouter remaining balance: `$44.206476634`, or
  `$19.206476634` above the protected `$25` reserve.
- OatML resources: prohibited.

Both live and local budget gates must be checked again before the paid command.

## Zero-Call Verification

The exact entry point completed a full deterministic dry run with 16 simulated
requests, sandbox execution, delayed target-output loading, scoring, gates, and
strict artifacts. The deliberately identical fixture passed mechanics and
failed all opportunity gates. Focused tests:

```text
pytest -q tests/test_orchid_code_particle_opportunity.py \
  tests/test_atd_code_first_link_audit.py
7 passed
```
