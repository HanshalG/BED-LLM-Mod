# Active Task Disambiguation Code First-Link Audit

Date: 2026-07-25  
Status: frozen before aggregate hidden-test evaluation  
Interface: `atd-code-first-link-audit-1`

## Question

Do exact information-gain scores over LLM-generated executable program
hypotheses rank candidate test inputs by an external program-correctness
endpoint?

This is a zero-cost development audit, not a policy result. It screens a
text-native substrate before any new depth-two generation. The LLM is
load-bearing in the released artifact: it generated both the program particles
and candidate test inputs. Candidate outputs, oracle observations, and hidden
tests are executed exactly.

## Frozen Source

- Official repository:
  `https://github.com/kasia-kobalczyk/active-task-disambiguation`
- Commit: `4c8ecb4d4ffdbffcc611366743fc1e2461037772`
- HumanEval JSONL SHA-256:
  `882c3d56432b2b5b9e568398d7ebdf54f2c84fdb05fef3b833a3d935ad71861c`
- Released method/model: `active-reasoning`, `gpt-4o-mini`, `iter_0`.
- Population: all 47 tasks with complete released hypothesis and question files:
  `5,6,17,26,33,36,38,39,41,50,54,55,64,70,73,74,76,77,81,82,90,91,93,95,96,98,101,103,106,107,109,110,111,114,115,118,121,122,123,134,138,139,141,143,147,154,159`.

All 47 tasks are public development data. No later claim may call them held out.
The APPS tasks are not inspected by this audit and remain available for a
separately frozen smoke only if every gate passes.

## Frozen Measurement

For each task:

1. Retain every released initial completion as a particle, including duplicate
   programs as separate samples.
2. Execute every released candidate input under every particle. Errors and
   timeouts remain outcome categories.
3. Score each root by the entropy of its empirical output partition in nats.
4. Freeze all root scores for all 47 tasks.
5. Only then execute each particle against the official hidden HumanEval tests.
6. For the canonical output of each root, filter to matching particles and
   measure their hidden-test pass fraction.

The target-blind selected root maximizes immediate EIG with a fixed lexical tie
break. The external oracle root maximizes posterior hidden-test pass fraction.
Report initial and selected pass mass, selected gain over the mean candidate,
within-task Spearman correlation, dynamic range, and top-1 regret.

## Frozen Gates

All must pass:

1. at least 35 usable tasks with at least eight particles and three roots;
2. at least 25 usable tasks with initial pass fraction in `[.02,.95]`;
3. at least 25 tasks with root pass-fraction range at least `.10`;
4. mean selected gain over initial pass fraction at least `.05`;
5. mean selected gain over the task's candidate-root mean at least `.02`;
6. mean finite within-task Spearman at least `.10`;
7. mean top-1 regret at most `.15`.

Failure closes this exact released-trace route. Do not select a favorable task
subset, remove error outcomes, deduplicate particles, change ties, or tune
thresholds.

## Conditional Next Step

Only a full pass authorizes a separately preregistered, sub-`$0.50` APPS smoke:
GPT-5.4 non-reasoning generates initial executable programs and candidate tests,
then regenerates programs and follow-up tests on simulated outcome branches.
The canonical program and official tests remain hidden until all target-blind
root scores freeze. No OatML resources are used.
