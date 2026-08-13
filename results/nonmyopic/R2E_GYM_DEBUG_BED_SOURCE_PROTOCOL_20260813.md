# R2E-Gym Debug-BED source protocol

Date frozen: 2026-08-13 Europe/London

Status: **prospective metadata-only source admission; zero model calls**

## Scientific purpose

This is a genuinely new debugger population after the SWE-smith population was
closed by its V3 privacy-ordering failure. R2E-Gym is considered because it
provides released executable repository environments, native test outcomes, and
an existing DebugGym PDB adapter. Source admission does not establish a
non-myopic opportunity or authorize task execution, model serving, or endpoint
inspection.

## Immutable source bindings

- R2E-Gym repository commit:
  `0d94c4eb9431cd195c55a7ea3abd54006c9a1735`
- R2E-Gym repository tree:
  `fcf3cab14b8fe62b0cddcc52f3a3a5d4fb2855cf`
- `R2E-Gym/R2E-Gym-Lite` dataset revision:
  `8d3163011f01f9393bb3dc7700497a79a8686ae5`
- dataset tree:
  `fa991fad5d79264d94047a50233b4240281e27a4`
- DebugGym repository commit:
  `cc3fe3ef4ce08919e522eb00ea1bea5689f3b53e`
- DebugGym repository tree:
  `54b04c0313b66d72c7f285dc255d2693164a7193`

Only the eight content-addressed `train-*.parquet` objects are in scope. Their
ordered LFS SHA-256 values and byte sizes are frozen in the audit implementation.
The expected population is exactly 4,578 rows with exactly these 14 fields:

```text
repo_name, docker_image, execution_result_content, expected_output_json,
modified_entity_summaries, modified_files, num_non_test_files,
num_non_test_func_methods, num_non_test_lines, parsed_commit_content,
problem_statement, prompt, relevant_files, commit_hash
```

## Metadata-only boundary

The source audit may project only these six columns:

```text
repo_name, docker_image, commit_hash, num_non_test_files,
num_non_test_func_methods, num_non_test_lines
```

It may materialize neither values nor lengths from `problem_statement`,
`prompt`, `parsed_commit_content`, `modified_files`,
`modified_entity_summaries`, `expected_output_json`,
`execution_result_content`, or `relevant_files`. Downloading the immutable
Parquet objects is not task access; only the projected Arrow result may be
materialized by this audit.

## Eligibility and split

A row is eligible exactly when:

1. repository, Docker image, and 40-character lowercase hexadecimal commit are
   all nonempty;
2. `num_non_test_files` is in `[1, 2]`;
3. `num_non_test_func_methods` is in `[2, 4]`;
4. `num_non_test_lines` is in `[4, 80]`;
5. its commit hash and Docker image are each unique in the train population.

Eligible rows are ordered by
`sha256("r2e-gym-debug-bed-v1|" + commit_hash + "|" + docker_image)`.
The first 16 are a private structural-screen prefix. The next 64 are the sealed
opportunity split, the next 64 the sealed development split, the next 96 the
sealed confirmation split, and all remaining rows are reserve. Public output
contains only aggregate counts, per-split ordered-ID hashes, a repository-count
histogram, and immutable source/runtime hashes. It must contain no repository,
image, commit, row offset, task text, patch, modified-file/entity value, expected
output, execution log, relevant source, or endpoint value.

After this protocol is committed and pushed, a passing source audit authorizes
only exact-row retrieval of the 16 screening rows using predicate-pushed
selection at the Arrow application boundary. The structural screen must be
separately frozen before opening those rows and must never materialize any
nonselected payload row.

## Source gates

Admission passes only if:

1. all repository, dataset, file-object, runtime-file, population, and schema
   bindings reproduce exactly;
2. at least 256 rows are eligible and they span at least eight repositories;
3. the five split segments are complete, ordered, unique, and disjoint;
4. every selected row satisfies the frozen eligibility rule;
5. the R2E-Gym Docker runtime exposes container reset and native test execution,
   while DebugGym exposes the R2E adapter, `/root/run_tests.sh`, and PDB tooling;
6. the public manifest satisfies the metadata-only serialization boundary;
7. OpenRouter calls, cost, endpoint access, and OATML cluster use are zero.

Any failed gate closes this exact source protocol. No threshold, subset, salt,
or population repair is permitted after observing the result.

## Downstream mechanics requirements

Source admission is necessary but insufficient. Before any model call, a
separate zero-call mechanics protocol must show on at least five of eight
screening tasks:

- at least three coherent executable counterfactual defect hypotheses;
- a genuinely adaptive probe whose second action depends on the first result;
- a changed first action and positive exact depth-two utility margin over a
  compute-matched receding-myopic control;
- deterministic fresh-container replay and a nonsaturated sealed patch/test
  endpoint under common random numbers.

Failure closes the route before serving. A later LLM interface additionally
requires exact schema, semantic answer-obedience, cost, and fail-closed gates.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

