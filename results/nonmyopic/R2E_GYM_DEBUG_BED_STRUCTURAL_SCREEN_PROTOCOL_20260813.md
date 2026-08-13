# R2E-Gym Debug-BED structural screen protocol

Date frozen: 2026-08-13 Europe/London

Status: **prospective exact-row structural screen; zero model calls**

## Dependency and scope

This screen requires the passing R2E-Gym source V3 canonical replay and its
unchanged V1 manifest. It may open only the exact 16-row structural-screen
prefix fixed by `sha256("r2e-gym-debug-bed-v1|" + commit_hash + "|" +
docker_image)` over the frozen eligible population.

Selection must first be reconstructed from the six permitted metadata columns.
Payload retrieval then uses an Arrow dataset scanner with an exact
`commit_hash IN <16 frozen hashes>` predicate and projects only:

```text
repo_name, docker_image, commit_hash, parsed_commit_content,
modified_files, modified_entity_summaries
```

At the application boundary exactly 16 payload rows and no nonselected payload
row may materialize. The screen must not read `problem_statement`, `prompt`,
`expected_output_json`, `execution_result_content`, `relevant_files`, or any
endpoint. Private selected rows remain outside the repository.

## Structural qualification

A selected row qualifies exactly when:

1. `parsed_commit_content` is valid JSON with a nonempty `file_diffs` list;
2. `modified_files` contains one or two unique nonempty paths;
3. `modified_entity_summaries` contains 2--4 unique function, method, or class
   entities in non-test files;
4. each retained entity has a nonempty name and file, integer
   `1 <= start_lineno <= end_lineno`, and its file occurs in `modified_files`;
5. at least two retained entities have distinct `(file, name, start, end)`
   identities;
6. the structural entity count equals the frozen
   `num_non_test_func_methods` metadata count.

Test files are paths containing a component or basename beginning with `test`,
ending in `_test.py`, or equal to `conftest.py`. Entity types are normalized
case-insensitively and qualify when `type` or `ast_type_str` contains
`function`, `method`, or `class`.

The screen passes only if at least eight of 16 rows qualify. Mechanics slots are
the first eight qualifying rows in the already frozen source order; there is no
reordering or best-case selection. Public output may contain counts, ordered ID
hashes, entity-count and file-count histograms, source/result hashes, gates, and
zero-call accounting only. It may not contain repositories, images, commits,
paths, entity names, line numbers, task text, patches, tests, source, or endpoint
values.

## Authorization

A pass authorizes only a separately frozen native-amd64 mechanics protocol over
those exact eight slots. That protocol must define three or more executable
counterfactual defect worlds, adaptive probe semantics, exact likelihoods,
depth-two versus compute-matched controls, fresh-container replay, and sealed
patch/test endpoints before execution. This screen authorizes no container,
model call, endpoint, development, confirmation, or paper claim.

A failure closes this exact R2E-Gym route. No threshold, entity rule, ordering,
or replacement prefix may be changed after observing the result.

## Accounting

- OpenRouter calls: `0`
- OpenRouter cost: `$0`
- OATML cluster use: none

