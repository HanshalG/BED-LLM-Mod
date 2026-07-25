# SWE-bench Lite Bug-Localization Unlock Audit

## Status

Frozen before computing any retrieval outcome and before any LLM call. This is
a zero-call development audit of whether real software issues expose a
non-myopic, semantically enabling search structure. Failure closes the
construction before model use.

## Data

- Official dataset: `SWE-bench/SWE-bench_Lite`.
- Hugging Face revision:
  `69611d31007e1c6731db8bd5b5c3f2d33f5bab6e`.
- Development Parquet SHA-256:
  `24d670403c3c8690a0f0d741bdcb8800322f58a407d6be148ad99d04b5d8bb32`.
- Test Parquet SHA-256:
  `f46f2e3f003f2552932393da4b223e1e0456a2c71eba8b73ae58f29646c1278b`.

Only the 23-instance official development split is eligible for this audit.
The 300 test rows have been read only through the `instance_id`, `repo`, and
`base_commit` columns. Their problem statements, patches, changed-file
endpoints, and tests remain unread.

SWE-bench Lite restricts each issue to a one-file gold edit. The hidden target
for this audit is that changed source file. A development issue is excluded as
directly leaked if its problem statement case-insensitively contains the full
target path, target basename, or a basename stem of at least five characters.

The frozen nine no-direct-leak development IDs are:

1. `sqlfluff__sqlfluff-1625`
2. `sqlfluff__sqlfluff-2419`
3. `sqlfluff__sqlfluff-1733`
4. `sqlfluff__sqlfluff-1763`
5. `pvlib__pvlib-python-1707`
6. `pylint-dev__astroid-1333`
7. `pyvista__pyvista-4315`
8. `pydicom__pydicom-1413`
9. `pydicom__pydicom-1139`

Every repository is indexed at the instance's exact `base_commit`.

## Deterministic Search Tree

The repository corpus contains tracked Python files at the base commit. Each
document combines six repetitions of path/identifier tokens with up to 200,000
characters of source. Tokenization splits paths, punctuation, snake case, and
camel case.

Each issue produces up to five target-blind first-search roots:

- full issue text;
- first nonempty line;
- concatenated backtick/code spans;
- exception and identifier-like tokens; and
- longest paragraph.

After normalization and deduplication, at least three roots are required. BM25
returns the top three files per root.

For each root, four observation-conditioned followups are constructed without
the endpoint:

- one expansion from each retrieved file using its 20 highest-IDF source terms;
  and
- one aggregate expansion using 12 highest-IDF terms across all three files.

Each followup retrieves three new files while excluding that root's first
results. No LLM, embedding model, patch text, test patch, or changed-file label
enters query generation or retrieval.

## Endpoint and Controls

Utility is binary changed-file coverage.

- Immediate value: whether a root's top three contains the changed file.
- Pair value: whether the union of a root and one followup contains it.
- Oracle pair: best root/followup pair.
- Immediate-greedy root: highest immediate value, with frozen root-order
  tie-breaking.
- Greedy-with-oracle-tail: best followup under that fixed greedy root.
- Non-myopic gap: oracle-pair value minus greedy-with-oracle-tail value.

This is an opportunity audit, not a deployable policy result.

## Frozen Gates

All gates must pass:

- all 9 issues reproduce their single changed-file endpoint and base commit;
- all issues have at least 3 distinct roots;
- mean distinct root top-1 files is at least `3.0`;
- direct root retrieval covers the target on at most 6/9 issues, preventing
  saturation;
- a second step improves over the best immediate root on at least 3/9 issues;
- mean pair gain is at least `.33`;
- the oracle root differs from the immediate-greedy root on at least 2/9;
- a positive non-myopic gap occurs on at least 2/9; and
- mean non-myopic gap is at least `.22`.

Passing authorizes only a separately preregistered LLM hypothesis-generation
smoke on fresh test IDs. Failure closes this deterministic
issue-to-code-retrieval construction without threshold, tokenizer, top-k,
repository, or cohort tuning.

## Budget

This audit uses zero API calls and zero OpenRouter spend. OatML remains paused,
and the protected `$25` reserve is unaffected.
