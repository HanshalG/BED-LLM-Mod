# Interactive SWE Path-Dependent Clarification Opportunity

## Status

Frozen before reading any hidden issue, gold patch, test patch, or modified-file
field outside the ten rows disclosed by the public dataset preview. This is a
zero-call opportunity audit, not an LLM-policy result.

## Source And Split

- Dataset: `cmu-lti/interactive-swe`, revision
  `a56830ede5eee0925cb7d735d422a207927925bd`.
- Parquet SHA-256:
  `bdc6663d3f931d9ea04ea2edaec5eca6e406c2f122d7f1956ef0ec859691b5cd`.
- Official Ask-or-Assume code: commit
  `728464a755f15fd448aff61a5cb32dce97fd9c1d`.
- Selection seed: `24402`.

The ten Astropy rows shown in the Hugging Face preview are excluded. Their
hidden fields are disclosed development only. Sorting and shuffling the other
490 IDs freezes:

- opportunity: 120 IDs, hash
  `8f361dd221076a7a8bdb14a55978a469a3890f687289d44460322ff7c5fade0f`;
- development: 40 IDs, hash
  `6f6bfdd0b0715b3cb7375ab7f3a2325f0d1a8e3111139c8fc464965542f842db`;
- sealed holdout: 330 IDs, hash
  `e945d95d8765d0c6fedb2c3d585ca5b00bc4b9f80f9f5166fa7df0dead238b0c`.

The public manifest contains IDs and repository names only. It is committed
before opportunity hidden fields are read. Its SHA-256 is
`7ac50a60bc59046b4406ef65b9ca148554b5b7c596194bd8d2ad66f33d811221`.

## Exact Clarification Environment

The visible state is the underspecified `problem_statement`. The hidden user
knowledge is the released full `original_issue`, split deterministically into
20--1,200-character paragraph/sentence chunks.

Eight fixed root probes represent clarification dimensions:

1. expected behavior;
2. actual failure;
3. reproduction;
4. affected API or component;
5. scope and edge cases;
6. version and environment;
7. implementation location; and
8. tests and acceptance criteria.

A dependency-free BM25 answerer returns the highest-scoring hidden chunk for a
probe, with source order breaking ties. Duplicate root answers are removed.
After a root answer, continuation queries comprise:

- its first ten frequency-ordered tokens absent from the visible issue; and
- each fixed probe augmented with the first two such tokens.

The second answer is the top hidden chunk other than the root answer. Duplicate
second-answer indices are removed. Every root answer and continuation is frozen
before endpoint columns load.

## External Endpoint

The target is the set of normalized gold-change tokens that:

- occur in modified file paths or added/removed gold/test-patch lines;
- also occur in the hidden full issue; and
- do not occur in the visible underspecified statement.

Utility is distinct target-token coverage in the revealed answer chunks.
Greedy maximizes immediate one-answer coverage. Depth two maximizes the best
two-answer coverage. Ties prefer greater alternate-horizon coverage and then
the earlier root.

A row is usable with at least five target tokens and three distinct root
answers. It is a strict opportunity only when depth two selects a different
root, sacrifices immediate coverage, and obtains strictly greater final
coverage after both roots receive their own best continuation.

The target is a deterministic first-link proxy, not SWE-bench resolve rate. It
tests whether hidden requirement evidence has a native delayed-value structure
before paying an LLM to generate semantic beliefs and questions.

## Frozen Gates

All must pass:

- exact source and split reproduction;
- exactly 120 opportunity rows;
- at least 60 usable rows;
- at least 18 strict opportunities;
- strict opportunities span at least six repositories;
- mean normalized depth-two gain over greedy is at least `.025` across usable
  rows; and
- every strict row has both the immediate sacrifice and final gain.

Failure closes this exact wrapper before model calls. There is no threshold,
chunker, token, probe, BM25, cohort, or endpoint repair.

The excluded ten-row pilot produced seven usable rows and three strict
opportunities under this exact broad gold-change vocabulary. This pilot fixes
the thresholds but is not included in any gate.

## Conditional LLM Stage

Passage authorizes only a target-free serving and ranking smoke on fresh
development rows. GPT-5.4 non-reasoning will generate requirement/file
hypotheses, clarification questions, and answer-conditioned refreshed supports.
The same deterministic hidden-issue answerer will be shared by:

- myopic immediate support improvement;
- fixed-support depth two;
- refreshed-support depth two;
- matched-width myopic; and
- seeded random.

Gold change tokens, modified files, and tests remain unavailable until every
question, support, likelihood, score, and selected root freezes. A separate
preregistration must define that policy and its endpoint before any call.

No OpenRouter calls or OatML resources are authorized here.
