# Interactive SWE Path Opportunity Mechanics Amendment

## Status

Frozen after the first opportunity invocation stopped during target-blind tree
construction and before any endpoint column was loaded.

## Failure

One opportunity hidden issue yielded only one valid 20--1,200-character chunk.
The preregistered second-answer rule excludes the root answer, leaving no
eligible continuation chunk. The runner raised instead of recording the row as
non-sequential.

No `patch`, `test_patch`, or `files` column was loaded. No audit artifact,
utility, root choice, aggregate metric, or gate value existed.

## Exact Amendment

When a hidden issue has fewer than two valid chunks, its root has an empty
continuation set. Final coverage equals immediate coverage. The unchanged
three-distinct-root-answer usability gate necessarily excludes the row.

This changes no tree with at least two chunks and does not alter chunking,
tokens, probes, BM25, action order, endpoint, cohort, thresholds, or gates.
The audit may restart once after this amendment and focused test are committed.

API calls and OatML use remain zero.
