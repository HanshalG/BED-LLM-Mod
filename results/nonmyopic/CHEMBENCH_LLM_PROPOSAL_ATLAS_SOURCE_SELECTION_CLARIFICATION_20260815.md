# ChemBench LLM Proposal-Atlas Source Selection Clarification

Date frozen: 2026-08-15 (Europe/London)

## Purpose

The proposal-atlas protocol requires both four tasks in each of the nine
`(difficulty, earliest_history_length)` strata and 36 unique hidden generators.
Its per-stratum family-coverage rule does not state how generators repeated
across difficulty slices are excluded. Independent selection within each
stratum yields only 29 unique generators and therefore contradicts the explicit
36-generator source gate.

This clarification fixes the deterministic allocation before any task
manifest, model response, API request, or semantic result exists. It changes no
task construction, source version, prompt, model setting, metric, threshold,
control, endpoint, or budget.

## Deterministic Allocation

After constructing every admissible source row exactly as frozen:

1. Form the nine strata.
2. Sort strata by increasing admissible-row count, breaking ties by difficulty
   in `easy`, `medium`, `hard` order and then by history length.
3. Maintain one global set of selected generator names.
4. Before selecting from a stratum, remove every row whose generator is in the
   global selected set.
5. Select four rows by the frozen within-stratum family-coverage rule: choose
   the row whose core family has appeared least often among selections for that
   stratum, then break ties by SHA256 of

   ```text
   2026083700|difficulty|history_length|model_name
   ```

   using literal `|` separators with no surrounding whitespace, and finally by
   model name.
6. Add each selected generator to the global set immediately.
7. Serialize the final manifest in canonical `easy`, `medium`, `hard` and
   history-length `1`, `2`, `3` order. Within each stratum, the first three
   selected rows are atlas-development tasks and the fourth is held out, as in
   the original protocol.

The source gate still requires exactly 36 unique generators and at least ten
core families. Failure to allocate four rows in any stratum stops before a
manifest or request.

## Zero-Call Preflight

Using only the pinned source and frozen trigger, admissible-row counts are:

| Stratum | Rows |
|---|---:|
| easy / 1 | 8 |
| hard / 2 | 8 |
| medium / 1 | 9 |
| hard / 1 | 10 |
| hard / 3 | 14 |
| medium / 3 | 14 |
| easy / 3 | 16 |
| medium / 2 | 20 |
| easy / 2 | 22 |

The clarified allocation produces 36 unique generators spanning 12 core
families. These counts are a source-only implementation preflight, not a model
or policy result.
