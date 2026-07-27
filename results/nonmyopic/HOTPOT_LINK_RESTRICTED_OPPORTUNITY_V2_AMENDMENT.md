# HotpotQA Link-Restricted Opportunity V2 Amendment

V1 materialized the frozen mechanics and development cohorts, then failed
before aggregate metrics because one official row referenced an invalid
supporting-fact sentence index.

Before rerunning qualification, V2 freezes exactly one source-validity rule:

- if the existing `qualification(row)` raises `ValueError`, mark that row
  `malformed: true` and `qualifies: false`;
- record malformed counts by split;
- do not catch any other exception.

Every split, source hash, ordered row, strict-unlock definition, top-four
qualification, paragraph-link action graph, first-20 selection rule, and
opportunity threshold remains unchanged. V2 makes zero model calls and does
not inspect confirmation or retained-holdout endpoints.

This amendment is frozen before any V2 aggregate count or selected task ID is
computed.
