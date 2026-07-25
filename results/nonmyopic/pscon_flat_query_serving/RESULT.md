# PSCon Flat Semantic-Query Serving Result

## Outcome

The target-free serving gate failed closed.

- Run: `pscon-flat-query-serving-20260725T140629Z`
- Requests / HTTP attempts: `10 / 10`
- Prompt / completion tokens: `13,760 / 601`
- Reasoning tokens / retries / forced exits: `0 / 0 / 0`
- Cost: `$0.0130245`
- Hidden target loaded: `false`
- Responder, score, and endpoint calls: `0`

All ten responses had the exact three-line shape and all assignment strings had the
required length of 20. Eight responses used all three labels. Two responses populated
only two labels:

- query 5 used `A/C` only;
- query 9 used `A/B` only.

Both questions were semantically binary despite the requested three-option interface.
The strict parser therefore rejected the batch at the frozen “every option used”
condition. There was no normalization, relabeling, repair, reissue, or target access.

## Interpretation

This is stronger than a punctuation or JSON-type failure: Mini can generate useful
questions and complete product assignments, but the joint task of inventing three
exhaustive options and assigning every product sometimes collapses to two semantic
outcomes. Per preregistration, the flat three-way interface is closed with no V3.

The next distinct PSCon mechanism, if pursued, should separate semantic roles:

1. generate a single binary clarification question;
2. classify product titles in separate calls using only `Y/N/U`;
3. let an independent responder produce the realized `Y/N/U` transition.

That decomposition keeps question and likelihood generation LLM-native while removing
the failed joint three-way serialization task. It requires a new target-free serving
gate before any efficacy endpoint.

Chinese PSCon data remains untouched. OatML was not used.
