# MultiHop-RAG Semantic BED Mechanics Gate

## Result

The target-blind, zero-LLM mechanics gate closes MultiHop-RAG before opening
the 400-task opportunity split.

Each of the five frozen mechanics tasks used multiple deterministic BM25 root
queries. Every root returned three documents. Continuation queries were derived
only from the question and retrieved root content, excluded all root documents,
and returned three unseen documents. Continuations improved best evidence
recall on 2/5 tasks, confirming that a second retrieval step can be useful.
However, the root query with the strongest immediate evidence always achieved
the best total evidence after continuation. Strict root-choice reversals were
0/5.

## Interpretation

MultiHop-RAG tests retrieval and reasoning across multiple documents, but this
mechanics sample does not expose the complementarity needed for non-myopic BED:
planning changes value only when a lower-immediate-value root enables a better
continuation than the immediate-greedy root. Extra retrieval helped here;
lookahead root selection did not.

This is an exploratory mechanics closure, not a benchmark-wide impossibility
claim. No OpenRouter calls were made, and the untouched opportunity,
development, and holdout task content remains unopened.

Public analysis SHA-256:
`fad9d2c69a14910d7f3345f84ab115e8b5552004cacfc9d086255dbc43783974`.
