# BrowseComp-Plus Cached Semantic Information Audit

## Result

The preregistered zero-call first-link gate passes on the five already-open
mechanics tasks and 30 cached root beliefs.

The primary target-blind score is semantic entropy reduction after the first
search. It reaches:

- 0.8000 pairwise accuracy against truth-mass gain over 50 comparable pairs;
- 0.6098 against immediate evidence over 41 pairs;
- 0.8684 against future evidence gain over 38 pairs; and
- 0.7857 against total two-search evidence over 42 pairs.

The entropy-selected root has at least as much truth mass as the original
direct-score root on four of five tasks, with two wins and one loss. The loss is
task 747: all six searches increase entropy, and choosing the least-negative
change misses the one root that introduces the true answer.

Jensen-Shannon divergence and novel posterior mass are stronger diagnostics,
but they were not the preregistered primary and are not substituted after the
fact. JSD reaches 0.9200 pairwise accuracy against truth-mass gain and 0.9048
against total evidence.

## Interpretation

The result isolates the failure in the preceding mechanics run. The LLM's
observation-conditioned answer distribution contains useful information, while
asking the same model to emit an absolute future-value score destroys the
ranking. This supports testing an output-distribution estimator, not another
prompted value scorer.

This audit is post hoc on open mechanics tasks. It is directional instrument
evidence only, not an efficacy result and not authorization for a large run.
No OpenRouter calls or OatML resources were used.

The public analysis is `ANALYSIS.json`, SHA-256
`3348404fd8d30147188292e26af36e6cf308d2ea58e89efdf57d6b79c9ddfe9e`.
