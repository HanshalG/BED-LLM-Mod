# SWE-bench Lite Graph Unlock V2 Result

## Decision

The frozen zero-call opportunity gate failed. The exact graph-expansion
construction is closed before any LLM call, development-stage endpoint access,
or paid policy evaluation.

## Results

All 24 opportunity rows reproduced at their exact base commits. Thirteen were
excluded because the issue directly named the single changed Python target,
leaving 11 eligible no-direct-leak cases, below the preregistered minimum of
12.

Among the 11 eligible cases:

- direct root retrieval covered `6/11` targets;
- graph continuation added target coverage for `2/11`, below the required
  `4/11`;
- strict non-myopic gaps occurred for `2/11`, below the required `4/11`;
- strict gaps spanned two repositories, below the required three;
- mean pair gain was `.1818`, passing the `.15` mean gate; and
- mean strict non-myopic gap was `.1818`, also passing the `.15` mean gate.

The two strict cases were:

- `pallets__flask-5063`: the root query `Flask routes to return
  domain/sub-domains information` retrieved files including
  `src/flask/app.py`; graph expansion then reached the hidden target
  `src/flask/cli.py`.
- `sphinx-doc__sphinx-8627`: the root query `autodoc isn't able to resolve
  struct.Struct type annotations` retrieved files including
  `sphinx/ext/autodoc/directive.py`; graph expansion then reached
  `sphinx/util/typing.py`.

The exact split, complete execution, root-diversity, direct-coverage, and mean
effect gates passed. The eligibility-count, pair-prevalence,
strict-gap-prevalence, and repository-breadth gates failed. The conjunction
therefore failed.

## Interpretation

Import-aware continuation can expose a meaningful delayed target on some
real bug reports: conditional on the 11 eligible cases, both mean effect
metrics exceed the frozen threshold. The opportunity is not prevalent or
broad enough for the proposed evaluation, however. More than half of the
opportunity set directly leaks the changed target, and only two independent
repositories exhibit a strict root-dependent advantage.

This does not show that LLMs cannot reason non-myopically about software. It
shows that this exact target-blind BM25-plus-import-graph interface does not
provide a sufficiently powered non-myopic decision problem on the
prospectively selected SWE-bench Lite cohort.

No top-k, edge weight, eligibility rule, cohort, or threshold was changed
after endpoints were read. The conditional multisample LLM smoke is not run.
The 12-task development and 264-task holdout problem, patch, changed-file, and
test columns remain sealed.

## Budget

- API calls: `0`.
- OpenRouter spend: `$0`.
- OatML cluster use: none.

## Artifacts

- Preregistration:
  `results/nonmyopic/SWEBENCH_LITE_GRAPH_UNLOCK_V2_PREREGISTRATION.md`
- Full opportunity audit:
  `results/nonmyopic/swebench_lite_graph_unlock_v2/OPPORTUNITY.json`
- Audit SHA-256:
  `b55d523cb2a68350b4a267fcad7e24345e6668d55cbdaacec69a6d7b3303cf95`
- Dataset revision:
  `69611d31007e1c6731db8bd5b5c3f2d33f5bab6e`
