# RegretBench Frozen Paper-Fragment Protocol

Date frozen: 2026-08-07

## Input Boundary

The fragment generator accepts only a `FROZEN_REPORT.json` produced under
`REGRETBENCH_REPORTING_PROTOCOL_20260807.md`. It recomputes that report from the
hash-bound `RESULT.json` and independent `VERIFICATION.json` and requires exact
equality before writing TeX.

Development and confirmation fragments target the same conditional manuscript
include. A verified development report may create a provisional or null
fragment. A later authorized, independently verified confirmation report may
replace it. No unverified, partial, pooled, optional-baseline, or secondary
analysis can write the fragment.

## Fixed Content

Every fragment states that the released RegretBench intents, aliases, facets,
slots, and finite CIG were hidden from the planner; DeepSeek generated the
eight semantic hypotheses, four clarification questions, and per-hypothesis
reply likelihoods; and the benchmark mapper supplied exact environment replies
only after selection.

For a mechanically valid result, the fragment prints all four primary paired
comparisons: myopic width, history-blind regeneration, fixed-support depth two,
and random. Each row includes first-root disagreements, dynamic-minus-control
Brier mean and sample SD, paired 95% interval, bootstrap probability of
improvement, and wins/ties/losses. The caption identifies the aligned generated
likelihood endpoint and invalid-trajectory penalty. It also prints the frozen
predicted-to-realized Spearman diagnostic.

For mechanics failure, no efficacy table is printed and the fixed mechanics
sentence is used.

## Fixed Claim Mapping

- `mechanics_failure_no_scientific_result`: no efficacy result.
- `development_policy_null_confirmation_forbidden`: development null and
  confirmation forbidden.
- `provisional_development_signal_confirmation_required`: provisional only;
  confirmation required.
- `confirmation_null_development_not_confirmed`: failed independent
  replication.
- `confirmed_llm_native_nonmyopic_signal`: independently confirmed signal.

The exact interpretation sentence comes from the frozen report. Fresh
regeneration, Luna, pooled analyses, and subgroups are omitted from the primary
table and cannot alter the paragraph or claim tier.

## Manuscript Boundary

`paper/main.tex` contains two conditionals keyed to the same generated file.
When the file is absent, the detailed late Number Game audit remains visible and
no RegretBench content is added. When the file is present, the audit beginning
with the post-hoc ranking analysis and ending with the fresh history-blind
mechanism evidence is suppressed, then the frozen fragment is included.

The core Number Game setup, primary powered replication, model-role swap, and
support-rich control remain visible in both states. This is a fixed page-budget
substitution independent of the RegretBench stage, result status, claim tier,
effect direction, or effect size. It cannot select content after observing an
outcome.

The include itself is:

```tex
\IfFileExists{generated/regretbench_result.tex}{%
  \input{generated/regretbench_result.tex}%
}{}
```

The generated file is absent before data, so the frozen manuscript still
renders the same six-page content. Successful-result and mechanics-failure
fragments must both compile within the four-to-six-page target and pass the
claim validator before commit.

This protocol changes no experiment or endpoint, makes zero model calls, and
costs `$0`.
