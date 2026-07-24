# ClinDiag Binary Joint-Model Gate Result

Date: 2026-07-24

Status: **10-call interface smoke failed on frozen content semantics; no 44-call
joint-model smoke, structural gate, planner, or holdout followed.**

## Result

The provider and grammar mechanics were clean:

- exactly 10 GPT-5.4 requests;
- zero reasoning tokens and zero retries;
- both supports parsed at size 12;
- both query sets contained six safe binary actions;
- no blocked confirmatory or intervention action passed the parser;
- all six requested gatekeeper answers parsed;
- both exact duplicate answer labels matched;
- no literal target diagnosis appeared.

However, the frozen protocol also required every response to avoid reporting
missing/unavailable source data. The `rare231` response to:

> Was there ambiguity of the external genitalia noted at birth?

was:

> External genitalia were not reported as ambiguous at birth.

This is an absence-of-documentation response, not the required synthesized
patient-specific `yes` or `no` fact. The exact duplicate returned the same `no` answer
and wording but changed provenance from `recorded` to `synthetic`.

## Instrument Finding

The automated artifact marked the smoke passed because the fixed missingness regex
covered “not recorded,” “not available,” and similar phrases but omitted “not
reported.” The manual content audit occurred before the larger stage and applies the
frozen semantic criterion directly. The automated pass is therefore an instrument
false positive, not authorization to continue.

The parser is not amended and the result is not rerun. Adding another synonym after
seeing it would repair the acceptance instrument post hoc. The exact per-query binary
gatekeeper line stops.

## Cost

- 10 requests;
- 7,312 prompt tokens and 1,147 completion tokens;
- zero reasoning tokens;
- `$0.03548500`;
- conservative remaining project balance: `$23.89722372`.

## Consequence

Binary propositions solve the non-exhaustive option-set problem, but independent
per-query gatekeeper calls still fall back to chart missingness and can disagree about
provenance. A materially distinct environment qualification should generate a complete
six-answer patient response function jointly in one hidden-case call, then audit and
duplicate that entire response function. Joint generation can enforce cross-query
consistency and makes “fill every absent fact” a single structured task rather than six
independent opportunities to abstain.
