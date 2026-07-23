# Animals Dynamic-Brier Development Plan

Status: **development instrument; no confirmatory claim**.

## Motivation

The target-blind semantic ranker failed both its ungated and
generation-capacity-gated fresh holdouts. Direct expected support retention was
also weak in post-endpoint analysis. The next scorer should use the actual LLM
belief transition directly rather than ask another LLM to predict it.

## Score

For each candidate question, current hypothesis `h`, and answer branch `y`,
the existing likelihood model supplies `p(y|h)` and the production
regeneration/filtering path returns approximate belief `q_y`. The target-blind
expected quadratic score is:

```text
sum_y [2 sum_h b(h) p(y|h) q_y(h) - p(y) sum_j q_y(j)^2]
```

The reported dynamic-Brier gain subtracts the current-belief score
`sum_h b(h)^2`. A regenerated belief receives credit for assigning probability
to plausible pseudo-truths and a concentration penalty for unsupported mass.
Omitted hypotheses receive probability zero without an arbitrary log floor.

## Development Gate

Instrument a small development probe with non-thinking Gemma 4 26B and the
unchanged production belief generator. Compare dynamic-Brier top choices with
immediate EIG using measurement-only expected truth coverage.

The first probe uses seed `24278`, 20 states, three candidates per state, and
the first 20 targets from the now-inspected capacity-holdout list. These targets
cannot support a confirmatory claim; they are deliberately reused to avoid
consuming a fresh set during score development.

Proceed to a separately frozen fresh holdout only if:

1. every target remains absent from model-visible inputs;
2. all branch updates and likelihood calls are fully accounted;
3. dynamic-Brier candidate rank association with expected truth coverage is
   positive;
4. its top-choice mean expected truth coverage exceeds immediate EIG;
5. wins exceed losses.

Failure stops this scorer without coefficient fitting or combining it with the
inspected semantic-ranker endpoints.
