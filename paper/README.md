# Path B Workshop Draft

This directory holds the workshop draft for the Path B Gate-0-failure outcome.
Path A is retained as the motivating entropy/task-loss mismatch; Path B tests
expected posterior RMSE as the repair and stops before Gate 1 when ranking
fidelity remains below threshold.

Draft validation:

```bash
python scripts/validate_paper_draft.py
```

The validator runs `pdflatex`, `bibtex`, and two final `pdflatex` passes in a
temporary build directory, then checks that the draft stays within the 4--6 page
workshop target.

Required figure slots:

- Path B task-loss ranking-fidelity diagnostic.
- Banked Path A constrained/unconstrained depth contrast.
