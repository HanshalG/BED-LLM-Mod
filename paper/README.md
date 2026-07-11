# Path E Workshop Draft

This directory holds the outcome-independent Path E draft for belief-guided EIG
arbitration on Paprika customer-service troubleshooting. The method, endpoint
audit, controls, held-out protocol, and limitations are written before the
50-task headline outcomes are opened.

Draft validation:

```bash
python scripts/validate_paper_draft.py
```

The validator runs `pdflatex`, `bibtex`, and two final `pdflatex` passes in a
temporary build directory, then checks that the draft stays within the 4--6 page
workshop target. Outcome-dependent figures remain deliberately absent until the
frozen analyzer and mandatory manual endpoint audit are complete.
