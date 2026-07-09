# Path A Workshop Draft

This directory holds the write-now paper skeleton for the location-finding
Path A result. The Phase 4 sweep numbers are intentionally left as placeholders
until `results/location_depth_sweeps/*_REPORT.md` exists and the
pre-registered analysis in `LOCATION_DEPTH_PATH_A_RUNBOOK.md` has been run.

Draft validation:

```bash
python scripts/validate_paper_draft.py
```

The validator runs `pdflatex`, `bibtex`, and two final `pdflatex` passes in a
temporary build directory, then checks that the draft stays within the 4--6 page
workshop target.

Required figure slots:

- Main paired constrained/unconstrained depth contrast.
- Ranking-fidelity diagnostic.
- Cost-vs-depth table/plot.
- Qualitative strategies plus query trajectories.
