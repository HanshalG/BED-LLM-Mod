# Validation-First Non-Myopic BED Draft

This directory holds the evidence-supported workshop draft on non-myopic BED
with LLM-derived probabilistic models. It combines exact planning controls,
the paired Rock Diagnosis policy result, the exact Gated Sensor qualification and
LLM-interface audit, exact UCI Mushroom, Cleveland, and Thyroid semantic-unlock
qualifications, the passed Thyroid 26B proposal gate and failed 26B/GPT trajectory
transfers, a strict exact range-gated d3 opportunity with a failed
26B serving gate, the banked animals result, the natural-location depth
audit, and the Paprika and MediQ validation failures. Endpoint-invalid Paprika outcomes
are diagnostic only, and no MediQ policy claim is made after the frozen likelihood
gate failed.

Draft validation:

```bash
python scripts/validate_paper_draft.py
```

The validator runs `pdflatex`, `bibtex`, and two final `pdflatex` passes in a
temporary build directory, then checks that the draft stays within the 4--6 page
workshop target and includes the required validity claims and figure.
