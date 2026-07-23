# Cleveland Heart Workup Qualification Audit

Audit passed: **True**.

- Entropy-AUC gain: `+0.068747`, independent 95% CI `[+0.056895, +0.080474]`.
- Truth-log-AUC gain: `+0.068747`, independent 95% CI `[+0.047452, +0.090717]`.
- Every stored action was independently re-scored and verified optimal.
- Every state, deterministic observation, posterior metric, paired truth, and aggregate comparison was replayed from the raw Cleveland cohort.
- The audit made zero LLM calls.
