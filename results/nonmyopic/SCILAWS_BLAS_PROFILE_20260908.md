# Matrix-thread overhead profile

Profiled the first frozen task (baseball), zero initial table, first action's first
order4 branch, all eight independent terminal references. Software workload only;
no new source or scientific outcomes. cProfile includes overhead; times are local
observations, not stable benchmarks. Both processes used NumPy1.26.4/SciPy1.15.3
and Python3.12. OpenBLAS0.3.23.dev reported eight threads by default.

| Setting | Profiled terminal time | Reference evaluations |
|---|---:|---:|
| Default eight BLAS threads | .379s | 1200 |
| Explicit one BLAS thread | .036s | 1200 |

Default82 matrix solves consumed .314s cumulatively. With one thread, adaptive
integration (.029s) becomes dominant. This supports explicit single-thread BLAS for
small-matrix numerical diagnostics. It does not establish speed on every task.

Add opt-in --single-thread-blas using threadpoolctl3.6.0; default execution remains
unchanged. The result records the requested setting. Thread limiting is scoped and
restored on exit. No quadrature, posterior, tolerance, node count or cap changes.

Do NOT rerun the full branch panel merely because timing improves. The profile uses
150 evaluations per terminal integral on this branch;1024 such integrals would
require153600 evaluations, above100000. The prior panel already reached93937 before
timing out. Eliminating time overhead may only expose the evaluation limit. A
dependency-valid next step needs a measured work-reduction design, not another
predictably incomplete full audit or a cap reset.

Next inspect the saved branch references and integration setup for analytically
removable repeated work or a separately bounded joint integration representation.
Keep all source/LLM/h3 qualification gaps explicit. No paid calls or source data.

Verification: three first-task synthetic initial scenarios, first actual branch,
all eight terminal references, eight-vs-one threads: scores agree to1e-12 and
evaluation counts agree. Four focused tests including the existing assessment gate
pass in8.44s; scoped lint passes. The thread-control test skips when its optional
dependency is absent; it was installed and executed here. Current account usage
220.376693994, balance24.623306006, dailyspend0. No active experiment remains.
