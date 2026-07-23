# UCI Thyroid Workup 26B Trajectory Confirmation Failure

The preregistered 50-patient trajectory confirmation **failed closed before any
performance endpoint was computed**. The exact qualification and passed 32-cell
proposal-quality gate remain banked; this result limits their transfer to sustained
receding-horizon execution with Gemma 4 26B.

Job `106379` accepted the first five logical policy cells. At cell 5, after the
history

`collect blood -> TSH -> T3 -> TT4 -> age`, the machine-fixed roots included
`query:t4u`. For T4U outcome branches 2--5, the first response proposed
`query:t4u` again as the second action. Repeating an already queried assay is not in
that root's legal follow-up menu. The registered validation-feedback retry again
proposed `query:t4u` for those branches, so the strict parser stopped the run.

The failure made seven physical requests: 16,251 prompt tokens and 1,005 completion
tokens, with zero reasoning tokens, zero forced exits, and `$0` API cost. Five cells
were accepted and two invalid attempts belong to the terminal cell. No random,
depth-one, depth-two, entropy-AUC, truth-log-AUC, recovery, or accuracy endpoint was
written or inspected.

Per the frozen registration, there is no 26B parser repair, alternate seed,
replacement patient, or rerun. The supported boundary is now:

- Exact depth two decisively improves realized UCI thyroid trajectories.
- Non-thinking 26B named continuations pass a fresh 32-cell exact proposal-quality gate.
- The same 26B interface is not robust enough for sustained trajectory execution as
  registered because it can repeat the current query after menus shrink.

Artifacts:

- `results/nonmyopic/THYROID_WORKUP_26B_CONFIRMATION_PREREGISTRATION.md`
- `results/nonmyopic/thyroid_workup_26b_confirmation_20260723/CONFIRMATION_FAILURE.json`
