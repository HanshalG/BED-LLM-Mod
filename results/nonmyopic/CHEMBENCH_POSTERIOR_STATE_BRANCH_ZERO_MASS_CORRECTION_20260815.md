# ChemBench Posterior-State Branch Zero-Mass Correction

Date: 2026-08-15 (Europe/London)

The first complete-run attempt stopped before case 7/36 and wrote no result.
The first six cases were printed only as progress diagnostics. No gate summary,
depth result, LLM call, network call, or endpoint was produced.

The failure occurred when duplicated child belief features left a retained
Lloyd center with no assigned outcomes. The frozen protocol says empty
clusters retain their prior center, each cluster receives a nearest realizable
observation, and branch probability is empirical cluster mass. The
implementation incorrectly raised instead of representing the empty cluster.

The correction chooses the globally nearest actual outcome as the retained
center's representative and assigns probability zero. Such a branch has no
effect on expected risk. Nonempty clusters are unchanged. A duplicate-feature
regression test requires probabilities `[1, 0, 0]` and total mass one.

All frozen data, seeds, features, clustering initialization, iteration limit,
reference count, thresholds, and decisions remain unchanged. A new
implementation hash must be committed and pushed before rerunning the complete
panel.
