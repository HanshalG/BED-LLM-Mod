# STATE — living project state (the agent MUST keep this file current)

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug — fix it before doing anything else. If this file
conflicts with GOAL.md's design history, this file wins.

## CURRENT STATE (updated 2026-07-09)

Phases 1–3 are DONE; Phase 4 is RUNNING; paper skeleton is started in `paper/` and
compiles. The Path A infrastructure is committed locally and tagged `path-a-final-sweep`.
GOAL.md holds the design specs, outcome playbook, and definition of done — consult it for
detail; execute from here.

Status of the Minimum Publishable Package:

1. Ranking-fidelity gate: **PASSED** — `results/ranking_fidelity/PHASE1_26B_A4B_GATE.md`
   (26B A4B, 60 records: entropy ρ 0.38–0.44, truth-log-prob ρ ≈ 0.37 positive at all
   depths, top-1 regret improves with depth 0.33→0.21, RMSE ρ ≈ 0).
2. Headline depth sweep: **RUNNING** — checked 2026-07-09 09:05 London after the user
   requested canceling the slow/low-yield GH200 path to free nodes for other users. GH200
   jobs `101993` and `101994` were already absent from the live queue; direct `gh200`
   partition checks through 09:05 London showed no user-owned GH200 jobs, so no `scancel`
   was needed. The 09:05 live queue also had no `gh200` entries for this user.
   Remaining Path A jobs are optimized MSC jobs `101998`/`101996` running on
   `oat15`/`oat14`, constrained MPP30 job `102018` running on `oat16`, and unconstrained
   MPP30 job `102019` running on `oat21`. At 08:43 London, those four MSC jobs were the
   only live jobs for this user. No final or recovered metrics files existed at the 09:03
   check. Decision logs were still at the cheap-control counts: MPP30 constrained `90`,
   MPP30 unconstrained `90`, full50 constrained `150`, and full50 unconstrained `150`. Slurm
   stderr is non-erroring for all four jobs. Recent progress snapshots were `101996`
   reached 86/256 in a fresh block; `101998` reached 31/198 in a fresh block; `102018`
   reached 218/256 in a fresh block; and `102019` reached 132/256 in a fresh block.
   None of the four decision logs advanced, and no metrics or recovered metrics files
   exist yet. All four jobs remain live in `squeue`; all four Slurm stderr files advanced
   during this poll. The 08:45 London progress snapshot had `101996` at 224/256 in its
   current block, `101998` at 91/198, and `102018`/`102019` starting fresh 256-prompt
   blocks; run logs were updated at 08:44–08:45 for the MPP30 pair and 08:40 for the
   full50 pair. The 08:46 London progress snapshot had `101996` in a shorter 61-prompt
   block at 16/61 after finishing the previous 256-prompt block, `101998` at 130/198,
   `102018` at 32/256, and `102019` at 27/256. The 08:47 London progress snapshot had
   `101996` at 10/256 in another block, `101998` at 170/198, `102018` at 71/256, and
   `102019` at 123/256. The 08:48 London progress snapshot had `101996` at 82/256,
   `101998` in a shorter 89-prompt block at 15/89 after finishing the previous 198-prompt
   block, `102018` at 131/256, and `102019` at 216/256. The 08:50 London progress snapshot
   had `101996` at 148/256, `101998` starting a tiny 8-prompt block, `102018` at 178/256,
   and `102019` starting another 256-prompt block after finishing the previous one. The
   08:51 London progress snapshot had `101996` at 191/256, `101998` at 13/256, `102018`
   nearly through its current block at 247/256, and `102019` at 32/256. The 08:53 London
   progress snapshot had `101996` at 8/256 in a new block, `101998` at 90/256, `102018`
   at 43/256 in a new block, and `102019` at 177/256. The 08:54 London progress snapshot
   had `101996` at 43/256, `101998` at 157/256, `102018` at 111/256, and `102019`
   starting another 256-prompt block after finishing the previous one. The 08:56 London
   progress snapshot had `101996` at 93/256, `101998` at 194/256, `102018` at 151/256,
   and `102019` at 55/256. The 08:57 London progress snapshot had `101996` at 169/256,
   `101998` nearly through its current block at 252/256, `102018` at 223/256, and `102019`
   at 145/256. The 08:58 London progress snapshot had `101996` at 226/256, `101998`
   starting a new 256-prompt block, `102018` in a short 50-prompt block at 8/50, and
   `102019` nearly through a short 127-prompt block at 115/127. The 08:59 London progress
   snapshot had `101996` starting another 256-prompt block, `101998` at 20/256, `102018`
   at 62/256, and `102019` at 31/256. The 09:01 London progress snapshot had `101996`
   at 36/256, `101998` at 91/256, `102018` at 112/256, and `102019` at 132/256. The
   09:02 London progress snapshot had `101996` at 59/256, `101998` at 167/256, `102018`
   at 183/256, and `102019` at 219/256. The 09:03 London progress snapshot had `101996`
   at 128/256, `101998` at 234/256, `102018` nearly through its current block at 246/256,
   and `102019` starting another 256-prompt block. Decision logs remained unchanged at the
   cheap-control counts (`90`, `90`, `150`, `150`) and no recovered metrics existed. The
   09:06 London poll again found no final/recovered metrics and the same decision counts;
   all four jobs were live with fresh stderr progress: `101996` at 11/72, `101998` at
   122/256, `102018` at 73/192, and `102019` at 214/256. The 09:07 London poll again
   found no final/recovered metrics and the same decision counts; all four jobs were live
   with fresh progress: `101996` at 14/256, `101998` at 147/256, `102018` at 123/192, and
   `102019` at 33/121 after completing its previous block. Error greps only found the
   known pip dependency-resolver setup warning, not runtime tracebacks/OOMs/kills. The
   09:08 London poll again found no final/recovered metrics and unchanged decision counts;
   all four jobs were live with fresh progress: `101996` at 60/256, `101998` at 219/256,
   `102018` at 161/192, and `102019` at 28/256. Runtime error counts for traceback,
   runtime error, OOM, and kill signatures were zero for all four jobs. The 09:09 London
   poll again found no final/recovered metrics and unchanged decision counts; all four
   jobs were live with fresh progress: `101996` at 92/256, `101998` at 255/256, `102018`
   starting a tiny 8-prompt block after completing its previous block, and `102019` at
   86/256. Runtime error counts remained zero for all four jobs. The 09:10 London poll
   again found no final/recovered metrics and unchanged decision counts; all four jobs
   were live with fresh progress: `101996` at 151/256, `101998` at 8/256 after completing
   its previous block, `102018` at 3/8, and `102019` at 140/256. Runtime error counts
   remained zero for all four jobs. The 09:11 London poll again found no final/recovered
   metrics and unchanged decision counts; all four jobs were live with fresh progress:
   `101996` at 200/256, `101998` at 18/256, `102018` starting a fresh 256-prompt block,
   and `102019` at 211/256. Runtime error counts remained zero for all four jobs. The
   09:12 London poll again found no final/recovered metrics and unchanged decision counts;
   all four jobs were live with fresh progress: `101996` at 233/256, `101998` at 34/256,
   `102018` still at the start of its fresh 256-prompt block, and `102019` at 44/116 after
   completing its previous block. Runtime error counts remained zero for all four jobs.
   A later 09:12 London poll again found no final/recovered metrics and unchanged decision
   counts; all four jobs were live with fresh progress: `101996` starting a short
   52-prompt block after completing its previous block, `101998` at 73/256, `102018` at
   5/256, and `102019` at 17/256 after completing its previous block. Runtime error
   counts remained zero for all four jobs. The 09:13 London poll again found no
   final/recovered metrics and unchanged decision counts; all four jobs were live with
   fresh progress: `101996` at 14/256 after completing its short block, `101998` at
   112/256, `102018` at 22/256, and `102019` at 37/256. Runtime error counts remained
   zero for all four jobs. The 09:14 London poll again found no final/recovered metrics
   and unchanged decision counts; all four jobs were live with fresh progress: `101996`
   at 42/256, `101998` at 146/256, `102018` at 81/256, and `102019` at 128/256. Runtime
   error counts remained zero for all four jobs. The 09:15 London poll again found no
   final/recovered metrics and unchanged decision counts; all four jobs were live with
   fresh progress: `101996` at 53/256, `101998` at 185/256, `102018` at 93/256, and
   `102019` at 185/256. Runtime error counts remained zero for all four jobs. The 09:16
   London poll again found no final/recovered metrics and unchanged decision counts; all
   four jobs were live with fresh progress: `101996` at 120/256, `101998` at 220/256,
   `102018` at 128/256, and `102019` at 87/144 after completing its previous block.
   Runtime error counts remained zero for all four jobs. The 09:17 London poll again found
   no final/recovered metrics and unchanged decision counts; all four jobs were live with
   fresh progress: `101996` at 161/256, `101998` at 33/74 after completing its previous
   block, `102018` at 164/256, and `102019` at 9/256 after completing its previous block.
   Runtime error counts remained zero for all four jobs. The 09:18 London poll again found
   no final/recovered metrics and unchanged decision counts; all four jobs were live with
   fresh progress: `101996` at 196/256, `101998` at 10/256 after completing its short
   block, `102018` at 169/256, and `102019` at 44/256. Runtime error counts remained zero
   for all four jobs. The 09:19 London poll again found no final/recovered metrics and
   unchanged decision counts; all four jobs were live with fresh progress: `101996` at
   252/256, `101998` at 58/256, `102018` at 190/256, and `102019` at 145/256. Runtime
   error counts remained zero for all four jobs. At 09:21 London, a fresh queue check
   confirmed there were still no user-owned `gh200` jobs. The slow full50 fallback MSC
   jobs `101996` and `101998` were canceled at the user's request to free nodes. The only
   remaining live Path A jobs are the optimized MPP30 pair: constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; both still had no final or recovered
   metrics at the cancellation check. The 09:23 London poll still found no final or
   recovered metrics and decision logs remained at `90`/`90`. `102019` was actively
   processing a 256-prompt block at 96/256 with zero runtime-error signatures. `102018`
   was also live and clean; a focused log check showed it had completed the previous
   depth-1 rollout refresh wave (`43625` parsed generated hypotheses across `3749/4800`
   nonempty refreshes) and had entered the expensive depth-2 belief-refresh phase
   requesting `4800` hypothetical source-support refreshes. The 09:24 London poll again
   found no final or recovered metrics and decision logs still at `90`/`90`. Both jobs
   remained live with fresh stderr progress and zero traceback/runtime/OOM/kill signatures:
   constrained `102018` was 20/256 in its current prompt-processing block, and
   unconstrained `102019` was 166/256 in its current block. The 09:25 London poll again
   found no final or recovered metrics and decision logs still at `90`/`90`. Both jobs
   remained live and clean with zero traceback/runtime/OOM/kill signatures. Constrained
   `102018` had advanced to 82/256 in its current prompt-processing block; unconstrained
   `102019` had just completed a 256-prompt block, and its run log updated at 09:25. The
   09:26 London poll again found no final or recovered metrics and decision logs still at
   `90`/`90`. Both jobs remained live and clean with zero traceback/runtime/OOM/kill
   signatures: constrained `102018` was 89/256 in its current slow prompt-processing
   block, and unconstrained `102019` was 29/256 in a fresh prompt-processing block. The
   09:28 London poll again found no final or recovered metrics and decision logs still at
   `90`/`90`. Both jobs remained live and clean with zero traceback/runtime/OOM/kill
   signatures; constrained `102018` had advanced to 160/256 in its current block, and
   unconstrained `102019` had advanced to 139/256 in its current block. The 09:29 London
   poll again found no final or recovered metrics and decision logs still at `90`/`90`.
   Both jobs remained live and clean with zero traceback/runtime/OOM/kill signatures;
   constrained `102018` had advanced to 169/256 in its current slower block, and
   unconstrained `102019` was nearly through its current block at 233/256. The 09:31
   London poll again found no final or recovered metrics and decision logs still at
   `90`/`90`. Both jobs remained live and clean with zero traceback/runtime/OOM/kill
   signatures; constrained `102018` was nearly through the slow block at 243/256, and
   unconstrained `102019` had started a fresh block after finishing the previous one
   (12/256). The unconstrained run log updated at 09:30. The 09:32 London poll again
   found no final or recovered metrics and decision logs still at `90`/`90`. Both jobs
   remained live and clean with zero traceback/runtime/OOM/kill signatures. Constrained
   `102018` had completed its previous slow block and started a new 145-prompt block
   (0/145), with its run log updated at 09:31; unconstrained `102019` was 101/256 in its
   current block. The 09:33 London poll again found no final or recovered metrics and
   decision logs still at `90`/`90`. Both jobs remained live and clean with zero
   traceback/runtime/OOM/kill signatures. Constrained `102018` had completed the
   145-prompt block and entered a fresh 256-prompt block, with its run log updated at
   09:32; unconstrained `102019` was 194/256 in its current block. The 09:34 London poll
   again found no final or recovered metrics and decision logs still at `90`/`90`. Both
   jobs remained live and clean with zero traceback/runtime/OOM/kill signatures.
   Constrained `102018` was still at the start of its fresh 256-prompt block in the latest
   visible stderr line; unconstrained `102019` had completed its prior block, started
   another fresh 256-prompt block, and updated its run log at 09:34. A 09:37 London
   Slurm check found no user-owned GH200 jobs remaining. The only active jobs were still
   the optimized MPP30 MSC pair: constrained `102018` running on `oat16` for 10:15:30
   and unconstrained `102019` running on `oat21` for 7:35:32. Neither run had final or
   recovered metrics; both decision logs remained at `90` rows. Both jobs were live and
   clean with zero traceback/runtime/OOM/kill signatures. Latest stderr progress was
   constrained `102018` at 85/256 in a prompt block and unconstrained `102019` at
   198/256 in a prompt block; run logs were last updated at 09:32 and 09:34 respectively.
   The 09:39 London poll still found only the two MSC MPP30 jobs active: constrained
   `102018` on `oat16` at 10:16:43 elapsed and unconstrained `102019` on `oat21` at
   7:36:45 elapsed. No final or recovered metrics existed for either run; both decision
   logs remained at `90` rows. Both jobs remained clean with zero traceback/runtime/OOM/
   kill signatures. Constrained `102018` had advanced to 117/256 in its current prompt
   block; unconstrained `102019` had updated its run log at 09:38 and started another
   fresh 256-prompt block. The 09:40 London poll again found no final or recovered
   metrics for either run and both decision logs still at `90` rows. The only active
   jobs were still constrained `102018` on `oat16` at 10:17:59 elapsed and unconstrained
   `102019` on `oat21` at 7:38:01 elapsed. Both remained clean with zero traceback/
   runtime/OOM/kill signatures. Latest stderr progress was constrained `102018` at
   163/256 and unconstrained `102019` at 54/256 in their current prompt blocks. The
   09:41 London poll again found no final or recovered metrics for either run and both
   decision logs still at `90` rows. The active jobs were still constrained `102018`
   on `oat16` at 10:19:04 elapsed and unconstrained `102019` on `oat21` at 7:39:06
   elapsed, with zero traceback/runtime/OOM/kill signatures. Latest stderr progress was
   constrained `102018` at 191/256 and unconstrained `102019` at 147/256 in their
   current prompt blocks. The 09:42 London poll again found no final or recovered
   metrics for either run and both decision logs still at `90` rows. The active jobs
   were still constrained `102018` on `oat16` at 10:20:10 elapsed and unconstrained
   `102019` on `oat21` at 7:40:12 elapsed, with zero traceback/runtime/OOM/kill
   signatures. Both jobs were near the end of their current prompt blocks: constrained
   `102018` at 246/256 and unconstrained `102019` at 239/256. The 09:43 London poll
   still found no final or recovered metrics and both decision logs at `90` rows. Both
   previous prompt blocks completed and both run logs updated (`102018` at 09:42:58,
   `102019` at 09:43:05), but the jobs entered additional prompt-processing waves rather
   than writing metrics: constrained `102018` started a 123-prompt block and unconstrained
   `102019` was 6/256 in a fresh block. Queue state remained constrained `102018` on
   `oat16` at 10:21:16 elapsed and unconstrained `102019` on `oat21` at 7:41:18 elapsed,
   with zero traceback/runtime/OOM/kill signatures. The 09:44 London poll again found no
   final or recovered metrics and both decision logs at `90` rows. Constrained `102018`
   completed the short 123-prompt wave, updated its run log at 09:43:59, and started a
   fresh 256-prompt block; unconstrained `102019` was 57/256 in its current prompt block.
   Queue state remained constrained `102018` on `oat16` at 10:22:22 elapsed and
   unconstrained `102019` on `oat21` at 7:42:24 elapsed, with zero
   traceback/runtime/OOM/kill signatures. The 09:45 London poll again found no final or
   recovered metrics and both decision logs at `90` rows. Queue state remained
   constrained `102018` on `oat16` at 10:23:32 elapsed and unconstrained `102019` on
   `oat21` at 7:43:34 elapsed, with zero traceback/runtime/OOM/kill signatures.
   Unconstrained `102019` advanced to 166/256 in its current prompt block; constrained
   `102018` still showed the start of its fresh 256-prompt block with stderr mtime
   09:44:31. The 09:47 London poll again found no final or recovered metrics and both
   decision logs at `90` rows. Queue state remained constrained `102018` on `oat16` at
   10:24:57 elapsed and unconstrained `102019` on `oat21` at 7:44:59 elapsed, with zero
   traceback/runtime/OOM/kill signatures. Constrained `102018` advanced to 78/256 in its
   current prompt block. Unconstrained `102019` completed the prior block, updated its
   run log at 09:47:00, and started another fresh 256-prompt block. The 09:48 London
   poll again found no final or recovered metrics and both decision logs at `90` rows.
   Queue state remained constrained `102018` on `oat16` at 10:26:06 elapsed and
   unconstrained `102019` on `oat21` at 7:46:08 elapsed, with zero
   traceback/runtime/OOM/kill signatures. Constrained `102018` was 87/256 in its current
   prompt block, and unconstrained `102019` was 23/256 in its current prompt block. The
   09:49 London poll again found no final or recovered metrics and both decision logs at
   `90` rows. Queue state remained constrained `102018` on `oat16` at 10:27:15 elapsed
   and unconstrained `102019` on `oat21` at 7:47:17 elapsed, with zero
   traceback/runtime/OOM/kill signatures. Constrained `102018` was 111/256 in its current
   prompt block, and unconstrained `102019` was 126/256 in its current prompt block. The
   09:50 London poll again found no final or recovered metrics and both decision logs at
   `90` rows. Queue state remained constrained `102018` on `oat16` at 10:28:27 elapsed
   and unconstrained `102019` on `oat21` at 7:48:29 elapsed, with zero
   traceback/runtime/OOM/kill signatures. Constrained `102018` was 164/256 in its current
   prompt block, and unconstrained `102019` was 223/256 in its current prompt block. The
   09:52 London poll again found no final or recovered metrics and both decision logs at
   `90` rows. Queue state remained constrained `102018` on `oat16` at 10:29:42 elapsed
   and unconstrained `102019` on `oat21` at 7:49:44 elapsed, with zero
   traceback/runtime/OOM/kill signatures. Constrained `102018` was 179/256 in its current
   prompt block. Unconstrained `102019` completed its prior block, updated its run log at
   09:51:45, and entered another fresh 256-prompt block. The 09:53 London poll again
   found no final or recovered metrics and both decision logs at `90` rows. Queue state
   remained constrained `102018` on `oat16` at 10:30:59 elapsed and unconstrained `102019`
   on `oat21` at 7:51:01 elapsed, with zero traceback/runtime/OOM/kill signatures.
   Constrained `102018` was 229/256 in its current prompt block, and unconstrained
   `102019` was 58/256 in its current prompt block. At 09:55 London, after the user
   requested canceling the stale/too-slow GH200 path if still live, both the full user
   queue and a GH200-only user queue showed no `hanyal` GH200 jobs, so no `scancel` was
   needed. The only live jobs remained the MSC MPP30 pair `102018` and `102019`. The
   09:56 London poll again found only the MSC MPP30 pair active and confirmed
   `GH200_COUNT 0`. Both runs still had no final or recovered metrics. The decision
   files were `fixed_root_depth_sweep_decisions.jsonl` with 90 rows for both runs
   (constrained mtime 00:01:48, unconstrained mtime 02:18:50), so the visible progress is
   still inside expensive prompt/scoring waves rather than new deployed decisions.
   Constrained `102018` had fresh run-log activity at 09:54:42 and was 2/256 in its latest
   prompt block; unconstrained `102019` had fresh run-log activity at 09:55:59 and was
   14/192 in its latest prompt block. Error counts remained zero for traceback, runtime,
   OOM, and killed signatures. The 09:57 London poll again found only the MSC MPP30 pair
   active, with `GH200_COUNT 0`. Both final and recovered metrics were still missing, and
   both decision logs remained at 90 rows. Queue state was constrained `102018` on
   `oat16` at 10:35:11 elapsed and unconstrained `102019` on `oat21` at 7:55:13 elapsed.
   Constrained `102018` had advanced to 18/256 in its current prompt block; unconstrained
   `102019` had advanced to 49/192. Error counts remained zero for traceback, runtime,
   OOM, and killed signatures. The 09:58 London poll again found only the MSC MPP30 pair
   active, with `GH200_COUNT 0`; no final or recovered metrics existed and both decision
   logs still had 90 rows. Queue state was constrained `102018` on `oat16` at 10:36:05
   elapsed and unconstrained `102019` on `oat21` at 7:56:07 elapsed. Constrained `102018`
   had advanced to 70/256 in its current prompt block; unconstrained `102019` had
   advanced to 137/192. Error counts remained zero for traceback, runtime, OOM, and killed
   signatures. The 09:59 London poll again found only the MSC MPP30 pair active, with
   `GH200_COUNT 0`; `ANY_METRICS 0` and both decision logs still had 90 rows. Queue state
   was constrained `102018` on `oat16` at 10:37:00 elapsed and unconstrained `102019` on
   `oat21` at 7:57:02 elapsed. Constrained `102018` had advanced to 82/256 in its current
   prompt block. Unconstrained `102019` finished its prior 192-prompt wave, updated its
   run log at 09:59:25, and was 38/98 in a new prompt block. Error counts remained zero
   for traceback, runtime, OOM, and killed signatures. The 10:00 London poll again found
   only the MSC MPP30 pair active, with `GH200_COUNT 0`; `ANY_METRICS 0` and both decision
   logs still had 90 rows. Queue state was constrained `102018` on `oat16` at 10:37:58
   elapsed and unconstrained `102019` on `oat21` at 7:58:00 elapsed. Constrained `102018`
   had advanced to 101/256 in its current prompt block. Unconstrained `102019` completed
   the 98-prompt wave, updated its run log at 10:00:06, and had just started another
   256-prompt block. Error counts remained zero for traceback, runtime, OOM, and killed
   signatures. The 10:01 London poll again found only the MSC MPP30 pair active, with
   `GH200_COUNT 0`; `ANY_METRICS 0` and both decision logs still had 90 rows. Queue state
   was constrained `102018` on `oat16` at 10:39:06 elapsed and unconstrained `102019` on
   `oat21` at 7:59:08 elapsed. Constrained `102018` had advanced to 162/256 in its
   current prompt block. Unconstrained `102019` was early in its new 256-prompt block at
   6/256 after the prior run-log update at 10:00:06. Error counts remained zero for
   traceback, runtime, OOM, and killed signatures. The 10:02 London poll again found only
   the MSC MPP30 pair active, with `GH200_COUNT 0`; `ANY_METRICS 0` and both decision logs
   still had 90 rows. Queue state was constrained `102018` on `oat16` at 10:40:05 elapsed
   and unconstrained `102019` on `oat21` at 8:00:07 elapsed. Constrained `102018` had
   moved slowly to 164/256 in its current prompt block. Unconstrained `102019` had advanced
   to 100/256 in its current prompt block. Error counts remained zero for traceback,
   runtime, OOM, and killed signatures.
3. Matched-compute myopic controls: **RUNNING** — included in the same jobs
   (`--include-myopic-controls`).
4. Cost-vs-depth table: script ready (`scripts/cost_vs_depth_table.py`), runs at packaging.

Oracle evidence: `results/constrained_oracle/REPORT.md` (branch-decoy/local-bump env,
greedy 0.56 vs planner 0.15 final RMSE, win rate 0.525 — heavy-tailed wins).
Environment robustness heatmap: **DONE** —
`results/constrained_oracle_robustness/branch_decoy_local_robustness_REPORT.md` and
`plots/constrained_oracle_robustness/branch_decoy_local_robustness_heatmap.png`
(27 cells, 100 trials/cell). The Phase 4 operating point is one of the strongest
mapped cells: planner − greedy final RMSE = -0.175 at lengthscale 0.5, radius 0.5,
noise 0.15; many nearby cells are weak/near-zero, so frame this as a mapped
planning-sensitive regime rather than a universal property of the location family.
Positioning: `results/POSITIONING.md` (COPEx + IPP covered). Operational commands:
`LOCATION_DEPTH_PATH_A_RUNBOOK.md` and `PHASE4_LAUNCH_HANDOFF.md`; both now point the
packaging command at the active MPP30 MSC run names first, with the optimized full50 MSC
pair as the fallback if it finishes first. The Phase 4 endpoint is pre-registered in
`LOCATION_DEPTH_PATH_A_RUNBOOK.md` before metrics landed. `validate_path_a_package.py`
currently passes the ranking-fidelity and oracle checks but fails the expected missing
depth-sweep report, headline plot, and cost-vs-depth table until Phase 4 metrics exist.
Paper env framing: **DONE for the current skeleton** — `paper/main.tex` now presents the
task as a mobile-sensor, movement-cost, finite-range-sensing, branch-decoy environment;
states that geometry selection was method-blind with respect to StrategyEIG; includes the
robustness heatmap figure; and scopes the claim to a mapped planning-sensitive regime.
Paper ranking-fidelity section: **DONE for the current skeleton** — `paper/main.tex`
now reports the 26B A4B Phase 1 gate numbers, includes the aggregate ranking-fidelity
plot, states the positive entropy/truth-log-prob first-link result, and preserves the
near-zero RMSE-rank caveat.
Paper method/protocol section: **DONE for the current skeleton** — `paper/main.tex`
now describes strategy/root generation, analytical rollout scoring, fixed-common paired
scoring controls, the ranking-fidelity deployment diagnostic, and the pre-registered
paired depth-sweep endpoints/myopic controls.
Paper related-work citations: **DONE for the current skeleton** — `paper/main.tex` now
cites DAD, COPEx/constrained BED, and BED-LLM using `paper/references.bib`. The paper
compiled to 5 pages with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`; generated PDF and
auxiliary files were removed from the worktree.
Paper limitations section: **DONE for the current skeleton** — `paper/main.tex` now
covers workshop-scale trial counts, one constrained environment family, constructed
method-blind geometry with robustness heatmap scope evidence, forced-thinking-exit/token
reporting, RMSE as a noisy secondary endpoint, and the missing MPC/action-sequence
ablation as future work. The paper compiled to 5 pages with `pdflatex` twice after this
edit; generated PDF/auxiliary files were removed from the worktree.

RMSE repair analysis: **DONE for current records** —
`results/ranking_fidelity/RMSE_REPAIR.md` and
`results/ranking_fidelity/rmse_repair_analysis.json`. Realized entropy/truth-log-prob
gains are only weakly rank-aligned with realized point-RMSE gains; expected posterior RMSE
cannot be recovered exactly from the current aggregate records because final posterior
supports/probabilities were not logged.
Posterior-state logging for future ranking-fidelity runs: **DONE in code** —
`scripts/strategy_ranking_fidelity.py` now records start expected posterior RMSE,
candidate-level expected posterior RMSE means/drops, and per-deployment final posterior
hypothesis supports/probabilities. `scripts/ranking_fidelity_rmse_repair.py` detects these
future fields and reports expected-posterior-RMSE-drop alignment when available.
Config archive cleanup: **DONE locally** — numbered pilot/smoke configs were moved from
top-level `configs/` into `configs/archive/numbered/`; live top-level configs are now the
descriptively named Path A/ranking/oracle configs plus `configs/cluster_smoke/`.

## NEXT ACTIONS (in order, all local-only, none touch the running jobs)

1. When jobs finish: recovery-or-normal packaging via `PHASE4_LAUNCH_HANDOFF.md`, then
   analysis strictly per the pre-registered section, then results into the skeleton
   following the OUTCOME PLAYBOOK row in GOAL.md that applies.

## OPERATIONAL KNOWLEDGE (repo memory — keep updated here, not in chat)

- Cluster: `ssh oat0`, checkout
  `/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z`.
- GH200 nodes MUST use the Singularity/container launchers
  (`scripts/run_*_gh200_singularity.sh`, container `docker://vllm/vllm-openai:gemma4`);
  the A100/conda path hits an ARM/aarch64 wheel mismatch.
- Partition preference: `gh200` > `msc` > `llm`. Keep ≤8 active jobs. Exclude `oat12`
  (`--exclude=oat12`). Watch oat19 scratch SSD usage (was nearly full once; caches and
  old model families were purged).
- Standard launch env: `BED_LLM_VLLM_KWARGS='{"max_num_seqs":100,"enforce_eager":false}'`,
  `BED_LLM_LOG_REASONING_TRACES=1` (add `BED_LLM_REASONING_TRACE_CHARS=full` for full
  traces); `BED_LLM_SKIP_ENV_SETUP=1` on prepared A100/conda launchers.
- Sync: `python scripts/path_a_sync_commands.py` (explicitly includes the final50
  configs because `.gitignore` ignores `configs/*` — keep that in mind for new configs).
- Models: thinking-enabled `google/gemma-4-26B-A4B-it` for headline evidence; 4B/E4B are
  too weak for spatial strategies (sanity runs only); non-thinking variants are not worth
  spending on. Thinking budget 4096 → ~30% forced-exit rate (12k/41k calls in the final
  sweeps) — first suspect if results are marginal; the one reserved appendix follow-up is
  an 8k-budget replicate of depths {1, 5}.
- Tests: `pytest tests/ -q` must stay green (last known: 404 passed, 1 skipped).
