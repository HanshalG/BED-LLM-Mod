# STATE — living project state (the agent MUST keep this file current)

Maintenance contract: at the end of every working session, update CURRENT STATE and NEXT
ACTIONS below to reflect reality, and append a row to `EXPERIMENTS.md` for anything
launched. A stale STATE.md is a bug — fix it before doing anything else. If this file
conflicts with GOAL.md's design history, this file wins.

## CURRENT STATE (updated 2026-07-09)

Phases 1–3 are DONE; Phase 4 is PAUSED after canceling too-slow cluster jobs; paper skeleton is started in `paper/` and
compiles. The Path A infrastructure is committed locally and tagged `path-a-final-sweep`.
GOAL.md holds the design specs, outcome playbook, and definition of done — consult it for
detail; execute from here.

Status of the Minimum Publishable Package:

1. Ranking-fidelity gate: **PASSED** — `results/ranking_fidelity/PHASE1_26B_A4B_GATE.md`
   (26B A4B, 60 records: entropy ρ 0.38–0.44, truth-log-prob ρ ≈ 0.37 positive at all
   depths, top-1 regret improves with depth 0.33→0.21, RMSE ρ ≈ 0).
2. Headline depth sweep: **PAUSED / NEEDS SPLIT RELAUNCH** — checked 2026-07-09 10:23 London after the user
   asked to cancel the slow 13-hour GH200 jobs to free nodes for other users. Live Slurm
   queue showed zero user-owned jobs on the `gh200` partition, so there were no GH200 job
   IDs to cancel. Only MSC jobs `102018` constrained MPP30 on `oat16` and `102019`
   unconstrained MPP30 on `oat21` were live.
   Follow-up 10:24 London poll: still exactly the same two live MSC jobs and zero
   user-owned `gh200` jobs. No final or recovered metrics exist for either MPP30 run.
   Decision logs remain at `90` rows for both runs. Error signatures remain zero for
   traceback, runtime error, OOM, and killed markers. Latest stderr progress was
   constrained `102018` at 167/256 in its current prompt block, while unconstrained
   `102019` had updated its run log at 10:22:33 and was at 8/256 in a fresh block.
   Follow-up 10:25 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; runtime/OOM/killed/traceback signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 185/256 and unconstrained `102019` at
   101/256 in their current prompt blocks.
   Follow-up 10:26 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Latest stderr progress advanced to constrained `102018`
   at 250/256, nearly through its current prompt block, and unconstrained `102019` at
   155/256.
   Follow-up 10:27 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Constrained `102018` completed the prior 256-prompt
   block, updated its run log at 10:26:21, and entered a short 113-prompt block.
   Unconstrained `102019` advanced to 213/256 in its current prompt block.
   Follow-up 10:28 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Both jobs completed their previous prompt blocks and
   entered fresh 256-prompt blocks: constrained `102018` run log updated at 10:27:16,
   unconstrained `102019` run log updated at 10:28:05.
   Follow-up 10:30 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Latest stderr progress was constrained `102018` at
   11/256 in the fresh block and unconstrained `102019` at 81/256.
   Follow-up 10:31 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Latest stderr progress advanced to constrained `102018`
   at 83/256 and unconstrained `102019` at 121/256 in their current prompt blocks.
   Follow-up 10:32 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Latest stderr progress was constrained `102018` at
   88/256 and unconstrained `102019` at 203/256 in their current prompt blocks.
   Follow-up 10:33 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Latest stderr progress advanced to constrained `102018`
   at 161/256. Unconstrained `102019` completed its prior block, updated its run log at
   10:33:38, and entered a fresh 256-prompt block.
   Follow-up 10:34 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Latest stderr progress was constrained `102018` at
   164/256 and unconstrained `102019` at 1/256 in its fresh prompt block.
   Follow-up 10:35 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Latest stderr progress advanced to constrained `102018`
   at 178/256 and unconstrained `102019` at 100/256 in their current prompt blocks.
   Follow-up 10:36 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Latest stderr progress advanced to constrained `102018`
   at 243/256, nearly through its current block, and unconstrained `102019` at 115/256.
   Follow-up 10:38 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Constrained `102018` completed its previous block,
   updated its run log at 10:37:19, and entered a short 120-prompt block at 4/120.
   Unconstrained `102019` advanced to 200/256.
   Follow-up 10:39 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Constrained `102018` completed the short 120-prompt
   block, updated its run log at 10:38:19, and entered a fresh 256-prompt block.
   Unconstrained `102019` updated its run log at 10:39:12 and was in a short 150-prompt
   block at 133/150.
   Follow-up 10:40 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics. Decision logs remain `90`/`90`; runtime
   error signatures remain zero. Both jobs were in fresh 256-prompt blocks.
   Follow-up 10:42 London cancellation check after the user agreed the slow GH200 jobs
   should be canceled if present: live queue showed zero user-owned `gh200` jobs, so no
   `scancel` was needed. The only live jobs were still MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`.
   Follow-up 10:43 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero for traceback, runtime error, OOM, and
   killed markers. Latest stderr progress was constrained `102018` at 92/256 in its
   current prompt block and unconstrained `102019` at 201/256, near the end of its
   current prompt block.
   Follow-up 10:44 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` advanced slowly
   to 101/256 in its current prompt block. Unconstrained `102019` updated its run log at
   10:44:11 after finishing the prior block and entered a short 144-prompt block.
   Follow-up 10:45 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` advanced to
   163/256 in its current prompt block. Unconstrained `102019` updated its run log at
   10:44:42 after finishing the short block and entered another 256-prompt block.
   Follow-up 10:46 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` inched from
   163/256 to 166/256 in its current block. Unconstrained `102019` was at 10/256 in its
   current 256-prompt block.
   Follow-up 10:47 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` advanced to
   186/256 in its current block. Unconstrained `102019` advanced to 100/256 in its
   current block.
   Follow-up 10:48 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` advanced to
   246/256, nearly through its current prompt block. Unconstrained `102019` advanced to
   141/256 in its current block.
   Follow-up 10:49 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` completed the
   prior block, updated its run log at 10:48:26, and entered a 125-prompt block.
   Unconstrained `102019` advanced to 209/256 in its current block.
   Follow-up 10:50 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Both jobs finished their previous
   prompt blocks, updated run logs (`102018` at 10:49:29, `102019` at 10:50:13), and
   entered fresh 256-prompt blocks.
   Follow-up 10:51 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Fresh prompt blocks started slowly:
   constrained `102018` at 6/256 and unconstrained `102019` at 2/256.
   Follow-up 10:52 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Both fresh prompt blocks accelerated
   after the slow start: constrained `102018` reached 69/256 and unconstrained `102019`
   reached 100/256.
   Follow-up 10:54 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` advanced to
   86/256 in its current block; unconstrained `102019` advanced to 164/256.
   Follow-up 10:55 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` advanced to
   119/256 in its current block. Unconstrained `102019` updated its run log at 10:55:13
   and entered a short 139-prompt block.
   Follow-up 10:56 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` advanced to
   165/256 in its current block. Unconstrained `102019` updated its run log at 10:55:43
   after finishing the short block and entered another 256-prompt block.
   Follow-up 10:58 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` advanced to
   182/256 in its current block, and unconstrained `102019` advanced to 100/256.
   Follow-up 10:59 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Constrained `102018` advanced to
   236/256, nearly through its current block, while unconstrained `102019` advanced to
   140/256.
   Follow-up 11:00 London poll: still the same two live MSC jobs, zero user-owned
   `gh200` jobs, and no final/recovered metrics for either MPP30 run. Decision logs
   remain `90`/`90`; error signatures remain zero. Both run logs advanced: constrained
   `102018` updated at 11:00:33 and was 124/125 through a short block; unconstrained
   `102019` updated at 11:00:38 and entered a 126-prompt block.
   Follow-up 11:04 London poll after the latest GH200 cancellation note: live queue again
   showed zero user-owned `gh200` jobs, so there was nothing to cancel on GH200. The only
   live jobs remain MSC `102018` constrained on `oat16` and MSC `102019` unconstrained on
   `oat21`. No final or recovered metrics exist for either MPP30 run. Decision logs remain
   `90`/`90`; traceback/runtime/OOM/killed signatures remain zero. Latest stderr progress
   was constrained `102018` at 79/256 in its current prompt block and unconstrained
   `102019` at 115/256. Run logs last updated at 11:00:33 and 11:01:06 respectively.
   Follow-up 11:05 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Latest stderr progress was constrained `102018`
   at 89/256 in its current prompt block and unconstrained `102019` at 205/256, while run
   logs last updated at 11:00:33 and 11:01:06 respectively.
   Follow-up 11:06 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Latest stderr progress was constrained `102018`
   at 133/256 in its current prompt block. Unconstrained `102019` updated its run log at
   11:06:35 after finishing a block and entered a fresh 256-prompt block.
   Follow-up 11:08 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Latest stderr progress was constrained `102018`
   at 167/256 in its current prompt block and unconstrained `102019` at 16/256 in the
   fresh prompt block it entered after the 11:06 run-log update.
   Follow-up 11:09 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Latest stderr progress was constrained `102018`
   at 216/256 in its current prompt block and unconstrained `102019` at 103/256.
   Follow-up 11:10 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Constrained `102018` advanced to 255/256 in
   its current prompt block and updated its run log at 11:10:35; unconstrained `102019`
   advanced to 202/256.
   Follow-up 11:12 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Both jobs advanced into short prompt blocks:
   constrained `102018` updated its run log at 11:11:39 and was 118/125; unconstrained
   `102019` updated its run log at 11:12:04 and was 130/145.
   Follow-up 11:13 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Constrained `102018` completed the short block
   and entered a fresh 256-prompt block; unconstrained `102019` also entered a fresh
   256-prompt block and was 19/256.
   Follow-up 11:15 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Latest stderr progress was constrained `102018`
   at 78/256 in its current prompt block and unconstrained `102019` at 115/256.
   Follow-up 11:16 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Latest stderr progress was constrained `102018`
   at 87/256 in its current prompt block and unconstrained `102019` at 233/256, near the
   end of its current block.
   Follow-up 11:18 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Latest stderr progress was constrained `102018`
   at 161/256 in its current prompt block. Unconstrained `102019` updated its run log at
   11:17:35 after finishing a block and entered a fresh 256-prompt block.
   Follow-up 11:20 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, and decision logs remain
   `90`/`90`. Error signatures remain zero. Latest stderr progress was constrained `102018`
   at 178/256 in its current prompt block and unconstrained `102019` at 101/256.
   Follow-up 11:23 London poll after the user noted the slow GH200 jobs should be canceled
   if present: user-owned `gh200` job count was zero, so no `scancel` was needed. The only
   live jobs remained MSC `102018` constrained on `oat16` and MSC `102019` unconstrained on
   `oat21`. No final or recovered metrics exist for either MPP30 run; decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress showed constrained `102018` had just entered a fresh 256-prompt block after a
   run-log update at 11:22:37; unconstrained `102019` was in a short 134-prompt block at
   116/134 with a run-log update at 11:23:02.
   Follow-up 11:24 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` still showed the start of a fresh 256-prompt block after its 11:22:37 run-log
   update; unconstrained `102019` completed the short block, kept updating its run log, and
   entered a fresh 256-prompt block at 1/256.
   Follow-up 11:25 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 16/256 in its current block and
   unconstrained `102019` at 101/256.
   Follow-up 11:26 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 80/256 and unconstrained `102019` at
   136/256 in their current prompt blocks.
   Follow-up 11:27 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 91/256 and unconstrained `102019` at
   209/256 in their current prompt blocks.
   Follow-up 11:28 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 121/256; unconstrained `102019` completed
   its prior block, updated its run log at 11:28:37, and entered a fresh 256-prompt block.
   Follow-up 11:30 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 163/256, while unconstrained `102019` was
   in the slow-start phase of its fresh block at 5/256.
   Follow-up 11:31 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 176/256 and unconstrained `102019` at
   101/256 in their current prompt blocks.
   Follow-up 11:32 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 243/256, nearly through its current block,
   and unconstrained `102019` at 154/256.
   Follow-up 11:33 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` completed the prior block, updated its run log at 11:32:41, and entered a
   short 124-prompt block; unconstrained `102019` advanced to 251/256 in its current block.
   Follow-up 11:35 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Both jobs completed
   their previous prompt blocks, updated run logs (`102018` at 11:33:42, `102019` at
   11:34:04), and entered fresh 256-prompt blocks.
   Follow-up 11:36 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress showed constrained `102018` in the slow early part of its fresh block at
   8/256, while unconstrained `102019` had accelerated to 89/256.
   Follow-up 11:37 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 71/256 and unconstrained `102019` at
   113/256 in their current prompt blocks.
   Follow-up 11:38 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 88/256 and unconstrained `102019` at
   202/256 in their current prompt blocks.
   Follow-up 11:39 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 128/256. Unconstrained `102019` completed
   its prior block, updated its run log at 11:39:37, and entered a 192-prompt block.
   Follow-up 11:41 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 165/256 and unconstrained `102019` at
   13/192 in its current short prompt block.
   Follow-up 11:42 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 188/256 and unconstrained `102019` at
   102/192 in its current short prompt block.
   Follow-up 11:43 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` reached 255/256 in its current block and updated its run log at 11:43:47;
   unconstrained `102019` completed its 192-prompt block and entered a short 98-prompt block.
   Follow-up 11:45 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` completed its previous block, updated its run log at 11:44:45, and entered a
   fresh 256-prompt block; unconstrained `102019` reached the end of a short 47-prompt block.
   Follow-up 11:46 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` still showed the start of its fresh 256-prompt block; unconstrained `102019`
   completed the short block, updated its run log at 11:46:22, and entered a fresh 256-prompt block.
   Follow-up 11:48 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 77/256 and unconstrained `102019` at
   60/256 in their current prompt blocks.
   Follow-up 11:49 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 84/256 and unconstrained `102019` at
   148/256 in their current prompt blocks.
   Follow-up 11:51 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 128/256. Unconstrained `102019` completed
   its prior block, updated its run log at 11:51:15, and entered a 114-prompt block.
   Follow-up 11:52 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 168/256. Unconstrained `102019` completed
   the 114-prompt block, updated its run log at 11:51:40, and entered a fresh 256-prompt block.
   Follow-up 11:54 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 224/256, near the end of its current block,
   and unconstrained `102019` at 102/256.
   Follow-up 11:56 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
   completed its prior block, updated its run log at 11:55:59, and was 105/134 through a
   short prompt block; unconstrained `102019` advanced to 208/256.
   Follow-up 11:57 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Both jobs completed
   their previous prompt blocks, updated run logs (`102018` at 11:55:59, `102019` at
   11:56:57), and entered fresh 256-prompt blocks.
   Follow-up 11:59 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 77/256 and unconstrained `102019` at
   100/256 in their current fresh prompt blocks.
   Follow-up 12:01 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 88/256 and unconstrained `102019` at
   182/256 in their current prompt blocks.
   Follow-up 12:03 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 165/256. Unconstrained `102019` crossed
   10 hours elapsed, updated its run log at 12:02:10, and entered a fresh 256-prompt block.
   Follow-up 12:04 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 201/256 and unconstrained `102019` at
   103/256 in their current prompt blocks.
   Follow-up 12:06 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
   completed a block, updated its run log at 12:05:55, and entered a short 128-prompt block;
   unconstrained `102019` advanced to 229/256, near the end of its current block.
   Follow-up 12:08 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Both jobs completed
   their previous blocks and entered fresh 256-prompt blocks; constrained `102018` updated
   its run log at 12:06:57, and unconstrained `102019` was at 5/256.
   Follow-up 12:10 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 77/256 and unconstrained `102019` at
   139/256 in their current prompt blocks.
   Follow-up 12:12 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 100/256; unconstrained `102019` reached
   the end of a short 134-prompt block and updated its run log at 12:12:56.
   Follow-up 12:14 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 169/256; unconstrained `102019` restarted
   after the short block and was at 73/256 in a fresh prompt block.
   Follow-up 12:17 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 252/256, near the end of its current block,
   and unconstrained `102019` at 202/256.
   Follow-up 12:19 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Both jobs completed
   their previous prompt blocks, updated run logs (`102018` at 12:17:59, `102019` at
   12:18:16), and entered fresh 256-prompt blocks.
   Follow-up 12:21 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 76/256 and unconstrained `102019` at
   127/256 in their current fresh prompt blocks.
   Follow-up 12:24 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 110/256; unconstrained `102019` completed
   its previous block, updated its run log at 12:23:35, and entered a fresh 256-prompt block.
   Follow-up 12:25 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 163/256 and unconstrained `102019` at
   51/256 in its fresh prompt block.
   Follow-up 12:27 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 226/256 and unconstrained `102019` at
   173/256 in their current prompt blocks.
   Follow-up 12:29 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
   entered a short 126-prompt block and was 110/126 through it; unconstrained `102019`
   completed a block, updated its run log at 12:28:52, and entered a fresh 256-prompt block.
   Follow-up 12:31 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
   completed the short 126-prompt block and entered a 192-prompt block; unconstrained
   `102019` advanced to 47/256 in its current prompt block.
   Follow-up 12:32 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest stderr
   progress advanced to constrained `102018` at 78/192 and unconstrained `102019` at
   202/256 in their current prompt blocks.
   Follow-up 12:35 London cancellation check after the user said to cancel the stale
   13-hour GH200 jobs if they were occupying nodes: live queue again showed zero
   user-owned jobs on the `gh200` partition, so there were no job IDs to `scancel`. The
   only live jobs remained MSC `102018` constrained on `oat16` and MSC `102019`
   unconstrained on `oat21`; both were left untouched.
   Follow-up 12:36 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero.
   Constrained `102018` advanced to 166/192 in its current prompt block, while
   unconstrained `102019` updated its run log at 12:34:16 and was 39/256 in a fresh
   prompt block.
   Follow-up 12:37 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero.
   Constrained `102018` is near the end of its current block at 187/192 with its run log
   updating at 12:37:01; unconstrained `102019` advanced to 102/256 in its current block.
   Follow-up 12:37:52 London poll: still exactly two live jobs, MSC `102018` constrained
   on `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` completed the prior 192-prompt block, updated its run log at 12:37:45, and
   entered a short 82-prompt block; unconstrained `102019` advanced to 153/256 in its
   current block.
   Follow-up 12:38 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` had rolled into another prompt block showing 0/133 in stderr; unconstrained
   `102019` advanced to 212/256 in its current block.
   Follow-up 12:39 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` still showed the current 133-prompt block at its start, while unconstrained
   `102019` completed the prior 256-prompt block, updated its run log at 12:39:33, and
   entered a fresh 256-prompt block.
   Follow-up 12:40 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 23/133 in its current prompt block; unconstrained `102019` was
   just starting its fresh 256-prompt block at 3/256.
   Follow-up 12:41 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 79/133 in its current prompt block; unconstrained `102019`
   advanced to 51/256 in its current block.
   Follow-up 12:42 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 99/133 in its current prompt block; unconstrained `102019`
   advanced to 112/256 in its current block.
   Follow-up 12:43 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` updated its run log at 12:43:13 and advanced to 117/133 in its current block;
   unconstrained `102019` advanced to 201/256 in its current block.
   Follow-up 12:44 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` updated its run log at 12:43:44 and rolled into a short 8-prompt block;
   unconstrained `102019` updated its run log at 12:44:26 and rolled into a 124-prompt
   block.
   Follow-up 12:45 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Both jobs
   advanced into fresh 256-prompt blocks, with constrained `102018` run log updated at
   12:45:01 and unconstrained `102019` run log updated at 12:44:53.
   Follow-up 12:46 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 11/256 in its fresh prompt block; unconstrained `102019` advanced
   to 33/256 in its fresh prompt block.
   Follow-up 12:47 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 35/256 in its current block; unconstrained `102019` advanced to
   103/256 in its current block.
   Follow-up 12:49 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 93/256 in its current block; unconstrained `102019` advanced to
   202/256 in its current block.
   Follow-up 12:50 London poll: still exactly two live jobs, MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; user-owned `gh200` job count
   remains zero. No final or recovered metrics exist for either MPP30 run, decision logs
   remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 108/256 in its current block; unconstrained `102019` updated its
   run log at 12:50:14 and was 88/125 through a short prompt block.
   Follow-up 12:52 London poll: user-owned `gh200` job count is still zero, so there were
   no GH200 jobs to cancel after the user confirmed the slow 13-hour GH200 path was not
   worth holding. The only live jobs remain MSC `102018` constrained on `oat16` and MSC
   `102019` unconstrained on `oat21`. No final or recovered metrics exist for either
   MPP30 run, decision logs remain `90`/`90`, and traceback/runtime/OOM/killed signatures
   remain zero. Constrained `102018` advanced to 208/256 in its current block;
   unconstrained `102019` was at 100/256 in a fresh 256-prompt block.
   Follow-up 12:53 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` updated its run log at 12:53:30 and reached 242/256 in its current block;
   unconstrained `102019` advanced to 180/256 in its current block.
   Follow-up 12:54 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` completed its prior prompt block, updated its run log at 12:54:13, and
   entered a fresh 256-prompt block; unconstrained `102019` advanced to 208/256 in its
   current block.
   Follow-up 12:55 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 22/256 in its fresh prompt block; unconstrained `102019`
   completed its prior block, updated its run log at 12:55:28, and entered a fresh
   256-prompt block.
   Follow-up 12:56 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 48/256 in its fresh prompt block; unconstrained `102019` was
   still at the start of its fresh 256-prompt block after updating at 12:55:28.
   Follow-up 12:57 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 63/256 in its current prompt block; unconstrained `102019`
   advanced to 49/256 in its current prompt block.
   Follow-up 12:58 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 118/256 in its current prompt block; unconstrained `102019`
   advanced to 103/256 in its current prompt block.
   Follow-up 12:59 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 141/256 in its current prompt block; unconstrained `102019`
   advanced to 145/256 in its current prompt block.
   Follow-up 12:59:54 London poll: still exactly two live MSC jobs, constrained `102018`
   on `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 166/256 in its current prompt block; unconstrained `102019`
   advanced to 206/256 in its current prompt block.
   Follow-up 13:00 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 220/256 in its current prompt block; unconstrained `102019`
   updated its run log at 13:00:53 and advanced to 118/135 in a short prompt block.
   Follow-up 13:01 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` updated its run log at 13:01:44 and advanced to 235/256 in its current prompt
   block; unconstrained `102019` completed its short block and entered a fresh 256-prompt
   block.
   Follow-up 13:02 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` completed its prior prompt block, updated its run log at 13:02:17, and
   entered a fresh 256-prompt block; unconstrained `102019` advanced to 55/256 in its
   current prompt block.
   Follow-up 13:04 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 30/256 in its current prompt block; unconstrained `102019`
   advanced to 114/256 in its current prompt block.
   Follow-up 13:05 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 76/256 in its current prompt block; unconstrained `102019`
   advanced to 201/256 in its current prompt block.
   Follow-up 13:05:56 London poll: still exactly two live MSC jobs, constrained `102018`
   on `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 91/256 in its current prompt block; unconstrained `102019`
   updated its run log at 13:05:44 and entered a short 112-prompt block.
   Follow-up 13:06 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 133/256 in its current prompt block; unconstrained `102019`
   completed its short block, updated its run log at 13:06:07, and entered a fresh
   256-prompt block.
   Follow-up 13:07 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 145/256 in its current prompt block; unconstrained `102019`
   advanced to 21/256 in its current prompt block.
   Follow-up 13:08 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 187/256 in its current prompt block; unconstrained `102019`
   advanced to 100/256 in its current prompt block.
   Follow-up 13:09 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` updated its run log at 13:09:48 and advanced to 250/256 in its current prompt
   block; unconstrained `102019` advanced to 172/256 in its current prompt block.
   Follow-up 13:12 London cancellation check after the user confirmed the 13-hour GH200
   jobs should be canceled to free nodes: live queue showed zero user-owned `gh200` jobs,
   so no `scancel` was needed. The only live jobs were still MSC `102018` constrained on
   `oat16` and MSC `102019` unconstrained on `oat21`; these were left running.
   Follow-up 13:13 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` has entered a fresh 256-prompt block and was at 63/256, with the run log last
   updated at 13:10:20. Unconstrained `102019` has also entered a fresh 256-prompt block
   and was at 47/256, with the run log last updated at 13:11:34.
   Follow-up 13:14 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 122/256 in its current prompt block. Unconstrained `102019`
   advanced to 101/256 in its current prompt block.
   Follow-up 13:15 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 141/256 in its current prompt block. Unconstrained `102019`
   advanced to 166/256 in its current prompt block.
   Follow-up 13:16 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 154/256 in its current prompt block. Unconstrained `102019`
   advanced to 214/256 in its current prompt block.
   Follow-up 13:17 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 212/256 in its current prompt block. Unconstrained `102019`
   completed its previous block, updated its run log at 13:16:52, and entered a fresh
   256-prompt block.
   Follow-up 13:18 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 227/256 in its current prompt block. Unconstrained `102019`
   remained at the start of the fresh 256-prompt block it entered after the 13:16:52
   run-log update.
   Follow-up 13:19 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` completed its previous block, updated its run log at 13:18:52, and entered a
   short 93-prompt block at 8/93. Unconstrained `102019` advanced to 58/256 in its
   current prompt block.
   Follow-up 13:20 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Latest constrained
   `102018` stderr progress showed a 256-prompt block at 22/256, with run log last
   updated at 13:18:52. Unconstrained `102019` advanced to 109/256 in its current prompt
   block.
   Follow-up 13:21 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 39/256 in its current prompt block. Unconstrained `102019`
   advanced to 202/256 in its current prompt block.
   Follow-up 13:22 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 80/256 in its current prompt block. Unconstrained `102019`
   completed its previous block, updated its run log at 13:21:40, and entered a short
   97-prompt block.
   Follow-up 13:23 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 126/256 in its current prompt block. Unconstrained `102019`
   completed its short block, updated its run log at 13:22:02, and entered a 192-prompt
   block.
   Follow-up 13:24 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 135/256 in its current prompt block. Unconstrained `102019`
   advanced to 35/192 in its current prompt block.
   Follow-up 13:25 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 207/256 in its current prompt block. Unconstrained `102019`
   advanced to 117/192 in its current prompt block.
   Follow-up 13:26 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` advanced to 231/256 in its current prompt block. Unconstrained `102019`
   completed its previous block, updated its run log at 13:25:48, and entered a
   121-prompt block.
   Follow-up 13:27 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` completed its previous 256-prompt block, updated its run log at 13:26:54, and
   entered a short 72-prompt block at 21/72. Unconstrained `102019` was at 2/121 in its
   current prompt block.
   Follow-up 13:28 London poll: still exactly two live MSC jobs, constrained `102018` on
   `oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains
   zero. No final or recovered metrics exist for either MPP30 run, decision logs remain
   `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
   `102018` completed its short block and entered a fresh 256-prompt block. Unconstrained
   `102019` advanced to 51/121 in its current prompt block.
   Earlier checks: checked 2026-07-09 09:05 London after the user
   requested canceling the slow/low-yield GH200 path to free nodes for other users. GH200
   jobs `101993` and `101994` were already absent from the live queue; direct `gh200`
   partition checks through 09:05 London showed no user-owned GH200 jobs, so no `scancel`
   was needed. The 09:05 live queue also had no `gh200` entries for this user. A fresh
   10:06 London cancellation check again found zero user-owned `gh200` jobs, so no
   cancellation was performed; only MSC jobs `102018` and `102019` were live. The 10:08
   London poll again found no final or recovered metrics, no user-owned `gh200` jobs, and
   only the same two MSC jobs live: constrained `102018` on `oat16` at 10:45:32 elapsed
   and unconstrained `102019` on `oat21` at 8:05:34 elapsed. Decision logs remained at
   `90` rows for both runs. Runtime error signatures were still zero for traceback,
   runtime error, OOM, and killed markers. Latest stderr progress was constrained `102018`
   at 10/256 in a prompt block and unconstrained `102019` at 100/256 in a prompt block.
   The 10:09 London poll was unchanged at the artifact level: no final/recovered metrics,
   decision logs still `90`/`90`, no `gh200` jobs, and only the same MSC pair live
   (`102018` at 10:46:33 elapsed on `oat16`, `102019` at 8:06:35 elapsed on `oat21`).
   Error signatures remained zero; latest stderr progress was constrained `102018` at
   25/256 and unconstrained `102019` at 111/256 in their current prompt blocks. The 10:10
   London poll again found no final/recovered metrics, decision logs still `90`/`90`, no
   `gh200` jobs, and only the same MSC pair live (`102018` at 10:47:29 elapsed on `oat16`,
   `102019` at 8:07:31 elapsed on `oat21`). Error signatures remained zero. Latest
   stderr progress was constrained `102018` at 83/256 and unconstrained `102019` at
   161/256 in their current prompt blocks. The 10:11 London poll again found no
   final/recovered metrics, decision logs still `90`/`90`, no `gh200` jobs, and only the
   same MSC pair live (`102018` at 10:48:22 elapsed on `oat16`, `102019` at 8:08:24
   elapsed on `oat21`). Error signatures remained zero. Latest stderr progress was
   constrained `102018` at 88/256; unconstrained `102019` completed its previous prompt
   block, updated its run log at 10:10:44, and entered a shorter 156-prompt block. The
   10:12 London poll again found no final/recovered metrics, decision logs still `90`/`90`,
   no `gh200` jobs, and only the same MSC pair live (`102018` at 10:49:15 elapsed on
   `oat16`, `102019` at 8:09:17 elapsed on `oat21`). Error signatures remained zero.
   Latest stderr progress was constrained `102018` at 106/256; unconstrained `102019`
   completed the short block, updated its run log at 10:11:17, and entered another
   256-prompt block. The 10:13 London poll again found no final/recovered metrics,
   decision logs still `90`/`90`, no `gh200` jobs, and only the same MSC pair live
   (`102018` at 10:50:09 elapsed on `oat16`, `102019` at 8:10:11 elapsed on `oat21`).
   Error signatures remained zero. Latest stderr progress was constrained `102018` at
   167/256; unconstrained `102019` was at 1/256 in its fresh prompt block. The 10:13:27
   London poll again found no final/recovered metrics, decision logs still `90`/`90`, no
   `gh200` jobs, and only the same MSC pair live (`102018` at 10:51:00 elapsed on `oat16`,
   `102019` at 8:11:02 elapsed on `oat21`). Error signatures remained zero. Latest
   stderr progress was constrained `102018` at 169/256 and unconstrained `102019` at
   100/256 in their current prompt blocks. The 10:14 London poll again found no
   final/recovered metrics, decision logs still `90`/`90`, no `gh200` jobs, and only the
   same MSC pair live (`102018` at 10:51:59 elapsed on `oat16`, `102019` at 8:12:01
   elapsed on `oat21`). Error signatures remained zero. Latest stderr progress was
   constrained `102018` at 188/256 and unconstrained `102019` at 102/256 in their current
   prompt blocks. The 10:15 London poll again found no final/recovered metrics, decision
   logs still `90`/`90`, no `gh200` jobs, and only the same MSC pair live (`102018` at
   10:52:52 elapsed on `oat16`, `102019` at 8:12:54 elapsed on `oat21`). Error signatures
   remained zero. Latest stderr progress was constrained `102018` at 220/256 and
   unconstrained `102019` at 187/256 in their current prompt blocks. The 10:16 London
   poll again found no final/recovered metrics, decision logs still `90`/`90`, no `gh200`
   jobs, and only the same MSC pair live (`102018` at 10:53:46 elapsed on `oat16`,
   `102019` at 8:13:48 elapsed on `oat21`). Error signatures remained zero. Constrained
   `102018` completed the prior block, updated its run log at 10:15:33, and entered a
   113-prompt block; unconstrained `102019` was near the end of its prompt block at
   231/256. The 10:17 London poll again found no final/recovered metrics, decision logs
   still `90`/`90`, no `gh200` jobs, and only the same MSC pair live (`102018` at
   10:54:42 elapsed on `oat16`, `102019` at 8:14:44 elapsed on `oat21`). Error signatures
   remained zero. Both jobs completed their previous prompt blocks, updated run logs
   (`102018` at 10:16:30, `102019` at 10:16:51), and entered fresh 256-prompt blocks. The
   10:18 London poll again found no final/recovered metrics, decision logs still `90`/`90`,
   no `gh200` jobs, and only the same MSC pair live (`102018` at 10:55:53 elapsed on
   `oat16`, `102019` at 8:15:55 elapsed on `oat21`). Error signatures remained zero.
   Latest stderr progress was constrained `102018` at 0/256 and unconstrained `102019` at
   6/256 in their fresh prompt blocks. The 10:19 London poll again found no final/recovered
   metrics, decision logs still `90`/`90`, no `gh200` jobs, and only the same MSC pair live
   (`102018` at 10:56:47 elapsed on `oat16`, `102019` at 8:16:49 elapsed on `oat21`).
   Error signatures remained zero. Latest stderr progress was constrained `102018` at
   12/256 and unconstrained `102019` at 100/256 in their current prompt blocks. The
   10:20 London poll again found no final/recovered metrics, decision logs still `90`/`90`,
   no `gh200` jobs, and only the same MSC pair live (`102018` at 10:57:50 elapsed on
   `oat16`, `102019` at 8:17:52 elapsed on `oat21`). Error signatures remained zero.
   Latest stderr progress was constrained `102018` at 78/256 and unconstrained `102019`
   at 117/256 in their current prompt blocks. The 10:21 London poll again found no
   final/recovered metrics, decision logs still `90`/`90`, no `gh200` jobs, and only the
   same MSC pair live (`102018` at 10:58:56 elapsed on `oat16`, `102019` at 8:18:58
   elapsed on `oat21`). Error signatures remained zero. Latest stderr progress was
   constrained `102018` at 88/256 and unconstrained `102019` at 202/256 in their current
   prompt blocks.
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
   runtime, OOM, and killed signatures. The 10:03 London poll again found only the MSC
   MPP30 pair active, with `GH200_COUNT 0`; `ANY_METRICS 0` and both decision logs still
   had 90 rows. Queue state was constrained `102018` on `oat16` at 10:41:05 elapsed and
   unconstrained `102019` on `oat21` at 8:01:07 elapsed. Constrained `102018` had advanced
   to 185/256 in its current prompt block. Unconstrained `102019` had advanced to 120/256
   in its current prompt block. Error counts remained zero for traceback, runtime, OOM,
   and killed signatures. The 10:04 London poll again found only the MSC MPP30 pair active,
   with `GH200_COUNT 0`; `ANY_METRICS 0` and both decision logs still had 90 rows. Queue
   state was constrained `102018` on `oat16` at 10:42:12 elapsed and unconstrained
   `102019` on `oat21` at 8:02:14 elapsed. Constrained `102018` was near the end of its
   current prompt block at 245/256. Unconstrained `102019` had advanced to 203/256 in its
   current prompt block. Error counts remained zero for traceback, runtime, OOM, and
   killed signatures.
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
Cluster courtesy check 2026-07-09 13:29 London: user asked to cancel the slow 13-hour
GH200 jobs if present because other people were waiting on those nodes. Fresh `squeue`
showed zero user-owned `gh200` jobs, so no `scancel` was issued. The only live jobs were
MSC MPP30 `102018` constrained on `oat16` and `102019` unconstrained on `oat21`.
Follow-up 13:30 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, and decision logs remain
`90`/`90`. Error signatures remain zero for traceback, runtime error, OOM, and killed
markers. Latest stderr progress was constrained `102018` at 112/256 in its current
prompt block and unconstrained `102019` at 102/256 in its current prompt block.
Follow-up 13:31 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Latest stderr progress advanced
to constrained `102018` at 136/256 and unconstrained `102019` at 195/256 in their current
prompt blocks.
Follow-up 13:32 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Latest stderr progress advanced
to constrained `102018` at 169/256; unconstrained `102019` finished its previous block,
updated its run log at 13:32:28, and entered a short 69-prompt block.
Follow-up 13:33 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Latest stderr progress advanced
to constrained `102018` at 187/256; unconstrained `102019` finished the short block,
updated its run log at 13:32:37, and was at 22/256 in another prompt block.
Follow-up 13:34 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Latest stderr progress advanced
to constrained `102018` at 252/256, nearly through its current block, and unconstrained
`102019` at 78/256.
Follow-up 13:35 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018` finished
the prior 256-prompt block, updated its run log at 13:34:47, and was at 31/75 in a short
prompt block. Unconstrained `102019` advanced to 173/256 in its current prompt block.
Follow-up 13:36 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018` moved into
another 256-prompt block and was at 34/256. Unconstrained `102019` finished its previous
block, updated its run log at 13:35:52, and entered a short 76-prompt block.
Follow-up 13:37 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018` advanced
to 60/256 in its current prompt block. Unconstrained `102019` updated its run log at
13:36:03 and advanced to 53/256 in its current prompt block.
Follow-up 13:38 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018` advanced
to 84/256 in its current prompt block, and unconstrained `102019` advanced to 144/256 in
its current prompt block.
Follow-up 13:39 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018` advanced
to 103/256 in its current prompt block, and unconstrained `102019` advanced to 237/256,
near the end of its current prompt block.
Follow-up 13:40 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018` advanced
to 161/256 in its current prompt block. Unconstrained `102019` updated its run log at
13:38:58 after finishing a block and had started another 256-prompt block.
Follow-up 13:40 London poll #2: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018` advanced
to 183/256 in its current prompt block, and unconstrained `102019` was at 46/256 in its
newer prompt block.
Follow-up 13:41 London poll: still exactly two live MSC jobs, constrained `102018` on
`oat16` and unconstrained `102019` on `oat21`; user-owned `gh200` job count remains zero.
No final or recovered metrics exist for either MPP30 run, decision logs remain `90`/`90`,
and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018` advanced
to 214/256, nearing the end of its current prompt block, and unconstrained `102019`
advanced to 114/256.
Follow-up 13:43 London cancellation check: after the user agreed the slow 13-hour GH200
jobs should be canceled to free shared nodes, `squeue -p gh200` returned no user-owned
GH200 job IDs before `scancel`, so there was nothing active to cancel. Post-check queue
still showed only the two MSC jobs: constrained `102018` on `oat16` at 14:20 elapsed and
unconstrained `102019` on `oat21` at 11:40 elapsed.
Follow-up 13:44 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
updated its run log at 13:42:17 and was at 38/256 in a fresh prompt block; unconstrained
`102019` updated its run log at 13:42:58 and was at 90/256 in its current prompt block.
Follow-up 13:45 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
remained in the same prompt block at 48/256; unconstrained `102019` advanced to 177/256
in its current prompt block.
Follow-up 13:46 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 118/256 in its current prompt block. Unconstrained `102019` updated its run
log at 13:46:18 after finishing a block and had just started a fresh 256-prompt block.
Follow-up 13:47 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 132/256 in its current prompt block; unconstrained `102019` advanced to
50/256 in its fresh prompt block.
Follow-up 13:48 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 166/256 in its current prompt block; unconstrained `102019` advanced to
88/256 in its current prompt block.
Follow-up 13:49 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 193/256 in its current prompt block; unconstrained `102019` advanced to
221/256 in its current prompt block.
Follow-up 13:49 London poll #2: still exactly two live MSC jobs and zero user-owned
`gh200` jobs. No final or recovered metrics exist for either MPP30 run, decision logs
remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
`102018` advanced to 206/256 in its current prompt block. Unconstrained `102019` updated
its run log at 13:49:50 after finishing a block and was at 53/76 in a short prompt block.
Follow-up 13:50 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
updated its run log at 13:50:40 and advanced to 249/256, nearly through its current
prompt block. Unconstrained `102019` had moved into a fresh 256-prompt block at 26/256.
Follow-up 13:51 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
updated its run log at 13:51:18 after completing the previous block and had just started
a fresh 256-prompt block. Unconstrained `102019` advanced to 130/256 in its current block.
Follow-up 13:52 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 15/256 in its fresh prompt block. Unconstrained `102019` advanced to 213/256
in its current prompt block.
Follow-up 13:53 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 29/256 in its fresh prompt block. Unconstrained `102019` updated its run log
at 13:53:01 after completing the previous block and was at 30/256 in a fresh block.
Follow-up 13:54 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 71/256 in its fresh prompt block; unconstrained `102019` advanced to 68/256
in its fresh prompt block.
Follow-up 13:55 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 111/256 in its current prompt block; unconstrained `102019` advanced to
146/256 in its current prompt block.
Follow-up 13:55 London poll #2: still exactly two live MSC jobs and zero user-owned
`gh200` jobs. No final or recovered metrics exist for either MPP30 run, decision logs
remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
`102018` advanced to 142/256 in its current prompt block; unconstrained `102019`
advanced to 231/256, nearing the end of its current prompt block.
Follow-up 13:57 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 184/256 in its current prompt block. Unconstrained `102019` updated its run
log at 13:56:29 after finishing a block and was at 33/256 in a fresh prompt block.
Follow-up 13:58 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 233/256, nearing the end of its current prompt block. Unconstrained `102019`
advanced to 90/256 in its fresh prompt block.
Follow-up 13:58 London poll #2: still exactly two live MSC jobs and zero user-owned
`gh200` jobs. No final or recovered metrics exist for either MPP30 run, decision logs
remain `90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained
`102018` updated its run log at 13:58:54 after completing a short 75-prompt block.
Unconstrained `102019` advanced to 195/256 in its current prompt block.
Follow-up 13:59 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
was at 17/256 in a fresh prompt block. Unconstrained `102019` updated its run log at
13:59:50 after finishing a block and had just started a fresh 256-prompt block.
Follow-up 14:01 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 41/256 in its current prompt block; unconstrained `102019` advanced to
62/256 in its current prompt block.
Follow-up 14:02 London poll: still exactly two live MSC jobs and zero user-owned `gh200`
jobs. No final or recovered metrics exist for either MPP30 run, decision logs remain
`90`/`90`, and traceback/runtime/OOM/killed signatures remain zero. Constrained `102018`
advanced to 57/256 in its current prompt block; unconstrained `102019` advanced to
167/256 in its current prompt block.
Follow-up 14:24 London cancellation: user requested freeing the effectively-too-slow
long-running jobs. There were zero user-owned `gh200` jobs at cancellation time; the only
active jobs were MSC `102018` (`loc_branch_constr26_mpp30` on `oat16`) and `102019`
(`loc_branch_uncon26_mpp30` on `oat21`). Both were canceled with `scancel`, and a
post-cancel `squeue -u hanyal` check returned no jobs.
Follow-up 14:45 London local recovery work: added split-trial support for
`scripts/location_fixed_root_depth_sweep.py` via `--trial-offset`, preserving paired RNG
by replaying skipped trial draws; added `scripts/combine_location_fixed_root_depth_sweeps.py`
to merge completed split blocks into the single metrics JSON expected by the Path A
package builder; updated `scripts/path_a_launch_commands.py` to print block-specific
`--num-trials`/`--trial-offset` commands. Full local suite passed
(`449 passed, 1 skipped`).
Follow-up 15:02 London split-RNG correction: the split commands now include
`--total-trials 30` so every 10-trial MPP30 block draws the same full 30-trial
observation-noise array before replaying hidden-state/branch RNG. This preserves pairing
relative to the intended monolithic 30-trial run instead of only replaying through the
end of the current block. Focused split tests and the full local suite pass
(`449 passed, 1 skipped`); non-mutating `squeue -u hanyal` check returned no jobs.
Follow-up 15:08 London split-RNG hardening: factored fixed-root sweep RNG setup into a
small `_make_depth_sweep_rng_plan` helper and added a regression test proving that MPP30
split block 10..19 with `--total-trials 30` matches the monolithic 30-trial stream for
observation noise, hidden-state RNG draws, and branch seeds. Full local suite passes
(`450 passed, 1 skipped`); non-mutating `squeue -u hanyal` still returned no jobs.
Follow-up 15:17 London sync/readiness hardening: `scripts/path_a_sync_commands.py` now
forces the split-run/packaging scripts into the rsync list even from a clean committed
tree, and `scripts/path_a_remote_readiness.py` now requires the split combiner. Read-only
remote readiness currently reports zero active jobs and an idle `gh200` node, but
`ok_to_launch=false` because the remote checkout is missing
`scripts/combine_location_fixed_root_depth_sweeps.py`; sync is required before any
split relaunch. Full local suite passes (`450 passed, 1 skipped`).
Follow-up 15:25 London split command ergonomics:
`python scripts/path_a_launch_commands.py --split-mpp30` now prints the complete split
MPP30 workflow in one non-mutating command set: six GH200 `sbatch` commands, two combiner
commands, and the final
`build_path_a_package.py` command. `PHASE4_LAUNCH_HANDOFF.md` now points to this as the
canonical command source. Full local suite passes (`451 passed, 1 skipped`). Read-only
remote readiness is unchanged: zero active jobs, one idle `gh200` node, and
`ok_to_launch=false` only because the combiner has not been synced to the cluster yet.
Follow-up 15:31 London preflight alignment: `scripts/path_a_preflight.py` now validates
the split-MPP30 workflow directly, including the local combiner script and six-command
split launch shape. Local preflight passes its launch-readiness checks and still reports
`package_validation_ok=false`, expected until sweep results exist. Full local suite
passes (`452 passed, 1 skipped`). Read-only remote readiness remains unchanged: zero
active jobs, one idle `gh200` node, and `ok_to_launch=false` only because the combiner
has not been synced to the cluster yet.
Follow-up 15:38 London remote-readiness cap fix: remote readiness now shares the forced
sync manifest and enforces the 8-job cap for a six-job split-MPP30 launch, so it only
allows launch when at most two jobs are already active. Read-only remote readiness
currently reports zero active jobs and one idle `gh200` node, but `ok_to_launch=false`
because the cluster checkout is missing
`scripts/combine_location_fixed_root_depth_sweeps.py` and
`scripts/recover_depth_sweep_metrics.py`. Full local suite passes
(`453 passed, 1 skipped`).
Follow-up 15:43 London handoff cleanup: `PHASE4_LAUNCH_HANDOFF.md` now monitors the six
split-MPP30 block run directories instead of the canceled monolithic MSC run names, and
`scripts/path_a_sync_commands.py` now describes its output as git-status paths plus
required Path A launch/package files. Full local suite still passes
(`453 passed, 1 skipped`). Read-only remote readiness remains unchanged: zero active
jobs, one idle `gh200` node, and `ok_to_launch=false` until the missing scripts are
synced.

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
Follow-up 16:02 London queue check: user requested canceling the slow 13-hour GH200 jobs
because others were waiting, but `squeue -u hanyal` returned no active or pending jobs;
there was nothing left to cancel.
Follow-up 14:48 London queue check: user again agreed the too-slow GH200 jobs should be
canceled if they were still alive. A fresh `squeue -u hanyal` returned an empty queue
header only: no running or pending user-owned jobs on `gh200`, `msc`, or any other
partition, so no `scancel` was issued.
Follow-up 16:12 London paper progress: filled the draft cost-vs-depth section with the
pre-registered Path A per-decision scaling for `B=10` candidate roots and `R=16`
rollouts (brute-force depth 5 proxy `11,111` candidate sets / `100,000` leaves vs
StrategyEIG `800` simulated rollout steps per decision). `pdflatex`, `bibtex`, and two
final `pdflatex` passes completed successfully; TeX build byproducts were removed.
Follow-up 16:23 London package-validator hardening: `validate_path_a_package.py` now
checks the fourth required figure family by requiring non-empty qualitative
StrategyEIG examples and a trajectory PNG under `results/location_qualitative/`; it
accepts a later valid report if another arm's qualitative report is empty. Updated the
package-builder test fixture to include mini fixed-root decisions so qualitative
extraction is validated end-to-end. Focused tests pass:
`pytest tests/test_validate_path_a_package.py tests/test_build_path_a_package.py
tests/test_extract_location_qualitative_examples.py -q` (`5 passed`). Current local
package validation still fails as expected until Phase 4 outputs exist: missing depth
sweep report, headline plot, qualitative examples, and cost report.
Follow-up 16:31 London verification: after the paper cost section and qualitative-package
validator changes, the full local suite passes: `pytest tests/ -q` (`454 passed, 1 skipped`).
Follow-up 16:46 London cost artifact closure: `scripts/cost_vs_depth_table.py` now supports
`--from-config --max-depth` planned cost proxies, with direct `python scripts/...` execution
working via the standard project-root bootstrap. Generated
`results/cost_vs_depth/path_a_preregistered_cost_vs_depth.{json,md}` from the constrained
and unconstrained final50 configs with `--max-depth 5`. Package validation now marks
`cost_vs_depth` OK; remaining missing checks are Phase-4-derived depth sweep report,
headline plot, and qualitative examples. Focused cost/package tests pass
(`pytest tests/test_cost_vs_depth_table.py tests/test_validate_path_a_package.py -q`,
`8 passed`).
Follow-up 16:48 London verification: full local suite passes after the planned-cost
artifact changes: `pytest tests/ -q` (`456 passed, 1 skipped`).
Follow-up 17:02 London paper-validation hardening: added `scripts/validate_paper_draft.py`
to compile `paper/main.tex` in a temporary directory with `pdflatex`, `bibtex`, and two
final `pdflatex` passes, then verify the 4--6 page workshop target. The current draft
validates at 6 pages. `scripts/path_a_preflight.py --json` now includes
`paper_validation.ok=true` while leaving `package_validation.ok=false` until Phase 4
depth-sweep/qualitative artifacts exist. Focused tests pass:
`pytest tests/test_validate_paper_draft.py tests/test_path_a_preflight.py -q` (`6 passed`).
Follow-up 17:03 London verification: full local suite passes after paper-validator and
preflight integration: `pytest tests/ -q` (`459 passed, 1 skipped`).
Follow-up 17:13 London experiment-ledger cleanup: non-mutating `squeue -u hanyal`
returned no active jobs, so `EXPERIMENTS.md` was updated to mark stale GH200/MSC
full50/MPP30 rows as canceled/stale with no final metrics. Added ledger rows for the
tracked planned cost artifact (`fa03be5`) and paper draft validation (`ad1a4c2`).
Follow-up 17:14 London verification: package validation still has ranking/oracle/cost
green and only Phase-4-derived depth sweep report/headline plot/qualitative examples
missing; paper validation is green at 6 pages; preflight is green with
`paper_validation.ok=true`; full local suite passes (`459 passed, 1 skipped`).
Follow-up 17:25 London ledger-validation hardening: added
`scripts/validate_experiments_ledger.py` and integrated it into `path_a_preflight.py`.
The ledger validator checks required evidence rows (ranking gate, constrained oracle,
robustness, planned cost, paper validation), rejects running/pending statuses, requires
commit/tag cells, and verifies complete-row artifacts exist. Current preflight has
paper and ledger validation green; package validation still waits only on Phase 4
depth-sweep/headline/qualitative outputs. Focused tests pass:
`pytest tests/test_validate_experiments_ledger.py tests/test_path_a_preflight.py -q`
(`6 passed`).
Follow-up 17:26 London verification: full local suite passes after ledger-validator
integration: `pytest tests/ -q` (`462 passed, 1 skipped`).
Follow-up 14:50 London paper-validation hardening: `scripts/validate_paper_draft.py`
now rejects unexpected `\todo{}` markers and verifies the goal-required limitations
coverage before running LaTeX. The current draft validates with exactly four allowed
Phase-4 placeholders, all six required limitation topics, successful compile, and 6-page
count. Preflight remains green for launch readiness, paper validation, and ledger
validation; package validation still waits only on Phase-4 depth-sweep/headline/
qualitative artifacts. Focused tests pass:
`pytest tests/test_validate_paper_draft.py tests/test_path_a_preflight.py -q`
(`8 passed`), and the full suite passes: `pytest tests/ -q`
(`464 passed, 1 skipped`).
Follow-up 14:53 London package-validation hardening: `scripts/validate_path_a_package.py`
now also requires a non-empty ranking-fidelity diagnostics plot, so the package validator
covers the banked ranking figure family rather than only the ranking report. Current
package validation marks ranking gate, ranking diagnostics plot, constrained oracle, and
planned cost green; it still fails only on the expected Phase-4 depth-sweep report,
headline RMSE plot, and qualitative strategy examples. Focused tests pass:
`pytest tests/test_validate_path_a_package.py tests/test_build_path_a_package.py
tests/test_path_a_preflight.py -q` (`7 passed`), and the full suite passes:
`pytest tests/ -q` (`464 passed, 1 skipped`).
Follow-up 14:56 London cost-figure closure: `scripts/cost_vs_depth_table.py` now emits a
PNG cost-vs-depth scaling figure alongside its JSON/Markdown outputs, and
`validate_path_a_package.py` requires that non-empty cost figure. Regenerated
`results/cost_vs_depth/path_a_preregistered_cost_vs_depth.{json,md,png}` from the active
26B A4B constrained/unconstrained final50 configs; the PNG is force-added because
`results/*` is broadly ignored. Current package validation marks ranking report,
ranking diagnostics plot, constrained oracle, planned cost report, and planned cost plot
green; it still fails only on Phase-4 depth-sweep report, headline RMSE plot, and
qualitative examples. Focused tests pass:
`pytest tests/test_cost_vs_depth_table.py tests/test_validate_path_a_package.py
tests/test_build_path_a_package.py tests/test_path_a_preflight.py -q` (`12 passed`),
and the full suite passes: `pytest tests/ -q` (`464 passed, 1 skipped`).
Follow-up 15:00 London paper figure coverage: `paper/main.tex` now uses the generated
cost-vs-depth PNG as Figure `fig:cost-depth` instead of only the inline cost table, while
the exact cost counts remain in the package Markdown table. `validate_paper_draft.py`
now verifies that the draft contains the four goal-required figure slots: ranking
diagnostics, main depth contrast, cost-vs-depth, and qualitative strategies. The current
paper validates with four allowed Phase-4 placeholders, all six limitation topics, four
required figure labels, successful compile, and 6-page count. Preflight remains green
for launch readiness, paper, and ledger; package validation still waits only on the
Phase-4 depth-sweep report, headline RMSE plot, and qualitative examples. Full suite
passes: `pytest tests/ -q` (`465 passed, 1 skipped`).
Follow-up 15:03 London sync/readiness hardening: `scripts/path_a_sync_commands.py` now
includes the full package-builder dependency chain (`compare_location_depth_sweeps.py`,
`cost_vs_depth_table.py`, `extract_location_qualitative_examples.py`, and
`llm_token_usage.py`) in required sync/readiness paths, preventing a remote package build
from using stale helper code after the local cost-figure changes. `PHASE4_LAUNCH_HANDOFF.md`
now reflects that cost-vs-depth Markdown and PNG artifacts are already present, and that
the package remains incomplete only until Phase-4 depth-sweep/headline/qualitative
artifacts exist. Focused tests pass:
`pytest tests/test_path_a_sync_commands.py tests/test_path_a_preflight.py
tests/test_validate_path_a_package.py -q` (`12 passed`), and the full suite passes:
`pytest tests/ -q` (`466 passed, 1 skipped`).
Follow-up 15:06 London runbook cleanup: `LOCATION_DEPTH_PATH_A_RUNBOOK.md` now matches
the current split-MPP30 relaunch plan rather than the stale monolithic MSC/GH200 state.
It records zero active Phase 4 jobs, the split-block combine/package path, the current
required evidence checklist including cost PNG and qualitative artifacts, and treats
rollout-count ablation as optional appendix evidence rather than MPP-required output.
Recovery instructions now use split-block run names. Full suite passes:
`pytest tests/ -q` (`466 passed, 1 skipped`).
Follow-up 15:09 London ledger evidence refresh: `EXPERIMENTS.md` now records the
cost-vs-depth PNG artifact and commit `d0e3957`, and the paper-validation row now records
the stronger validator scope including `paper_required_figures` with commit `ebd6e54`.
`validate_experiments_ledger.py` now requires the cost PNG and paper figure-scope text in
the ledger evidence checks. Focused tests pass:
`pytest tests/test_validate_experiments_ledger.py tests/test_path_a_preflight.py -q`
(`6 passed`), and the full suite passes: `pytest tests/ -q`
(`466 passed, 1 skipped`).

## NEXT ACTIONS (in order)

1. No active cluster jobs are currently running for this goal. Before relaunch, sync the
   required Path A files with `python scripts/path_a_sync_commands.py`; current remote
   readiness is false only because required scripts are not on the cluster checkout yet.
2. If the user asks to relaunch, prefer the split MPP30 path: three 10-trial blocks per side
   (`--trial-offset` 0, 10, 20 plus `--total-trials 30`), depths 1/3/5, myopic controls
   3/5, GH200 Singularity launcher. Generate the exact launch/combine/package commands
   with `python scripts/path_a_launch_commands.py --split-mpp30`.
3. If relaunching on GH200, use the Singularity/container launchers only; do not use the
   A100/conda ranking or 20-questions scripts on GH200.

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
- Tests: `pytest tests/ -q` must stay green (last known: 466 passed, 1 skipped).
