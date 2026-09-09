# Fixed-slot controller and fresh source preflight

Previous goal turn was progress: the actual slot runtime and predictive-failure
semantics were verified. This turn integrated them into `rearc_slot_panel` and
froze a disjoint cohort/protocol at pushed commit 63acab86 before source checks.

Metadata-only selected IDs: bdad9b1f, 2dee498d, 1caeab9d, 99b1bc43.
Selection excludes all 16 tasks in the three closed cohorts, uses the frozen
SHA256(bed-rearc-slots-v1:+id) order, and permits no replacements. Source seeds
36000..36002, demos36100..36102, targets36200..36207; initial model seeds36300+2i,
paired aware/blind36400+2i, repair+1. Existing medium reasoning, scientific gates,
candidate limits, search caps and $1.44 block ceiling are unchanged.

All 12 isolated source checks passed exact generator/reference equality with
hashes/shapes only. The public collector then made exactly 44 source requests:
12 demonstration outputs and 32 target inputs plus output hashes. Target output
grids remain sealed. All 12 initial/aware/blind proposal request bodies fit the
32768-byte bound; maximum 28083 bytes. Repair requests will still be checked
individually rather than truncating content or increasing the frozen cap.

Controller adversarial tests cover complete positive-gate mechanics (not a real
positive), exact 24 calls, 8-call initial null, all-invalid no-execution behavior,
failed-search termination, no refill/reorder/missing slots, paired seeds,
history-blind feedback privacy, 64/128/16 candidate slot counts, and seal-before-
target ordering. Source tests bind cohort/source hashes and ordered seed coverage.
26 focused tests passed in 1.44s. Original paid failure still replays exactly.
No containers remain; model calls and spend this turn are zero.

Next dependency is the paid transport/bank integration: reuse the banked public
collector and source-only symbolic prefix, hash nested raw-worker artifacts,
reconstruct successes and failures without process/model dispatch, and enforce
the existing per-attempt reservations and account-wide daily limit. Only after
that integration passes should the new four-task Luna-medium qualification run.
Do not regenerate this public collector or resume a closed cohort.

This is source/controller readiness, not semantic calibration, predictive
transfer, or non-myopic improvement. Account balance remains $23.581723111;
conservative London Sep9 spend $0.99999841. Research goal remains unachieved.
