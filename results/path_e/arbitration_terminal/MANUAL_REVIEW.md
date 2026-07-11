# Paprika Naive-Primary Arbitration Manual Review

Date: 2026-07-11

## Verdict

PASS. All 10 arbitration transcripts were reviewed against their private solutions.
All six recorded resolutions directly implement the private remedy. None of the four
unresolved transcripts performs the exact private remedy and is then told that it
failed. The automated endpoint-valid result is accepted.

## Task Audit

| Task | Result | Private-remedy review |
|---|---:|---|
| 0000 refrigerator | Resolved, turn 1 | Checking that the doors are fully closed directly closes/seals the slightly ajar door. |
| 0001 label printer | Resolved, turn 2 | Replacing the thermal ribbon/ink supply exactly matches the depleted-ribbon remedy. |
| 0002 fleet API | Unresolved | The policy never updates the expired API key. Log checks and token regeneration were only proposed or attempted as diagnostics. |
| 0003 dishwasher | Unresolved | The policy inspects the hose externally but never clears the internal hose clog. Replacing the pump is correctly nonterminal. |
| 0004 package scale | Resolved, turn 1 | Running the standard calibration procedure exactly matches the calibration remedy. |
| 0005 trailer lights | Resolved, turn 4 | The customer discovers the loose connector, pushes it in, and tightens it; this exactly matches securing the connector. |
| 0006 navigation | Unresolved | The policy never checks or enables the van's onboard Wi-Fi hotspot. |
| 0007 controller | Resolved, turn 3 | Connecting the uncharged controller to wall power directly charges it. |
| 0008 pressure cooker | Unresolved | Cleaning/reseating the gasket is not the hidden action of properly closing and locking the lid; the policy never explicitly reseals the lid. |
| 0009 kiosk | Resolved, turn 2 | Cleaning the screen exactly matches the dirty-screen remedy. |

The dishwasher alternative actions remained nonterminal. No transcript contains a
correctly performed private remedy followed by a failure response.

## Override Audit

The EIG gate overrode candidate 0 on 12 of 33 turns (36.4%). Four overrides selected
the exact remedy and resolved immediately: 0001 turn 2, 0004 turn 1, 0007 turn 3, and
0009 turn 2.

The override policy was not uniformly good. Notable poor overrides include selecting
drain-pump replacement instead of checking the dishwasher hose connection (0003 turn
5), reinstalling the navigation app (0006 turn 4), and checking pressure-cooker liquid
instead of the sealing valve (0008 turn 2). Other overrides were exploratory and have
no identifiable counterfactual outcome. The 12/33 override rate and four immediate
remedy selections describe the observed mechanism; they do not causally attribute the
full arm-level gain to EIG because the native candidate prompt is stochastic.

## Recovery Provenance

Original offset 8 failed during its round-4 analytical belief refresh after exhausting
five structured repairs and produced empty metrics. The accepted offset-8 artifact is
the exact-config recovery `20260711T153606`; all other accepted shards are from the
original `20260711T143605` launch (`o0` started at `143606`). No partial state from the
failed shard was reused.
