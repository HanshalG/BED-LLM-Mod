# Tau2 MMS Documented Split Calibration Protocol

Date frozen: 2026-08-13

## Predecessor Boundary

The split root/native interface is terminally closed. Its twelve calls were
mechanically clean, but all six native responses collapsed four distinct
permission states to one partition. A zero-call audit localized a missing
public action contract: candidate descriptions named SMS/storage permission
differences, while the prompt never stated that `check_app_permissions` exposes
granted permission names. No development or endpoint opened.

This successor changes cohort, seeds, and semantic context before any new
response. It does not retry the predecessor or relax any gate.

## Fresh Cohort

Select reserve positions twelve through fourteen within each of `mms_abroad`
and `mms_home`: six untouched four-world episodes, hash-bound in the public
manifest. They are disjoint from all four earlier Tau2 semantic cohorts. Task
IDs, source fault names, official observations, repairs, rewards, and endpoints
remain unserialized.

## Public Tool Contract

Bind Tau2 telecom source commit
`1d244f5dca42944b67a379b44bfeb9f5748f189d` and
`user_tools.py` SHA-256
`03fa751eeea3734313a3f5274223d42750fd5c1a28f2624265a678f271ac309d`.
The model receives only public read-tool semantics available to an agent:

- status bar reports airplane/SIM/network/data/Wi-Fi status;
- network status reports complete cellular and Wi-Fi state;
- network mode reports the preferred cellular mode;
- APN settings report APN name and MMSC URL;
- Wi-Fi Calling reports whether it is on;
- speed test reports connection speed or failure;
- MMS probe reports whether an MMS can be sent;
- installed apps lists app names;
- after `installed_apps`, `messaging_permissions` calls
  `check_app_permissions("messaging")`, which reports the names of currently
  granted permissions. The relevant visible names are `sms`, `storage`, and
  `phone`; an absent name is not granted.

These are interface definitions, not selected outcomes. The model still must
infer each world's response from its candidate description.

## Split LLM Interface

Use exact `deepseek/deepseek-v4-flash-0731` nonreasoning. Each episode has two
independent strict requests:

1. **Root equivalence:** eight ordered booleans with confidence, now accompanied
   by the eight bound public tool descriptions and visible field semantics.
2. **Native partition:** the same one-action canonical four-label partition,
   now accompanied by the exact public permission-read contract above.

Code alone maps judgments to likelihoods, performs Bayesian updates, and
computes greedy and depth-two information values.

## Frozen Execution Envelope

- Twelve calls with fresh root seeds `202608131000`--`1005` and native seeds
  `202608131100`--`1105`.
- Temperature zero; root/native output caps `900`/`300` tokens.
- Zero retries; concurrency two; exactly twelve maximum HTTP attempts.
- Atomic mixed checkpoints and complete bank before official observations.
- Prospective stage cap `$0.06`; per-request reservation `$0.004`.
- Hard account-wide Europe/London cap `$5.00`; any execution must use a new
  dated wrapper chained from the latest authenticated daily ledger.

No paid execution is authorized by this protocol alone. A producer,
producer-independent verifier, source audit, adversarial tests, dated budget
wrapper, immutable binding, pushed commit, and same-day authenticated preflight
are all required first.

## Gates

Retain the predecessor's full conjunction unchanged: clean serving and replay;
at least `47/48` exact roots and root Brier at most `0.08`; all six native
partitions and all 36 pair relations exact with native partition Brier at most
`0.08`; all 24 native answers top-ranked with mean truth posterior at least
`0.65` and posterior Brier at most `0.18`; equivalence mean/max TV at most
`0.03`/`0.10`; greedy avoids and depth two selects `installed_apps` in all six;
every horizon gain at least `0.50` nats; semantic/source depth-two Spearman at
least `0.90`.

A pass authorizes only a separately frozen paired development protocol with
compute-matched myopic and random controls and sealed task-success endpoints.
A failure closes this interface, cohort, and seeds.
