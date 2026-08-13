# Tau2 Native Prerequisite BED Source Protocol

Frozen: 2026-08-13 Europe/London, after inspecting public source constructors,
workflow graphs, and invariant ticket templates, but before executing any
selected simulator tool or obtaining any model response.

## Purpose

Test whether the official Tau2 telecom generator contains a population of
source-grounded non-myopic diagnostic episodes, rather than one hand-selected
account example. Code owns executable tools and legal prerequisites. A later
LLM interface may own semantic hypotheses and predictive likelihoods, but only
after this source gate independently establishes a real horizon opportunity.

This is scientifically distinct from every closed Tau2 prompt-variant study.
All physical fault sets in the official `base` split and every physical world
used by the earlier MMS/account prerequisite scripts are excluded before
selection. Their prompts, seeds, and outcomes may not be reused.

## Immutable Source

- official repository `sierra-research/tau2-bench` at
  `1d244f5dca42944b67a379b44bfeb9f5748f189d`;
- executable T3 adapter at
  `492f31fa05d2065c750a72d5e798385af282fa5d`;
- `tasks_full.json` SHA-256
  `37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4`;
- `split_tasks.json` SHA-256
  `605b488bb9a6acb3c7f4505240a855fdc8681d09aadb16a8f38b2efcfc5c3aec`;
- expected official tasks / unique physical fault sets: `2285 / 2285`;
- expected excluded official-base / prior-study sets: `114 / 90`;
- expected fresh physical sets: `2092`.

The split algorithm depends only on task IDs and the official base-ID list,
although the standard JSON parser materializes each public task object. It
publishes no task ID, fault name, ticket, initialization action, tool response,
policy, user instruction, evaluation criterion, or endpoint. Public ticket
templates and source code are development context, not sealed outcomes.

## Native Episode Families

An episode has a uniform prior over source worlds that share an issue family
and all non-target faults. Only one native hidden subsystem varies:

1. `mms_abroad`: abroad MMS tasks, holding every non-app fault fixed while
   varying `none`, SMS permission, storage permission, or both permissions.
2. `mms_home`: the same app-permission variation for at-home MMS tasks.
3. `mobile_abroad`: abroad mobile-data tasks, holding every direct-device fault
   fixed while varying carrier roaming/device roaming and data exhaustion.

An episode is eligible only with at least four distinct source worlds and no
duplicate target state. Expected eligible episode counts are respectively
`297 / 77 / 29` (`403` total).

Sort within each family by
`SHA256("tau2-native-prerequisite-v1|" + family + "|" + sorted_backbone)`.
Allocate exact family quotas:

| split | MMS abroad | MMS home | Mobile abroad | episodes | worlds |
|---|---:|---:|---:|---:|---:|
| mechanics | 2 | 2 | 2 | 6 | 26 |
| opportunity | 24 | 12 | 8 | 44 | 186 |
| development | 40 | 16 | 8 | 64 | 267 |
| confirmation | 64 | 24 | 8 | 96 | 394 |
| reserve | 167 | 23 | 3 | 193 | 777 |

Public manifests contain only split/family counts, episode hashes, state
counts, and hashes of selected task IDs. Development, confirmation, and reserve
source responses stay sealed through opportunity and mechanics.

## Executable Actions

All actions are deterministic read-only calls on a freshly initialized official
environment. The caller receives rendered tool output with run-specific IDs and
personal values canonicalized before partitioning.

MMS root actions:

- status bar, network status, network mode, APN settings, Wi-Fi Calling,
  speed test, MMS probe, and installed-app list.

Only `installed_apps` unlocks `messaging_permissions`, which reads permissions
for the code-selected installed app named `messaging`. The permission action is
illegal after every other root.

Mobile-data root actions:

- status bar, network status, speed test, payment request, SIM status, and
  customer lookup by the known public phone number.

Only `customer_lookup` unlocks code-resolved reads of line details, data usage,
and customer bills. IDs are taken from the realized lookup result, never
hard-coded or supplied to an LLM.

## Exact Source Planner

For each episode, compute deterministic mutual information under its uniform
source prior. Greedy selects maximum root information. Exact depth two selects
the root with maximum expected information after an optimal legal second read
in each realized branch. Lexical action ID breaks exact ties. Horizon gain is
depth-two information minus the two-step information obtained when the first
action is forced to greedy.

The opportunity gate passes only if all integrity conditions hold and:

- at least 36/44 opportunity episodes are not root-saturated (`max root MI <
  .95 * prior entropy`);
- at least 30/44 change first action at depth two;
- at least 30/44 have horizon gain at least `.10` nats;
- mean horizon gain is at least `.15` nats;
- each family has at least half its episodes with gain at least `.10` nats;
- the depth-two selected first action is the native prerequisite on at least
  30/44 episodes;
- all gains are finite and nonnegative.

Thresholds, family definitions, splits, actions, canonicalization, and priors
are frozen before selected tool responses. A failure closes this construction;
no source-conditioned repair is permitted.

## Conditional LLM Stage

A source pass authorizes only a separately frozen mechanics interface. The LLM
must generate semantic hypotheses and calibrated response likelihoods from the
ticket, tool descriptions, and realized history. It may emit action IDs but
cannot invent tools, arguments, source states, or observations. The mechanics
gate must require schema validity, answer obedience, truth coverage and
calibration, positive score-to-source-value ranking fidelity, and a real
non-myopic action change. Any efficacy stage must use paired source worlds,
compute-matched myopic and random controls, sealed confirmation outcomes, an
independent replay, and a fresh account-wide budget boundary.
