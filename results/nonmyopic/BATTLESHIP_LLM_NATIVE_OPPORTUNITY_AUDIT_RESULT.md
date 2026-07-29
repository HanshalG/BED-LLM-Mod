# Battleship LLM-Native Opportunity Audit

Date: 2026-07-29

Status: **all zero-call opportunity gates pass; a fresh exact serving smoke is
authorized**.

## Source

The audit binds the official Collaborative Battleship artifact from *Shoot
First, Ask Questions Later?*:

- repository: `https://github.com/gabegrand/battleship`
- commit: `b98a4ba1c55be1bd5aa8038d42ad0070c41cabcb`
- released trajectory SHA-256:
  `c39aa87d888fb98d5c6750ce4a3d70da6757adab9a5d4a69a44801f298290e94`
- public result SHA-256:
  `16aff4012fd5992341f161c01a4dfab48d562e04fe2591eb6fd87d85feba0872`

The source contains 48 released model trajectories. All 39 natural-language
questions generated at an empty-board root have executable LLM-generated
semantic programs. Every program compiled, returned Boolean answers, and was
nonconstant on both independent posterior blocks. Joint behavioral
deduplication retained 25 questions.

## Protocol

The official conditional board sampler generated two independent blocks of
4,096 boards with seeds `39600` and `39601`. The observation model is the
paper's binary symmetric channel with `epsilon=0.1`.

For each fixed semantic question bank, exact finite-horizon dynamic programming
maximizes the expected MAP probability that the next shot hits a ship. Every
policy receives the same execution budget of three questions:

- horizon 1 replans one question at a time;
- horizon 2 replans up to two questions at a time;
- horizon 3 plans through the full remaining budget;
- greedy EIG maximizes immediate board-configuration information gain.

## Result

The selected root is stable across both posterior blocks and changes at every
planning horizon:

| Horizon | Selected root question |
|---:|---|
| 1 | Is there any part of a ship on tile D4? |
| 2 | Is there a part of any ship in row D, columns 4 or 5? |
| 3 | Is there any piece of a ship in D4, D5, E4, or E5? |

The roots form a coherent coarse-to-fine family: longer planning first asks
about a larger region that later questions can subdivide.

| Policy | Seed 39600 | Seed 39601 |
|---|---:|---:|
| horizon 1, three-question endpoint | 0.638089 | 0.635858 |
| horizon 2, three-question endpoint | 0.662852 | 0.658023 |
| horizon 3, three-question endpoint | **0.677839** | **0.672531** |
| greedy EIG, three-question endpoint | 0.436162 | 0.400848 |

Every adjacent horizon gain exceeds the frozen `0.01` threshold. Greedy EIG
selects a different root on each posterior block despite almost identical
root EIG (`0.53094` and `0.53096` bits). It gathers information about the full
board configuration but is poorly aligned with the next-shot hit objective.

## Interpretation

This is a retrospective source/opportunity result, not a fresh efficacy
claim. It nevertheless establishes all three ingredients needed for the next
stage:

1. LLM-generated language has distinct executable semantic consequences;
2. longer-horizon planning changes the first action in a stable,
   interpretable way; and
3. the same execution budget has monotonically better modeled task utility.

The result also provides a direct explanation for the location-finding
failure mode: maximizing broad latent-state entropy can be detached from the
decision-relevant endpoint. The appropriate successor is non-myopic expected
task utility over LLM-native semantic experiments, with greedy EIG retained
as a control.

## Consequence

Authorize only a separately frozen, exact 10-call serving smoke for fresh
question generation and independent semantic translation. No fresh policy
endpoint is yet authorized. The serving smoke must establish parseability,
cross-translator semantic agreement, nonconstant partitions, and question
diversity before any branch-conditioned tree is generated.

