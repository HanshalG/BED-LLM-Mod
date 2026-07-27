# HotpotQA Shared Comparative V4 Result

## Decision

V4 passes serving and completes development cleanly, but **fails the frozen
scientific gate**. Non-myopic and myopic both cover `9/10` support documents:
gain `0`, five ties, and no wins or losses. The non-myopic policy changes only
`2/5` first roots.

Do not open the holdout.

## Development Result

- Exact logical requests / HTTP attempts / retries: `50 / 50 / 0`
- Prompt / completion / reasoning tokens: `53,475 / 10,275 / 0`
- Forced exits / parser repairs: `0 / 0`
- Adapter-recorded cost: `$0.2878125`
- All `20/20` root-conditioned beliefs changed
- All `5/5` tasks had four mutually distinct branch beliefs
- Mean final-answer token F1: `.96`

Policy support totals:

| Policy | Supports / 10 |
|---|---:|
| Fixed initial belief | 10 |
| Non-myopic aligned | 9 |
| Myopic receding | 9 |
| Random receding | 6 |
| Shuffled branch beliefs | 5 |

Aligned beliefs differ from fixed and shuffled orders on `4/5` tasks. Correct
alignment beats shuffled by `+4` supports with four wins and zero losses, so
path-conditioned LLM beliefs are behaviorally load-bearing. They do not,
however, improve the first-link endpoint over myopic, and fixed initial
beliefs are one support better than aligned beliefs.

Task-level first-link behavior:

| Task | Myopic root | Non-myopic root | Changed | Myopic / non-myopic coverage |
|---|---|---|---:|---:|
| `5adf874e` | answer | answer | no | 2 / 2 |
| `5a7fc819` | enabling | enabling | no | 2 / 2 |
| `5abd5126` | answer | enabling | yes | 2 / 2 |
| `5a8b1dd6` | answer | distractor | yes | 1 / 1 |
| `5ac49ff6` | enabling | enabling | no | 2 / 2 |

The only two root changes are endpoint-neutral. One swaps two equally useful
support roots; the other swaps two one-support paths.

## Action-Space Audit

After the frozen V4 result, a zero-call source audit found that the runner did
not implement the directional transition used to qualify these tasks.

The source audit defines a strict unlock by paragraph links:

- the enabling paragraph mentions exactly the answer article;
- the answer paragraph does not link back to the enabling article; and
- therefore enabling-first support coverage is `2`, while answer-first is
  `1`.

V4 instead used `_candidate_titles`, which offered **all nine remaining
context titles after every root**. This let an answer-first policy directly
select the enabling title even when no reverse link exists. Four of five V4
tasks consequently had two optimal support roots under the implemented
action set.

On the already-open five rows, a diagnostic mention-restricted replay of the
action graph, without model rescoring, gives:

| Task | Root oracle values under paragraph-link actions | Optimal roots |
|---|---|---:|
| `5adf874e` | `[1, 2, 1, 1]` | 1 |
| `5a7fc819` | `[0, 1, 0, 2]` | 1 |
| `5abd5126` | `[0, 2, 1, 0]` | 1 |
| `5a8b1dd6` | `[1, 2, 0, 1]` | 1 |
| `5ac49ff6` | `[0, 2, 1, 0]` | 1 |

In every case the unique optimum is the enabling root. This diagnostic is not
a posthoc V4 policy score because V4's model saw the larger action set.

## Interpretation

V4 establishes two things:

1. LLM-generated branch beliefs are path-dependent and their correct alignment
   matters strongly relative to shuffled beliefs.
2. In the implemented unrestricted-title environment, non-myopic planning
   does not beat a strong myopic baseline.

It does **not** validly test the intended directional-unlock environment,
because unrestricted follow-ups erased the first-link asymmetry. A corrected
successor must freeze paragraph-mentioned follow-up actions before opening
fresh rows. It may use these five rows only for mechanics design, never for a
new efficacy claim.

## Integrity

- Serving run:
  `hotpot-shared-comparative-v4-serving-20260727T221349Z`
- Serving: all gates pass, exact `10/10`, cost `$0.0663525`
- Development run:
  `hotpot-shared-comparative-v4-development-20260727T221517Z`
- Development private raw SHA-256 is recorded in public `DEVELOPMENT.json`.
- Holdout remains sealed.
- Authenticated post-development balance: `$29.287059594`
- No fixed reserve.
- OpenRouter only; no OatML or Slurm.
