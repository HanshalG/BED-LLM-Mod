# InteractComp Regenerated-Support First-Link Preregistration

Date: 2026-07-25

## Question

Can a non-thinking LLM maintain a useful answer-hypothesis population for a
genuinely underspecified question, propose informative clarification questions,
and regenerate support after the true user's answer such that estimated
immediate EIG ranks externally measured exact-answer recovery?

This is a first-link opportunity gate. It tests the transition needed by
non-myopic BED, but it does not compare depth-one and depth-two policies. A
depth-two run is authorized only through a separate preregistration if every
gate below passes.

## Frozen Source And Tasks

- InteractComp repository:
  `https://github.com/FoundationAgents/InteractComp`.
- Repository commit: `9cdf7f804f527ad32a405efaa6c86aae03692556`.
- Encrypted `InteractComp210.jsonl` SHA-256:
  `0bd0ccc4b69c228c04c15b1147211adc6c6483b852d74e5ac7e5a34c9db80496`.
- Eligibility: context has at least three nonempty lines; answer length is
  between 2 and 80 characters; normalized answer is absent from both question
  and context; inspected indices 0-2 are excluded.
- Eligible tasks: 156 of 210.
- Manifest seed: `24377`.
- Seeded eligible-manifest SHA-256:
  `9375f5cb3e52590c5eafca25c96f6378c6f44da674162ffdf63c5fe949eeb6c6`.
- Frozen first two entries: dataset indices `75` and `37`, benchmark IDs `76`
  and `38`.

The frozen task contents have not been inspected. Plaintext benchmark data
remains outside the repository and raw model responses remain private.

## Models And Support

- Support generator, question generator, particle classifier, and support
  refresher: `openai/gpt-5.4-mini`.
- Closed-mode true responder: `openai/gpt-5.4`.
- Reasoning is disabled for both models.
- Generation temperature is `.7`; classification and response temperature is
  `0`.
- Eight independent answer/profile particles are generated per task.
- Four independent, entity-name-free yes/no clarification questions are
  generated per task.
- Each initial particle is classified as `Y`, `N`, or `U` for all four
  questions. Immediate EIG is the natural-log entropy of this particle outcome
  partition.
- The true responder sees only the hidden InteractComp context and one proposed
  question, following the benchmark's closed-mode `yes`/`no`/`i don't know`
  contract.
- After each realized response, eight fresh answer/profile particles are
  generated from the original question and that clarification history. Invalid
  strict two-line responses are filtered; at least six particles must remain.
- Repeated candidates retain particle multiplicity. Exact normalized entity
  equality to the benchmark answer defines truth mass.

## Staged Access

The harness enforces this order:

1. decrypt only the two ambiguous questions;
2. generate 16 initial particles;
3. generate eight candidate clarification questions;
4. classify all 16 particles and freeze all eight immediate-EIG values;
5. decrypt hidden contexts and obtain eight true closed-mode responses;
6. generate 64 outcome-conditioned refreshed particles;
7. checkpoint every target-blind call and frozen score;
8. only then decrypt the two exact target answers and compute endpoints.

The target answer is never included in a prompt. Hidden context is available
only to the separate responder and is never supplied to the support generator.

## Frozen Metrics

For each task:

- initial endpoint: exact target-answer mass in the initial eight particles;
- root endpoint: exact target-answer mass in the refreshed particles after that
  root's realized response;
- selected root: maximum immediate EIG, with frozen root-order tie breaking;
- opportunity oracle: maximum externally measured root endpoint;
- ranking fidelity: Spearman correlation between immediate EIG and root
  endpoint; and
- gains: selected endpoint minus both initial and mean-candidate endpoints.

These metrics test only the first link between the model-induced belief score
and realized support improvement. They do not establish non-myopic policy
efficacy.

## Exact Calls

| Stage | Mini calls | GPT-5.4 calls |
|---|---:|---:|
| Initial particles | 16 | 0 |
| Clarification questions | 8 | 0 |
| Particle outcome classifications | 16 | 0 |
| True closed-mode responses | 0 | 8 |
| Outcome-conditioned refreshes | 64 | 0 |
| **Total** | **104** | **8** |

The run must make exactly 112 physical requests and 112 HTTP attempts. No
scientific response repair, replacement, or reissue is allowed.

## Exact Gates

Every gate must pass:

1. exactly 112 physical requests and 112 HTTP attempts;
2. zero transport retries, reasoning tokens, and forced exits;
3. all initial populations contain eight valid particles;
4. at least three unique clarification questions per task;
5. at least two questions with EIG at least `.30` nats per task;
6. at least two non-unknown true responses per task;
7. every refreshed root retains at least six valid particles;
8. the exact target appears with mass at least `1/8` after at least one root per
   task;
9. root endpoint range is at least `1/8` per task;
10. EIG-endpoint Spearman correlation is finite and positive per task, with
    mean at least `.20`;
11. mean selected endpoint gain over the initial population is at least `.125`;
12. mean selected endpoint gain over the four-root mean is at least `.05`; and
13. adapter cost is at most `$1.50`.

Failure closes this exact two-task first-link interface. There is no task
replacement, favorable-subset reporting, parser broadening, threshold change,
or same-interface rerun. Passage authorizes only a separately frozen
path-dependent depth-two test.

## Budget

- Projected cost: `$0.40`.
- Hard run cap: `$1.50`.
- User-authorized new-work allowance through Monday: at most `$15`.
- Project-ledger ceiling: `$101.14330481920742`.
- Project-ledger spend before this run: `$86.18932031920745`.
- Local allowance remaining before this run: `$14.9539845`.
- Last authenticated OpenRouter remaining balance: `$44.202352384`, or
  `$19.202352384` above the protected `$25` reserve.
- OatML resources: prohibited.

The stricter of the live balance, local allowance, and run cap applies. Both
budget sources must be checked again immediately before the paid command.

## Deterministic Verification

The full deterministic fixture completed exactly 112 requests: 104 generator
and eight responder calls, with zero retries, reasoning tokens, or forced
exits. It checkpoints target-blind responses before answer decryption and fails
the scientific gates by construction because fixture particles never contain
the hidden target. Focused tests:

```text
pytest -q tests/test_interactcomp_first_link_opportunity.py
4 passed
```
