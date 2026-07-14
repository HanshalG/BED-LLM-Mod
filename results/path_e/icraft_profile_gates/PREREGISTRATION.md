# iCRAFT-MD Diagnosis-Profile Gate Preregistration

Date: 2026-07-14

Commit at registration: `d5dc7f5`.

## Scope

This registers iCRAFT-MD diagnosis-profile validation gates only. It does not authorize
a policy comparison, a full-depth evaluation, or a retry after a failed gate. The initial
OpenRouter cap is `$0.50` per run. The 26B A4B scaffold is non-thinking.

## Immutable Contract

- Repository: `https://github.com/stellali7/MediQ.git`
- Commit: `faa2ce62fef0423e35af4c31d7537aad973173eb`
- Split: `data/all_craft_md.jsonl`
- SHA-256: `658441e6c6692c84d78fdf1c5ddb406ee6705f4a43330f0f759a1b941d8bfcdc`
- Target: released four-way diagnosis label.
- Patient channel: official Fact-Select only.
- Outcome support: `Yes`, `No`, and `Information unavailable / not in record`.

The release contains two raw `answer` text disagreements with the indexed option
(source IDs 112 and 129). The official `mediQ_benchmark.py` scores `answer_idx`, so
this work uses that released index as the target, preserves the raw text in artifacts,
and reports both disagreements in every data manifest. This parser transparency change
does not affect any preregistered partition or threshold.

The fixed source-ID partition comes from `random.Random(1304).shuffle(range(140))`.

| Partition | Count | Source IDs |
|---|---:|---|
| Development | 8 | 125, 83, 124, 38, 8, 92, 104, 135 |
| Calibration | 12 | 2, 99, 132, 60, 62, 64, 23, 137, 96, 100, 117, 40 |
| Structural | 12 | 30, 52, 103, 33, 85, 36, 94, 113, 34, 139, 15, 68 |
| Pilot reserve | 10 | 93, 80, 21, 115, 138, 12, 108, 119, 131, 126 |
| Headline reserve | 98 | All remaining source IDs |

## Fixed Model

Each diagnosis label receives exactly three validated counterfactual patient profiles.
Profiles use only initial evidence and the diagnosis; they may not use hidden atomic facts
or assert treatment, management, or test-order decisions. The direct temperature-zero
diagnosis prior is divided uniformly among its three profiles. Profile support is fixed
after validation: it is never regenerated or refreshed in a hypothetical branch.

Record availability is label-independent. Conditional on availability, Yes/No is
profile-conditioned. The same likelihood table is used for EIG, synthetic branches, and
deployed Fact-Select Bayes updates.

## Gates

1. Local contract: routing tests and the full suite pass before paid calls.
2. Cost micro-smoke: source ID 125, one round, two shared candidates, no parse/runtime
   failure, and spend at most `$0.05`.
3. Profile/prior: all calibration labels retain three profiles; mean log loss is below
   `log(4)`; mean Brier is below `0.75`; at most one wrong top diagnosis has probability
   at least `0.75`.
4. Likelihood/deployment/ranking: four root candidates for each calibration case, with
   at least 24 available outcomes, unavailable diagnosis movement at most `1e-12`,
   positive available true-label log gain, at least 60% true-label favoring, and
   Spearman EIG correlations at least `0.20` with realized entropy and truth-log gain.
5. Structural: on the structural partition, exact depth-two value minus the one-step
   root's depth-two value has mean at least `0.02` nats, at least 8/12 positive cases,
   and a one-sided 90% paired-bootstrap lower bound above zero (10,000 replicates).

Any failed gate stops this path. A structural failure does not permit a FactSelect rescue
attempt. A structural pass still requires a second authorization before a policy run.
The banked animals audit is one-step evidence and a negative paired depth result, never
positive non-myopic evidence.

## Authorized Stronger-Generator Retry Addendum

Registered before execution on 2026-07-14, after the Track-1 animals audit collapsed.
Hanshal authorized exactly one apparatus retry, capped at `$2`, in which a stronger
OpenRouter model may replace the 26B model for **profile narrative generation only**.
All partitions, prompts, support size, bounded repairs, seeds, likelihoods, thresholds,
and stopping rules above remain frozen.

The selected profile generator is `openai/gpt-5.4` with reasoning effort `none`.
OpenRouter's live model registry identifies it as a frontier model with structured-output
support; its listed input/output prices project the complete retry below `$1.00` from the
failed run's token shape, leaving margin below the hard `$2` run cap. Disabling its
default reasoning preserves output space for the strict JSON profile list and makes this
a capability substitution rather than a reasoning-budget intervention.

Role separation is strict:

- `openai/gpt-5.4`: author the fixed profile narratives only;
- `google/gemma-4-26b-a4b-it`, non-thinking: validate profiles, judge priors, generate
  and validate questions, score likelihoods, map/answer FactSelect interactions, and
  perform every questioner or policy role.

The one registered attempt is the calibration execution below. There is no extra paid
smoke and no prompt repair. If profile construction or any calibration criterion fails,
the external claim closes permanently. If calibration passes, the unchanged structural
gate runs next; a policy comparison still requires separate authorization.

```bash
set -a; source .env; set +a
python scripts/run_icraft_profile_gates.py \
  --config configs/config_mediq_icraft_profile_retry_openrouter.yaml \
  --stage calibration \
  --run-id icraft-profile-gate-stronger-generator-retry \
  --profile-generator-model openai/gpt-5.4 \
  --profile-generator-reasoning-effort none \
  --output results/path_e/icraft_profile_gates/STRONGER_GENERATOR_RETRY_CALIBRATION.json
```

## Retry Outcome

The single registered retry completed on 2026-07-14. It used 1,000 requests and
`$0.44032225`, below the `$2` cap: 55 `openai/gpt-5.4` profile-authoring calls
(`$0.39681000`) and 945 non-thinking 26B calls for every other role
(`$0.04351225`). It used zero reasoning tokens and no forced exits.

The stronger author completed the fixed support and the run reached all calibration
metrics. The prior passed its two registered quality thresholds (mean log loss
`0.56755 < log(4)`; mean Brier `0.28125 < 0.75`), but the terminal availability
criterion failed: only 14 of 48 realized candidate outcomes were answerable, below
the required 24. This failure alone closes the gate. The raw, pre-fix rank telemetry
was also negative (EIG versus entropy `-0.04131`; EIG versus truth-log gain
`-0.06777`), but it is not used as a separate conclusion because a later offline
audit found that posterior-floor smoothing perturbed low-mass profiles after an
otherwise neutral unavailable outcome. The code now regression-tests exact neutrality;
the frozen availability failure requires no rerun to establish closure.

The canonical outcome artifact is `STRONGER_GENERATOR_RETRY_CALIBRATION.json` and the
closure report is `STRONGER_GENERATOR_RETRY_FINAL_REPORT.md`. No further iCRAFT model
calls, calibration reruns, structural gates, ranking tests, policy runs, or depth runs
are authorized.
