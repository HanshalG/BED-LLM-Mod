# VoI Communication Source Audit Result

## Decision

The newly released ACL 2026 Value-of-Information code is the strongest fresh
candidate found in this audit, but its existing multi-step implementation must not
be used as evidence of non-myopic planning.

The flight task is closed because its five questions reveal independent,
predefined preference attributes. The medical task passes source eligibility for
a separately preregistered, corrected future-first mechanics test: semantic
yes/no questions, a hidden diagnosis, LLM-generated response likelihoods,
history-conditioned question proposals, and heterogeneous terminal stakes are all
present.

No OpenRouter calls were made.

## Frozen Source

- Paper: `https://arxiv.org/abs/2601.06407`
- Official repository: `https://github.com/dong-river/VOI_communication`
- Audited commit: `27466a7832d5aafff82017a659e08942e18b01ae`
- MedDG SHA-256:
  `e851864a9cb53c36304245bc3213a8a894cf7b86f8945d60978923e1f1ef0169`
- Flight train SHA-256:
  `d270986be41fbdc1a058e19ef08d2371b77b7b1d570d92b3295b4d9091a7f8e3`
- Flight eval SHA-256:
  `5b504b16453dac64ed80aa426de2c75f843621245251289b83e74a83bc7502f5`

The release contains 499 MedDG records spanning the same 15 diagnoses used by
the default medical candidate pool. It also includes 100 complete ten-turn VoI
medical transcripts and the corresponding confidence baseline transcripts.

## Why Flight Is Myopic

Flight questions are a fixed five-item menu: price, stops, layover, arrival, and
airline. Once one feature is answered it becomes a deterministic point mass and
the question is removed. An answer does not create a new semantic variable,
unlock a new question, or alter the response model for another feature. Under
additive feature utility, deeper planning mainly chooses an order over independent
measurements rather than exercising an answer-conditioned semantic policy.

## Why Medical Is Promising

The medical environment has the ingredients missing from the rejected
clarification benchmarks:

- a finite latent diagnosis set and empirical patient records;
- free semantic yes/no questions proposed from the current dialogue;
- batched LLM predictions of answers under every candidate diagnosis;
- an LLM posterior over diagnoses after realized or hypothetical answers;
- branch-specific follow-up questions in the realized transcripts;
- diagnosis-dependent terminal rewards.

The released transcripts demonstrate that the interaction is not a static
checklist. From the same empty initial history, common abdominal-pain roots lead
to later questions about location, diarrhea, vomiting, cough, or sore throat as
the answer history evolves.

## Released Lookahead Defect

`ValueOfInformationStrategy` exposes `lookahead_k`, but only depth one is
scientifically valid as written.

At recursive depth, `_best_value_k_steps` carries only a numerical belief key.
`_generate_candidate_questions` still reads the real environment history, not
the hypothetical branch history. Likewise, a deeper
`get_posterior_distribution` call includes the real history plus only the newest
hypothetical question and answer; it drops the ancestor hypothetical questions
and answers that produced the incoming belief. Consequently, `lookahead_k > 1`
does not construct a coherent branch-conditioned conversation tree.

The released runner defaults to `lookahead_k=1`, consistent with the paper's
one-step VoI method. Existing published transcripts therefore remain useful
myopic baselines, not non-myopic evidence.

## Authorized Next Step

Only a separately frozen mechanics test is justified:

1. use the released 15-diagnosis support and empirical prior;
2. generate a small root question set without exposing a hidden patient;
3. append each hypothetical root answer to the branch history before proposing
   follow-ups;
4. freeze LLM answer maps for every diagnosis and every question;
5. compute depth-one and depth-two policies exactly on that frozen tree;
6. require a root-policy change and positive exact expected-utility gap before
   any patient-level serving run.

The mechanics test must use non-reasoning calls, remain within the current
`$1.28170905` allowance, and preserve the `$25` Monday reserve. OatML and cluster
execution remain prohibited.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
