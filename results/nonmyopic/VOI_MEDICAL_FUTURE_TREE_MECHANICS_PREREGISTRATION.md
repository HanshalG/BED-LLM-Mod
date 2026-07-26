# VoI Medical Coherent Future-Tree Mechanics Preregistration

## Claim And Scope

This is a source-qualified mechanics gate for LLM-native non-myopic sequential
BED. It does not evaluate patient accuracy and cannot establish an efficacy
claim. Its sole question is whether a semantic response model and genuinely
answer-conditioned follow-up generator produce a frozen two-step tree on which
exact depth-two EIG selects a better root than matched-compute greedy EIG.

The protocol corrects the released VoI `lookahead_k>1` history defect. Every
follow-up prompt contains the complete hypothetical root question and answer,
and every posterior is computed from the same frozen semantic answer maps that
define that branch.

## Frozen Source

- Official repository: `https://github.com/dong-river/VOI_communication`
- Commit: `27466a7832d5aafff82017a659e08942e18b01ae`
- Data: `mixed_20q/data/MedDG.json`
- SHA-256:
  `e851864a9cb53c36304245bc3213a8a894cf7b86f8945d60978923e1f1ef0169`
- Rows: 499
- Support: the 15 diagnoses occurring in the data and used by the released
  default medical task
- Prior: exact empirical target frequencies over all 499 rows

No record text, hidden patient, realized answer, or endpoint is exposed to the
question generator. Only diagnosis names and the empirical prior are supplied.

## Frozen Model Interface

- Model: `openai/gpt-5.4` through OpenRouter
- Reasoning: disabled
- Temperature: `0`
- Scientific retries, repairs, reissues, and parser fallbacks: zero
- OpenRouter concurrency: at most 12
- Expected physical requests: exactly 18
- Projected cost: `$0.18`
- Hard run cap: `$0.35`
- Protected balance: `$25` through Monday remains untouched
- OatML/cluster execution: prohibited

The 18 requests are:

1. one four-root question proposal;
2. one complete 4-by-15 root answer matrix;
3. twelve two-question follow-up proposals, one for every root crossed with
   `Yes`, `No`, and `Maybe`;
4. four complete follow-up answer matrices, one per root.

Question output is a strict `Q<number>|<question>` line grammar. Answer maps use
strict `Q<number>|<diagnosis>|<Yes/No/Maybe>` lines. Responses are checkpointed
privately before parsing. Any malformed, missing, duplicated, repeated-root, or
inconsistent cell fails closed.

## Exact Policies

The LLM answer labels form a deterministic likelihood table. Bayes updates and
entropy calculations after generation are exact and make no further model calls.

- **Myopic:** select the root with maximum immediate entropy reduction; within
  each realized root branch, select the better of its two frozen follow-ups.
- **Depth two:** select the root with maximum expected total entropy reduction,
  allowing each root outcome to choose its own best frozen follow-up.

Thus both policies use the identical tree and identical second-step compute. The
only difference is whether future branch value affects the first action.

## Conjunctive Gate

The mechanics result passes only if all conditions hold:

- source hash, row count, diagnosis set, and empirical prior reproduce;
- exactly 18 physical requests and 18 HTTP attempts complete;
- zero retries, reasoning tokens, and forced exits;
- all four root maps and all twelve branch question sets parse completely;
- every root has at least two positive-probability outcomes;
- every root has at least two distinct answer-conditioned follow-up sets;
- immediate-EIG range is at least `0.05` nats;
- depth-two-EIG range is at least `0.05` nats;
- depth two selects a different root from myopic;
- the depth-two-selected root improves expected two-step EIG by at least `0.03`
  nats over the myopic-selected root on the same tree;
- total adapter cost is at most `$0.35`.

Failure closes this exact tree/interface. There is no alternate seed, threshold
change, parser repair, model substitution, or partial-tree analysis. Passing
authorizes only a separately preregistered small patient-grounded first-link
validation; it does not authorize a scale run.
