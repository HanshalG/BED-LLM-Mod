# AmbigDocs Binary Semantic Serving Gate

## Purpose

Validate the exact question-generation and likelihood interface needed for
truth-preserving AmbigDocs semantic lookahead before sampling any hidden entity.

## Frozen Protocol

- Dataset: `yoonsanglee/AmbigDocs`
- Revision: `19d318a5e6717f63b9b864aa804bc57f69e824df`
- Dev SHA256:
  `43ab72b880337fac6442ea04accbda38d6a8e785eb2db4970585fac7ae775d68`
- Already-open row 49, `qid=43608`, six Minsk entities
- Model: `openai/gpt-5.4-mini`, explicitly non-thinking
- Seed: `24388`
- Five one-line yes/no question calls at temperature `.7`
- Five separate exact six-character `Y/N/U` likelihood calls at temperature `0`
- Exactly 10 calls, concurrency 5
- Raw checkpoint before parsing
- No target sampling, responder, score, endpoint, normalization, repair, or reissue
- Cost cap `$0.15`; projected cost `$0.03`
- No OatML

## Frozen Gates

All must pass:

- exactly 10 requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all five questions and six-character label maps parse;
- at least four unique questions;
- at least four unique likelihood partitions;
- at least four partitions use two or more labels and have entropy at least `.30`
  nats;
- informative EIG range at least `.10` nats;
- cost at most `$0.15`.

Failure closes this exact AmbigDocs Mini interface before target sampling. Passage
authorizes only a separately preregistered development efficacy gate with a frozen
hidden target, independent GPT-5.4 responder, and compute-matched
myopic/lookahead/width/random controls. The official test split remains sealed.

## Dry Verification

Before real calls, compilation and focused tests pass. The full deterministic
source-backed 10-call fixture passes every gate with five unique informative
partitions and `.2426` nats of EIG range.
