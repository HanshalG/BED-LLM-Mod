# tau-Knowledge Retrieval Opportunity Gate

## Purpose

Test an LLM-native non-myopic mechanism with an externally grounded transition
and endpoint. GPT-5.4 generates semantic information-need hypotheses and search
queries, while the official tau-Knowledge banking corpus, BM25 implementation,
and required-document annotations define deterministic observations and truth.
This avoids the unstable LLM semantic environment that closed PAPRIKA.

This first gate tests opportunity only. It does not train or evaluate a scorer
and does not claim a policy improvement.

## Source and fresh split

- Official repository:
  `https://github.com/sierra-research/tau2-bench`.
- Frozen commit: `1d244f5dca42944b67a379b44bfeb9f5748f189d`.
- Corpus shape: 698 banking documents and 97 tasks.
- Selection seed: `24334`.
- The earlier zero-call development audit used 36 tasks recognized by the old
  narrow opening regex. Tasks whose notes or full records were inspected were
  also excluded.
- Fresh eligibility requires an exact quoted, numbered scripted opening matched
  by `1. **Opening...:** "..."`.
- Smoke: `task_055`, `task_056`.
- Opportunity: `task_064`, `task_073`, `task_078`, `task_070`, `task_075`,
  `task_065`.
- Sealed confirmation: `task_039`, `task_054`, `task_074`, `task_080`,
  `task_058`, `task_076`, `task_038`, `task_063`, `task_035`, `task_069`,
  `task_061`, `task_040`, `task_066`, `task_057`, `task_077`, `task_053`,
  `task_041`, `task_059`, `task_072`, `task_079`.

## Protocol

- GPT-5.4 runs with provider reasoning disabled and temperature zero.
- From only the exact first customer utterance, it emits eight concrete
  information-need hypotheses, two direct queries, and three enabling queries.
- Each of the five queries is executed by `rank-bm25==0.2.2` `BM25Okapi` over
  official document title plus content, matching the official whitespace
  tokenization. Each search returns top 3.
- For each first result set, GPT-5.4 refreshes eight information needs and emits
  four document-conditioned followup queries.
- Each followup is executed by the same deterministic BM25 retrieval.
- The model never receives `required_documents`, evaluation actions, task
  notes, or later user instructions.
- Endpoint is the exact number of unique annotated required document IDs in the
  one-search or two-search result union.
- "Greedy first" is the root with the greatest exact one-search required
  document count, with original order for ties. This is an oracle-strength
  myopic structural control, not a deployable policy.
- "Oracle pair" maximizes exact two-search required-document count.
- Non-myopic gap is oracle-pair count minus the best continuation below the
  greedy first root.

## Gates

The two-task serving smoke is exactly 12 physical LLM calls. It passes only with
complete schemas and trees, zero reasoning, exact request count, and at least
three distinct first top-1 documents in each case. Failure closes the interface.

If smoke passes, the six-task opportunity stage is exactly 36 calls. It passes
only if all hold:

- mean distinct first top-1 documents >=3;
- oracle pairs retrieve at least one required document on >=4/6 tasks;
- pair gain over best one-step is >=1 document on >=3/6 tasks;
- oracle and greedy first roots differ on >=2/6 tasks;
- non-myopic gap is >=1 document on >=2/6 tasks;
- mean pair gain >=0.50 documents; and
- mean non-myopic gap >=0.33 documents.

Failure closes this exact query-tree route. Passing only authorizes a separately
preregistered target-blind depth-2 scorer on the 20 sealed tasks.

No malformed response, task, query, or threshold may be replaced. Smoke and
opportunity efficacy values cannot modify the frozen protocol.

## Budget

Before implementation, the project ledger had `$35.03984803` headroom and the
live account had `$35.03984804` above the protected `$25` Monday reserve.
OatML remains paused. Smoke is projected at `$0.20` with a `$0.75` cap;
opportunity is projected at `$0.60` with a `$2` cap. Check both balances before
each stage.
