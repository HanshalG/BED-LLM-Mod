# PSCon Semantic-Tree Development Smoke

## Question

Can non-myopic lookahead over LLM-generated semantic questions rank a better
first clarification than one-step EIG when the hidden intent support is externally
fixed and guaranteed to contain the truth?

This is a development smoke, not an official RegretBench result. Its design is
informed by RegretBench's hidden-intent and clarification-tree formulation, but it
uses the public PSCon source directly because the repository cited by RegretBench is
currently unavailable.

## Frozen Source

- Repository: `https://github.com/JieZouIR/PSCon`
- Commit: `42eabef33bdc7207841290fdbf4309e1a8d960f9`
- English conversations SHA256:
  `219c54ebd94bceca302c3c10b9e9b3b3c0beeda40fa4480c57c7241151bfd49d`
- English product graph SHA256:
  `d7b9dacc8c83aacaa174bb9955ec4240531076f1ff155bc4ccd07d4e53a4293b`
- Development conversation: `64937`
- Random seed: `24385`

Conversation `64937` was already opened during the zero-call structural audit. It
contains a semantically coherent TV preference revision and a final recommendation
pool with 20 title-bearing products. The liked product is in that pool. No Chinese
PSCon conversation is opened by this smoke; Chinese data remains available for a
separately frozen cross-market confirmation.

## Information Boundary

The planner receives:

- the user utterances before the final recommendation;
- all 20 candidate product titles in fixed source order;
- no liked/disliked annotation and no target marker.

The generator completes every root and followup call, and all immediate and depth-two
scores are checkpointed, before the code extracts the liked product from
`user_rating`.

The independent responder then receives only:

- the hidden liked product's title and selected product-graph metadata;
- one generated question and its numbered options.

It returns one option index. The generator never sees this target marker. The
responder never supplies a score or generates a question. Thus realized transitions
are independent of the planner's predicted product assignments.

## Models And Interface

- Semantic question/partition generator: `openai/gpt-5.4-mini`
- Independent hidden-product responder: `openai/gpt-5.4`
- Both models are explicitly non-thinking.
- Temperature: `.7` for generation and `0` for responses.
- No parse repair, normalization, continuation, or reissue is allowed.
- OpenRouter concurrency: `64`.
- Run cap: `$0.75`; projected cost: `$0.20`.
- OpenRouter is sourced from `.env`; OatML is not used.

Each generated query is strict JSON with exactly:

```json
{
  "question": "...?",
  "options": ["...", "...", "..."],
  "assignments": [1, 2, 1]
}
```

Assignments must cover every supplied candidate exactly once. Root queries use
exactly three nonempty options, all of which must be used. Followups use one to three
options depending on branch size.

## Policies And Compute

The semantic-tree bank uses:

- 5 root questions;
- 3 answer branches per root;
- 2 independently generated followups per branch.

This costs exactly `5 + 5 * 3 * 2 = 35` generator calls.

The compute-matched width bank uses 35 independently generated root questions. Its
first query maximizes immediate EIG. After the realized independent response, it
chooses the best remaining width-bank query by immediate EIG on the surviving
support. It therefore executes two questions and consumes exactly the same number of
generator calls as semantic lookahead.

The tree policies share the same generated tree and responder calls:

- `myopic`: maximize root immediate EIG, then choose the best immediate followup in
  the realized branch;
- `nonmyopic`: maximize root EIG plus expected best followup EIG;
- `random`: seeded random tree root, then choose the best immediate followup.

All EIG values are exact entropies of the LLM-generated deterministic product
partitions under a uniform 20-product prior. No LLM scores its own question.

The fixed request count is:

- generator: 70 (`35` tree plus `35` width);
- responder: 12 (`5` tree roots, `5` selected branch followups, and `2` width);
- total: 82.

## Endpoint

For each of the five tree roots, execute its independent root response and the
best predicted followup in that realized branch. Filter products by the generator's
assignments after each independent response. The endpoint is posterior mass on the
liked product:

- `1 / remaining support size` if the target remains;
- `0` if generator/responder disagreement filters it out.

This explicitly penalizes semantic simulator divergence. Root and followup
generator/responder consistency are also reported.

## Frozen Gates

The smoke passes only if every condition holds:

### Mechanics

- exactly 82 physical requests and 82 HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- source support contains exactly 20 products and the liked target;
- cost is at most `$0.75`.

### Semantic Opportunity

- at least 4 of 5 tree root questions are unique;
- immediate-EIG range is at least `.10` nats;
- depth-two-score range is at least `.10` nats;
- independent responder agrees with the generator's target assignment on at least
  80% of roots and at least 80% of selected followups;
- non-myopic and myopic choose different roots.

### First-Link Efficacy

- depth-two score versus realized endpoint Spearman correlation is positive;
- depth-two correlation strictly exceeds immediate-EIG correlation;
- non-myopic endpoint strictly exceeds myopic, compute-matched width, and seeded
  random endpoints.

Failure closes this exact one-case prompt, schema, model pair, and score. There is no
threshold repair, alternate development case, favorable-root subset, or rerun.
Passage authorizes a separately preregistered multi-case English development gate
before any Chinese confirmation. A pass is evidence for an LLM-native semantic
first link, not yet a generalization claim.

## Dry Verification

Before this preregistration:

- 5 focused unit tests passed;
- Python compilation and `git diff --check` passed;
- the deterministic full harness completed exactly 70 generator and 12 responder
  fixture requests, with zero retries/reasoning and the external target in support.

The synthetic fixture did not pass the scientific gates, as expected; it validates
mechanics only.
