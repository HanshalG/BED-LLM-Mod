# InfoQuest Information-Need Belief Preregistration

Frozen after the all-candidate target-alignment audit and before any live
information-need response.

## Question

Can an LLM-generated belief over target-relevant unresolved information needs
rank clarification questions by realized target gain, where a conventional
LLM-generated scenario-support entropy objective was anti-aligned?

This is a first-link development gate. It is not yet a non-myopic policy test.
A pass would authorize a separate branch-regeneration experiment in which the
LLM's information-need belief changes along simulated paths.

## Bound Inputs

The gate uses the same six disclosed InfoQuest fixtures, 30 cached
root/world histories, and four remaining-root actions as the prior V3 and
target-alignment audits. It binds:

- cached-history public/private SHA-256
  `140f77447da9bd3d83e8a7be7fd45dc8fc792422d4e1cef398f6d8460d13ccc3`
  and
  `c4dec013386083afb4279754f4ffb1b0a2f6559bb2a2cf1863d80a849fe6e93e`;
- V3 public/private SHA-256
  `0cfbe3d1590001af508d351d131d12d747f2fcadf0448e2e7868809f40d968f7`
  and
  `134a9659dc5f86df3dba6e18fdc8c72a79f3524b166d2cd52d91a7fe88747e1e`;
- cached-answer diagnostic public/private SHA-256
  `9855e57d3a00afcd074e15812aa14bb987ede1bc8b09d97a9382ae8b901451f2`
  and
  `e5a128d92fa48693e3b6f28b216d3b8c314bb3c67081ad1ba6cdb485e932dc9a`;
- target-alignment audit SHA-256
  `d550cdfe9770b355c59ef999059175b79a5c7e01a8b0c3f82a96d62a98d916cf`;
- private fixture SHA-256
  `63248345976e884b9fac127033415fe68bacb3a0e32a32e498285851719251cc`.

The target proxy remains the number of immediate checklist bits supplied by a
candidate root but not by the current root. Its frozen additive agreement is
`.9467` over 300 directly judged bits. Target gain varies in 29/30 cells, and
the oracle gains `1.10` bits versus `.40` for the fixed baseline.

## LLM-Owned Belief

For each cached history, one non-reasoning, temperature-zero
`openai/gpt-5.4` call sees only:

- the ambiguous user request;
- the observed root question and answer;
- the four remaining candidate questions labeled `A-D`.

It does not receive the original scenario hypotheses, hidden setting, world,
truth packet, constraints, solution, checklist, target-gain label, prior V3
score, prior choice, or result.

The model emits exactly:

```json
{"n":[5 strings],"w":[5 integers],"a":[5 integers],"b":[5 integers],"c":[5 integers],"d":[5 integers]}
```

The five strings are distinct atomic decision-critical unknown values that
remain unresolved. Weights are integers in `1..100`. Each action array contains
five integer probabilities in `0..100` that receiving the answer would resolve
the corresponding need. Old formats, missing values, duplicate needs,
out-of-range values, repairs, inference, padding, and reissues are forbidden.

For action \(q\), exact expected resolved target mass is

\[
S(q)=\frac{\sum_i w_i r_{qi}}{100\sum_i w_i}.
\]

The deterministic argmax is selected with `A-D` order as the exact tie break.
The LLM never chooses an action directly.

## Serving Gate

The synthetic serving gate makes exactly two physical requests and HTTP
attempts. Both responses must parse, contain five distinct needs, and assign
nonconstant scores across actions. It also requires zero retries, reasoning
tokens, and forced exits, with cost at most `$0.03`.

No scientific fixture, target label, simulator, checklist judge, or endpoint is
evaluated. A serving pass alone authorizes one unchanged mechanics run.

## Mechanics Gate

The mechanics run makes exactly 30 compiler requests and no simulator or judge
request. All raw responses are checkpointed privately before parsing. Public
output contains only aggregate metrics, fixture IDs, gains, hashes, gates, and
accounting; generated semantic needs remain private.

All gates are conjunctive:

1. exact six fixtures, 30 cells, and 120 candidate actions;
2. exact 30 physical requests and HTTP attempts;
3. zero retries, reasoning tokens, and forced exits;
4. total mechanics cost at most `$0.35`;
5. nonconstant information-need scores in at least 24/30 cells;
6. defined within-cell target correlations in at least 15 cells;
7. mean within-cell Spearman correlation at least `.20`;
8. mean selected target gain at least `.15` above frozen fixed support;
9. information-need target-gain wins exceed losses against fixed support;
10. positive mean gain over fixed support in at least 4/6 fixtures;
11. at least 12/30 selected actions are target-optimal.

The frozen V3 dynamic and fixed selected gains and scores are comparators only.
No prompt or parser is changed after serving responses begin.

## Interpretation and Next Gate

A failure closes this exact information-need compiler. It does not authorize a
prompt repair, likelihood variant, reasoning variant, larger model, or depth
run on the same disclosed cells.

A pass is post hoc development evidence that the new LLM-owned representation
repairs the first link. It does not establish non-myopic efficacy. The next
experiment must be separately preregistered on fresh records and compare
branch-regenerated information-need beliefs against compute-matched myopic,
fixed-belief, and random controls using a nonsaturated target endpoint.

The deterministic oracle dry mode exists only to exercise score direction,
hash loading, request accounting, and all gates. Its outputs are excluded from
scientific evidence.

No live request is authorized during the current budget pause. The protected
pre-Monday operational allowance remains `$2.08203240`; OatML jobs remain `0`.
