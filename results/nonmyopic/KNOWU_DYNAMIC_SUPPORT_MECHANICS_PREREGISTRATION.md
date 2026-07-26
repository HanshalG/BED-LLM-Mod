# KnowU Dynamic-Support Mechanics Preregistration

Date frozen: 2026-07-26

## Question

On the six opened KnowU mechanics worlds, can an LLM-generated semantic
hypothesis support recover a true task-relevant preference state that was
missing before an informative, atomic clarification?

This is a gate on the first causal link required by the proposed non-myopic
method. It is not a policy-comparison result and does not evaluate task
completion.

## Frozen substrate

- KnowU source commit:
  `c03a825991ede13add6631f2ed19b90755930dc6`
- Frozen V2 manifest mechanics families:
  `BuyComputerPreferenceAskUserTask` and
  `MattermostLeaveNoticeTask`
- Three supported official profile worlds per family; uniform prior within
  family.
- Profile labels are hidden from all support and question prompts.
- Initial evidence is eight clean behavior logs selected by a frozen,
  task-specific term-frequency rule. Profile YAML is not visible to the
  support policy.
- Model: `openai/gpt-5.4`, temperature 0, reasoning disabled.

Exact private fixture SHA-256:
`2cb9d0e34e3e13aee896b5d4f2c6bf0c470d1a92096a23128f6d8cc26905eff8`.
Public fixture metadata records hashes of every visible-log packet and private
truth packet without publishing private endpoint content.

## Frozen interaction

For each of six worlds:

1. Generate four coherent semantic preference-state hypotheses and four
   distinct atomic clarification questions in one call.
2. Use one profile-conditioned user-simulator call to answer all four
   questions independently.
3. Regenerate four hypotheses after each answer. Each of the 24 branch
   refreshes is a separate physical request containing exactly one
   question-answer pair, preventing sibling-branch leakage.
4. After all support-generation calls are complete, use one semantic endpoint
   call per world to score truth coverage of the initial and four refreshed
   supports.

Total mechanics requests: 42. No repairs or reissues are allowed.

## Parsing and endpoint

- Strict OpenRouter JSON-schema output is required.
- Every support has exactly four distinct hypotheses.
- Every question has one terminal question mark, a unique dimension, no
  profile/persona label, no conjunction joining independent dimensions, and
  no request for a complete product, order, or message.
- The semantic endpoint selects the best support hypothesis and scores joint
  coverage from 0 to 100.
- Truth is present iff the endpoint score is at least 70 and its best index is
  1 through 4. Partial states omitting a required task dimension must score
  below 70.

Primary mechanics gates:

- at least one world lacks truth in its initial support;
- in at least one such world, truth enters after at least one atomic question;
- at least one branch has positive truth-support score gain;
- all 42 requests and exactly 42 HTTP attempts complete with zero retry,
  reasoning token, forced exit, repair, or reissue;
- total mechanics cost is at most $0.75.

Report, but do not gate on, per-world root support-gain range and which atomic
questions caused truth entry.

## Serving gate and spending

Before mechanics, run an exact 10-request synthetic strict-schema serving
gate covering all four response types. It evaluates no scientific endpoint,
allows no retry, and costs at most $0.10.

OpenRouter is the only permitted execution backend. No OatML job is permitted.
Reasoning is reserved for a later naive-thinking baseline, not the method.

If serving fails, mechanics is not run. If mechanics fails a scientific gate,
this exact KnowU first-link construction is closed without relaxing criteria
or inspecting opportunity/development/holdout content.
