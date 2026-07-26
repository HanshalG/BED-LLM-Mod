# VoI Medical Dynamic-Support Serving Preregistration

## Scope

This is a target-blind serving and relevance gate for the five frozen MedDG
mechanics rows. It does not score a policy, simulate branches, or evaluate a
patient endpoint.

The purpose is to establish that GPT-5.4 can generate a diverse free-form
clinical hypothesis set and discriminative semantic questions without receiving
the released diagnosis list. Passing is required before implementing the
path-dependent support tree.

## Frozen Inputs

- Manifest SHA-256:
  `70a943c1c943e7d53944cb88a4e3aff809a09219dd07da39fc38bc04a07d9c37`
- Mechanics row IDs: `474, 67, 151, 50, 284`
- Visible input: each row's `self_repo` only
- Hidden during generation: `target`, `conv_hist`, the released diagnosis list,
  other split values, and all endpoints

Source audit found that row `474` literally contains its target in the
self-report. Freeze an exact case-insensitive target-substring exclusion for
coverage reporting. The other four mechanics rows form the eligible coverage
denominator. This exclusion is fixed before model output.

## Interface

For each of five cases:

1. generate exactly six distinct free-form diagnosis hypotheses and
   one-sentence rationales;
2. show that generated differential plus the self-report to a second prompt;
3. generate exactly four distinct atomic yes/no questions without diagnosis
   names or already stated facts.

Strict line grammars are used. Raw responses are privately checkpointed before
parsing. No repair, normalization beyond whitespace/case identity, reissue, or
fallback is allowed.

## Conjunctive Gate

- exact `10` requests and HTTP attempts;
- zero retries, reasoning tokens, forced exits, and forced finalization;
- five complete six-hypothesis sets and five complete four-question sets;
- at least `20` unique hypotheses among `30`;
- at least `15` unique questions among `20`;
- exactly the one frozen target-leak exclusion;
- exact phrase coverage of the external target in at least `2/4` eligible
  initial supports;
- cost at most `$0.10`.

Target values are loaded only after every target-blind response parses and is
frozen. Coverage is a relevance gate, not a policy result.

## Budget And Consequence

- Model: `openai/gpt-5.4` through OpenRouter
- Temperature: `0.7`
- Reasoning: disabled
- Projected cost: `$0.03`
- Hard cap: `$0.10`
- Frozen allowance: `$1.13174155`
- Protected balance: `$25` through Monday
- OatML/cluster use: prohibited

Passing authorizes only a separately preregistered path-dependent tree
mechanics test on these same five rows. Failure closes this exact generation
interface with no alternate seed or parser adjustment.
