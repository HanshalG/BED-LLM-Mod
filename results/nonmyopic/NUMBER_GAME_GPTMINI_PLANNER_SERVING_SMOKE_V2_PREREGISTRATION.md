# Number Game GPT-5.4 Mini Planner Serving Smoke V2

Date frozen: 2026-07-29, after the V1 transport-clean support null and before
any response from the fresh V2 seed `53100`.

## Change From V1

V1 incorrectly gated raw rejuvenation draws even though the prospective
planner uses retained-rejuvenation. V2 evaluates the actual support-update
operator on a linked mini-tree. It also requires the model to use a private,
explicit substitution procedure before emitting each expression. V1
responses are not reused or repaired, and no efficacy is inspected.

## Frozen Design

- Model: `openai/gpt-5.4-mini`, non-reasoning, temperature `0.7`.
- Exactly ten fresh accepted calls under request seed `53100`.
- Two initial supports.
- Four linked one-observation supports: `10=YES`, `10=NO`, `42=YES`,
  `42=NO`.
- Four linked two-observation supports:
  `10=YES,20=NO`; `10=NO,20=YES`; `42=YES,75=YES`;
  `42=NO,75=NO`.
- First supports merge the relevant initial parent with the generated child;
  second supports merge the relevant first parent with the generated child.
- Raw responses are checkpointed before parsing.
- Hard run cap: `$0.10`.

## Gates

The smoke passes only if all of the following hold:

- exactly ten parsed responses, accepted requests, and HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- both initial supports contain at least 16 valid unique hypotheses;
- every conditioned LLM draw contributes at least four valid unique
  hypotheses before retention;
- every merged first support contains at least eight hypotheses;
- every merged second support contains at least four hypotheses;
- cost is at most `$0.10`.

These are mechanics gates only. Passing authorizes one prospectively frozen
fresh external-canonical confirmation; failure closes this interface.
