# InfoQuest Support-Causal-Link Serving Result

## Verdict

The four-model support-causal-link serving gate **fails closed on call 9**.
The disclosed-case mechanics run is not authorized and was not started.

This is a synthetic serving/instrument failure. It provides no evidence for or
against support recovery, root ranking, dynamic continuation, or InfoQuest
policy efficacy.

## Failure

The first shell invocation did not export `.env` variables and stopped before
constructing an adapter. It made zero requests and exposed no response. The
unchanged committed gate was then invoked with the key exported.

The live serving run completed:

- two GPT-5.4 initial-support calls;
- two Gemini-2.5-Flash root-simulator calls;
- two GPT-5.4 refresh calls;
- one GPT-5.4 fixed-follow-up call;
- one Gemini-2.5-Flash follow-up call;
- one Gemma4-26B support-judge call.

The Gemma support-judge response violated the frozen six-line grammar. The
checklist-judge call was never made. The stage checkpointed responses after
parsing, so the malformed support text is not retained in the private raw
artifact and cannot be diagnosed or reparsed.

| Accounting | Observed |
| --- | ---: |
| Physical requests | 9 |
| HTTP attempts | 9 |
| Retries | 0 |
| Reasoning tokens | 0 |
| Forced exits | 0 |
| Cost | `$0.02862995` |

Private raw SHA-256:
`9ca693eee861a4811fd71df31bb1e9b149ee649ed6b572d05875b734b079401d`.

Per registration, there is no parser relaxation, response reissue, model swap,
or mechanics run on this line. No disclosed scientific fixture and no
opportunity/development/holdout record was evaluated.

## Next Route

The hidden-truth support judge added subjectivity and a fragile transport
boundary without being necessary for the core causal question. A distinct
preregistered test can hold each first root and hidden-user answer fixed, then
compare:

- a follow-up chosen after LLM support regeneration;
- a follow-up chosen from the unchanged initial support.

The official checklist endpoint can score both continuations directly. This
tests whether path-dependent LLM belief regeneration improves the next action,
without an oracle support score or hidden-truth ranker.

OpenRouter spend leaves `$3.97137005` of the self-imposed pre-Monday `$4`
allowance. The live provider credit endpoint still lags and reports `$37.0145`
remaining. OatML jobs: `0`.
