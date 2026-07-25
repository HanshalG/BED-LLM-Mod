# DiscoverPhysics LLM-Native BED Smoke Result

## Decision

The frozen smoke failed closed during the first tree-generation call, before
any simulator experiment, policy selection, final-law generation, or MSE
endpoint. This exact interface is closed without repair or rerun.

## Serving Result

GPT-5.4 returned a complete exact-JSON object with:

- five distinct prior hypotheses;
- four root candidates;
- two branches per root;
- five-entry posterior and regenerated-terminal beliefs;
- branch and belief probability vectors summing exactly to one; and
- zero reported reasoning tokens.

The experiment schema failed. Only `2/4` root experiments and `3/8`
continuation experiments passed the frozen schedule validator. The first two
roots ended their ten measurement times at `4.9` and `3.0`, below the explicit
minimum of `5`. Five continuation schedules had the same defect. The parser
encountered the first invalid root and stopped as preregistered.

The upstream executor would clamp internal integration duration to five, but
it would still return observations only at the model's requested times. The
minimum reported horizon was deliberately part of the frozen experimental
design, so silently accepting, padding, projecting, or regenerating these
actions would change the interface after seeing a response.

## Scientific Status

There is no myopic-versus-depth-two result:

- no shared tree was admitted;
- no root was selected;
- no external experiment was executed;
- no divergent history exists;
- no final law was generated; and
- no default or crossover-stress MSE was evaluated.

This is a structured-action serving failure, not evidence for or against
non-myopic scientific discovery. The response shows that GPT-5.4 can populate
the semantic belief tree and probability structure, but it did not reliably
satisfy the numerical experiment protocol.

No prompt emphasis, schedule projection, invalid-candidate drop, parser
relaxation, same-world retry, reasoning rescue, or selected-valid-subtree
diagnostic is used. The paper remains unchanged.

## Budget

- OpenRouter requests: `1/5`.
- Prompt tokens: `513`.
- Completion tokens: `2,260`.
- Reasoning tokens: `0`.
- Project-ledger cost: `$0.0351825`.
- Protected `$25` reserve affected: no.
- OatML use: none.

## Artifacts

- Preregistration:
  `results/nonmyopic/DISCOVERPHYSICS_LLM_BED_SMOKE_PREREGISTRATION.md`
- Public summary:
  `results/nonmyopic/discoverphysics_llm_bed_smoke/RESULT.json`
- Raw response SHA-256:
  `c14779085a413699c120b43943668fcc09514ffa1a989e3fc31dc6e542ed1fc9`
- Failure artifact SHA-256:
  `b70d5c09a9007ca829020de685b6fa10b2d1d766af715bae2073ca36af91c34f`
- DiscoverPhysics commit:
  `33b7fa9df96de9c35744efd181ca7e5a8dd60ad5`
