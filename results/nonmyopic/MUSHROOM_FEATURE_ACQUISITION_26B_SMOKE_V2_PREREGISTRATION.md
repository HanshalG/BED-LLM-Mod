# Mushroom Feature Acquisition 26B Serving Smoke V2 Preregistration

Registered 2026-07-23 after S0 v1 failed its serving-format gate and before any
Mushroom proposal-quality endpoint was computed or inspected.

## V1 Failure

Job `106344` initialized `google/gemma-4-26B-A4B-it` successfully, but both bounded
attempts on the first cell repeated nested zero-valued rows until the exact
2,048-completion-token cap. Both responses ended with incomplete JSON fences. No cell
was accepted, so v1 produces no policy-quality evidence. The failure artifact records
two requests, 7,979 prompt tokens, 4,096 completion tokens, zero reasoning tokens,
zero forced exits, and zero API cost.

## Frozen Format Repair

- Use fresh smoke seed `24128` and a fresh output directory.
- Keep the same model, non-thinking temperature-zero serving, K4 machine-fixed roots,
  semantic branch probabilities, legal menus, one validation retry, scheduler, and
  all serving/mechanics gates from v1.
- Assign each legal menu option a one-character base-32 code from
  `0123456789ABCDEFGHIJKLMNOPQRSTUV`. Collected-state query roots can have up to 21
  legal follow-ups, so the larger alphabet is required even though the collection
  root itself exposes only the 17 newly unlocked specimen features.
- Flatten all branch choices, in root then branch order, into one JSON string:
  `{"choices":"..."}`. Its exact required length is stated in the prompt and enforced
  by the parser.
- Reduce the completion cap from 2,048 to 128 tokens. The required answer is shorter
  than this cap; truncation or repetition still fails closed.

The repair changes only response serialization and its token ceiling. It does not
change roots, branch supports, beliefs, feature descriptions, action legality, or the
meaning of any policy choice. V2 remains a serving/mechanics gate only. Proposal
quality is quarantined and cannot qualify or tune this smoke.
