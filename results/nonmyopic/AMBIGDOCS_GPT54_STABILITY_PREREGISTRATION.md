# AmbigDocs GPT-5.4 Likelihood-Stability Gate

## Purpose

Test whether GPT-5.4 fixes the semantic likelihood instability observed with Mini,
without sampling a target or changing the AmbigDocs source.

## Frozen Protocol

- Same pinned row 49 / `qid=43608` / six documents.
- GPT-5.4, explicitly non-thinking.
- Four question calls at temperature `.7`.
- Six classification calls at temperature `0`: one per question plus two exact
  repeats of question 0, so its identical request is classified three times.
- Exactly 10 requests, no repair/reissue/normalization.
- No target, responder, score, endpoint, test split, or OatML.
- Seed `24389`; projected cost `$0.20`; cap `$0.50`.

## Frozen Gates

All must pass:

- exact 10 requests/attempts and zero retries/reasoning/forced exits;
- every question and six-label map parses;
- all four questions unique;
- at least three unique base partitions;
- all four base partitions informative with entropy at least `.30`;
- base EIG range at least `.10`;
- all three byte-identical repeat-request maps agree exactly;
- cost at most `$0.50`.

Failure closes AmbigDocs. Passage authorizes a separately frozen development efficacy
gate with no further serving adjustment.

The deterministic source-backed fixture passes all gates, including exact repeated-map
agreement.
