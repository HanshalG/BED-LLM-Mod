# MuSiQue Chain-Transition Gate Preregistration

Date: 2026-07-24
Seed: `24313`
Status at registration: no model response has been requested for this apparatus.

## Question

Does opening different natural-language documents induce meaningfully different
updates in an LLM-generated support over two-hop reasoning chains, and does one-step
EIG fail to select the document that produces the best realized support?

This is a mechanism gate, not yet a policy or depth claim. It is the first gate for a
MuSiQue route in which the LLM supplies the non-enumerable semantic belief state.

## External task and frozen data

MuSiQue is constructed by composing connected single-hop questions so that later
steps depend on answers to earlier steps. The official paper reports a 30-point
single-hop F1 drop relative to multihop execution, making it a stronger structural
choice than HotpotQA for this test.

- Official repository: `StonyBrookNLP/musique`, commit
  `922ac98f19a201998dbdae6d7f2887a5258dbdeb`.
- Artifact: `musique_ans_v1.0_dev.jsonl`.
- SHA-256:
  `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`.
- License: CC BY 4.0.

Eligibility is fixed before sampling: answerable two-hop rows with 20 paragraphs;
non-yes/no answers; answer absent from the composed question and from every paragraph
title; two distinct supporting paragraphs. This yields 1,111 eligible rows.

Frozen serving-smoke IDs:

1. `2hop__554167_451128`
2. `2hop__256336_714772`

Frozen formal IDs:

1. `2hop__171254_383727`
2. `2hop__6827_55848`
3. `2hop__131818_161450`
4. `2hop__635187_861533`
5. `2hop__819974_129669`
6. `2hop__809785_606637`
7. `2hop__286268_97805`
8. `2hop__70584_198548`
9. `2hop__131275_72870`
10. `2hop__228_90265`
11. `2hop__559273_152023`
12. `2hop__486392_35739`

## Frozen interface

Gemma 4 26B A4B runs non-thinking at temperature 0. For each row it sees only:

- the composed question;
- all 20 unmarked paragraph titles;
- after an action, the title and full text of that one opened paragraph; and
- its previous candidate chains as context for refresh.

It never sees the answer, aliases, decomposition, supporting flags, or any statement
that an opened paragraph is relevant.

The model must emit exactly eight unique ordered title pairs with bridge hypotheses
and exactly six distinct first titles. Those six first titles are the candidate
actions. Opening each action supplies the deterministic stored paragraph, after which
the model regenerates eight chains. There is no LLM likelihood estimator, answerer, or
semantic judge. The BED target for this gate is the gold ordered supporting-title
pair; the final answer remains a separate downstream decode.

Immediate EIG is the binary entropy of whether each uniformly weighted chain predicts
that a candidate title is the first support. Gold-chain coverage is measured only
after generation by exact ordered title equality.

## Stages and stopping

The serving smoke is exactly 14 accepted requests: two initial supports and twelve
branch refreshes. It is mechanics-only and passes only if all schemas parse, all
branches complete, physical request count is exactly 14, and reasoning tokens are
zero. Any semantic retry or parser relaxation closes this exact interface.

The formal gate is exactly 84 accepted requests: 12 initial supports and 72 branch
refreshes. It passes only if all of the following hold:

1. all rows and branches complete with exactly 84 physical requests and zero
   reasoning tokens;
2. the true root title is among the six actions for at least 8/12 rows;
3. the gold chain is initially omitted for at least 4/12 rows;
4. at least 3 initially omitted gold chains are recovered by some branch;
5. mean oracle realized gold-chain coverage gain is at least `0.15`;
6. at least 4/12 rows have branch-dependent gold-chain coverage;
7. mean immediate-EIG regret in realized gold-chain coverage is at least `0.10`; and
8. immediate EIG has positive regret on at least 3/12 rows.

This conjunction deliberately requires both support dynamics and a reason for
lookahead. Failure stops the exact MuSiQue chain-support interface before any ranking
or policy run. Passage authorizes only a fresh, separately preregistered ranking gate
that must predict branch quality without using gold decompositions.

## Budget

The OpenRouter ledger ceiling remains `105.38480269545715`, preserving at least $25
of the live balance through Monday. The smoke has a `$0.50` run cap and the formal
stage a `$1.00` run cap. Live balance is checked before each paid stage. No OatML
cluster work is authorized.

## Serving amendment V2

The first smoke stopped after its two initial requests, before any document opening,
refresh, or endpoint. Cost was `$0.00037381`, reasoning tokens were zero, and the raw
checkpoint SHA-256 is
`21efeb855d66d76736367e5a900ee0267da5750c091fe54232be397d2e449d48`.
The flat eight-row schema failed its global diversity constraint: one response had
one unique root and six unique pairs, while the other had five unique roots.

That exact flat schema is closed. Before any V2 response, the sole repair is frozen:
emit six explicit root rows. Roots 1–2 each contain two distinct continuations and
roots 3–6 each contain one, for eight chains total. The parser flattens this structure
without deduplication or repair. Cases, model, temperature, prompts' substantive
information, observations, target, metrics, gates, request counts, and budget are
unchanged. A V2 serving failure closes this MuSiQue interface.
