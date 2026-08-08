# Bongard History-Blind Estimand Clarification

Frozen: 2026-08-08, before any Bongard mechanics, development, or
confirmation response and before any scientific endpoint was opened.

This amendment narrows the interpretation of the registered
`dynamic_depth2` versus `history_blind_depth2` comparison. It changes no task,
image, model call, prompt, seed, support, score, policy action, endpoint,
threshold, gate, request count, or cost cap.

## Exact Executable Contrast

Both policies start from the same root belief. For each possible first query
and simulated binary answer:

- dynamic depth two scores the continuation using a semantic support generated
  with the simulated answer in the VLM prompt;
- history-blind depth two scores the continuation using a same-seed fresh
  semantic support generated without the simulated answer, followed by the
  same analytical Bernoulli update for that answer.

The resulting score maps may select different first queries. After the real
first answer is observed, however, both policies use the same corresponding
answer-conditioned branch belief to select their second query. Both then use
the same final-history regeneration mechanism. The original history-blind
support is not deployed as the realized updater.

Therefore the paired endpoint contrast estimates whether conditioning the
**simulated support regeneration used for first-query planning** on a possible
answer leads to better selected two-query histories than a same-seed
history-blind simulation, under a common realized answer-conditioned updater.

It does not estimate whether a deployed answer-conditioned updater produces
better realized beliefs than a deployed history-blind updater. No such
history-blind deployed-updater arm exists. Phrases such as "matched mechanism"
refer only to this first-query planning-model mechanism and must not be used to
claim a realized belief-updater effect.

## Allowed Interpretation

A passing matched family may support this statement:

> Under a common realized answer-conditioned updater, first-query planning
> with answer-conditioned simulated support regeneration outperformed
> same-seed history-blind simulated regeneration on the registered endpoint.

It may not support these broader statements:

- answer-conditioned belief regeneration is intrinsically better than
  history-blind belief regeneration;
- the realized answer-conditioned updater caused the endpoint improvement;
- a deployed history-blind updater would have produced the measured control
  endpoint.

This clarification only narrows claims. All registered matched-family gates
remain mandatory and retain exactly their original numerical definitions.
