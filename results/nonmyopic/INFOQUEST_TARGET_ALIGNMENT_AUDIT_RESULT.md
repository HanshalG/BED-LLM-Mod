# InfoQuest Target-Alignment Audit Result

## Status

The deterministic post hoc audit completed exactly and failed the two frozen
target-alignment gates. The opportunity, proxy-validity, and accounting gates
all passed.

Public audit SHA-256:
`d550cdfe9770b355c59ef999059175b79a5c7e01a8b0c3f82a96d62a98d916cf`.

## Protocol

The audit binds the preregistered V3 exact-EIG and cached-answer artifacts and
uses no model, provider, cluster, or private semantic-text output. It evaluates
all four candidate actions in all 30 root/world cells across six disclosed
fixtures. Candidate target gain is the number of immediate checklist bits
provided by the candidate root but not by the current root.

This is a post hoc development audit, not fresh confirmatory evidence.

## Results

The additive target-gain proxy agreed with 284/300 (`.9467`) directly judged
selected-path bits. Candidate target gain varied in 29/30 cells. The target
oracle gained `1.10` bits per cell, compared with `.40` for the fixed-support
selection, leaving `.70` bits of mean oracle headroom.

Despite that opportunity, dynamic-support EIG was negatively aligned with
target gain. Its mean within-cell Spearman correlation was `-.1444` over 26
defined cells, below the frozen `.20` gate. Dynamic selections gained `.30`
bits per cell and had `.80` mean oracle regret, compared with `.40` gain and
`.70` regret for fixed support. Dynamic versus fixed target gain was
`3/21/6` wins/ties/losses. Dynamic and fixed choices were target-optimal in
8/30 and 10/30 cells, respectively.

Exact accounting was zero LLM calls and `$0` cost.

## Interpretation

The action bank contains abundant target-relevant opportunity, so lack of
candidate headroom does not explain the failed first link. Instead, the current
semantic-support entropy objective rewards distinctions that are poorly aligned
with the official target. This agrees with the qualitative loss inspection:
broad goal, domain, and constraint questions can have high internal entropy
while concrete audience, timing, outcome, and routine questions reveal more
checklist information.

Further InfoQuest work should therefore change the belief representation toward
target-relevant information needs before revisiting likelihood prompting,
rollout depth, or policy-scale experiments.
