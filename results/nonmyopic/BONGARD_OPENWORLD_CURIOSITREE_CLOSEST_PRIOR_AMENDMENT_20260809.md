# Bongard OpenWorld CuriosiTree Closest-Prior Amendment

Frozen: 2026-08-09 Europe/London, before any August 10 Bongard model response,
candidate label, endpoint label, or terminal artifact was opened.

Status: **prospective literature-only amendment; authorizes no model request**.

## Newly Audited Prior

Cooper et al., *The Curious Language Model: Strategic Test-Time Information
Acquisition* (arXiv:2506.09173; ICML 2025 PUT Workshop), is a materially closer
prior than the generic LLM-agent literature. CuriosiTree operates at test time,
uses prompted LLMs both as a semantic predictor and as an imperfect environment
simulator, samples possible responses to candidate information-gathering actions,
and scores those actions by expected entropy reduction relative to cost.

Its deployed action score is nevertheless one-step. The paper's equations 4--6
evaluate the entropy change caused by the current action, and its practical
surrogate rewards elimination among the current top-k semantic labels. After the
real response, it repeats the greedy procedure. It does not assign the current
action a continuation value through a second adaptive action whose semantic
support or predictive matrix is regenerated under each simulated answer.

## Surviving Boundary

The present paper must not claim first test-time LLM simulation for information
acquisition, first semantic EIG with an LLM environment simulator, or first
iterated greedy information gathering over LLM beliefs. CuriosiTree receives
explicit credit for all three.

The narrower target remains:

> explicit test-time lookahead that scores a current query partly through the
> endpoint value of answer-conditioned LLM beliefs regenerated for a later
> adaptive decision.

This is a distinction in decision objective and evaluation, not a priority
claim. It remains conditional on the preregistered dynamic-versus-myopic,
history-blind, matched-updater, shuffled-continuation, random, and classical
controls.

## Exact Scope

This amendment adds one related-work sentence to `paper/main.tex`, one
bibliographic record to `paper/references.bib`, and the corresponding fail-closed
paper-renderer bindings. It changes no paid model, effort, prompt, image, task,
split, seed, action, likelihood, policy, endpoint, threshold, gate, request count,
cost cap, authorization, result mapping, generated metric wording, or headline
rule.

The historical pre-amendment hashes remain:

- manuscript `3757e64ce787338ba589e6264ce82a0770b73d9099fb69383399625c3af41ef8`;
- references `5a3a2adc9bc5b359a002cb6e1eebb07973bcb7a5dc4b96076d5f2ec04ac2a60e`;
- Luna renderer `9cf6dc0e187330de7592a5d72ec2abeb465dc0a195a2d865ce69f8ca66497c53`;
- mandatory V6 wrapper `57fa47391f85cf66ec194a93998f2bf50a68da674709a84f58115f1b7bd5b151`.

Primary source:

- https://arxiv.org/abs/2506.09173

This amendment made zero model calls, opened no Bongard label or endpoint, and
cost `$0`.
