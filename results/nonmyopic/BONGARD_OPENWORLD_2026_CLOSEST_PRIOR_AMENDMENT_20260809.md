# Bongard OpenWorld 2026 Closest-Prior Amendment

Frozen: 2026-08-09 13:43 Europe/London, before any August 10 Bongard model
response, candidate label, endpoint label, or terminal artifact was opened.

Status: **prospective literature-only amendment; authorizes no model request**.

## Newly Audited Priors

Two 2026 papers narrow the defensible novelty statement beyond the existing
BED-LLM and LLM-SMC-S citation amendment.

Hartmann et al., *Amortising Bayesian Experimental Design for Sequential
Information Gathering in LLMs* (arXiv:2607.03426; ICML 2026 FoGen workshop),
already regenerates and filters LLM beliefs from the current interaction history
at each turn. Its multi-turn GRPO return uses a one-question EIG horizon
`E=1` and a three-question outcome horizon `O=3`. Thus ASIG can learn
long-horizon information-gathering behavior through its policy and outcome
reward; it is inaccurate to call the whole method simply myopic. Its explicit
EIG component, however, does not give the current question information-gain
credit for the quality of answer-conditioned beliefs regenerated at a future
turn.

Qin et al., *Hypothesis Generation and Inductive Inference in Children and
Language Models* (arXiv:2605.24528), already studies online LLM-generated and
rejuvenated executable program hypotheses during sequential interaction. Its
LLM agents use the evolving hypotheses for greedy action selection, and partial
observations can change later LLM context, but it does not evaluate those
observations under an explicit multi-step EIG objective over future regenerated
hypotheses.

## Surviving Distinction

The paper must not claim first use of online LLM hypothesis generation, first
multi-turn training for LLM information gathering, or first combination of LLM
belief regeneration with sequential action selection. The narrower target is:

> explicit test-time lookahead that scores a current query partly through the
> endpoint value of answer-conditioned LLM beliefs regenerated for a later
> adaptive decision.

This is a distinction in decision objective and evaluation, not a priority
claim. The planned matched history-blind, shuffled-continuation, common-updater,
myopic, random, and classical controls remain necessary to support it.

## Exact Scope

This amendment changes only related-work and motivation prose in
`paper/main.tex`, one bibliographic record in `paper/references.bib`, and the
hash bindings in the deterministic paper renderer and mandatory paper wrapper.
The historical pre-amendment hashes remain:

- manuscript `825b74e8030fea6c44406c8ca625345a57dffbded3622e505e9f764601ba88e2`;
- references `44cd1c38638f57e6cbe9c53c3c5d94007a230686ca59949a58f49ed3b0d66768`;
- renderer `e9462325703667cb2e2133c27134643c6fc13f269bc5096768f72076124c84d1`;
- mandatory wrapper `0cca8468d39482361b118f39f174ca67b155bfd11778b0d10fe954f00e1e6edb`.

It changes no paid model, effort, prompt, image, task, split, seed, candidate
action, likelihood, policy, endpoint, threshold, gate, request count, cost cap,
authorization, result mapping, generated metric wording, or headline rule.
It makes zero model calls, opens no Bongard label or endpoint, and costs `$0`.
