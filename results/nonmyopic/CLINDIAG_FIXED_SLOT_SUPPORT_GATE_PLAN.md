# ClinDiag Fixed-Slot Support-Refresh Stability Gate

Date: 2026-07-24

Status: **preregistered before any serving call.**

## Purpose

The deterministic fixed-slot construction removes LLM-generated patient outcomes.
This gate tests the remaining load-bearing mechanism: whether GPT-5.4 can regenerate
an open-world diagnosis support stably as stored evidence arrives along a path.

This is an interface/stability gate, not an efficacy result. Passing only authorizes a
small truth-anchored structural opportunity screen.

## Frozen Cases And Path

Seed `24294` selected one fresh challenging and one fresh rare case from the static
eligible pool:

- challenging `25992750`;
- rare `rare167`.

Both cases use the same sequential path:

1. initial presentation;
2. add stored `present_illness`;
3. add stored `lab_1`;
4. replay the exact step-3 prompt through an independent adapter.

Every generated support must contain exactly 12 distinct diagnoses. The model sees
only the initial presentation, ordered stored observations, and its previous generated
support. The harness never inserts the final diagnosis, benchmark title, or answer
options. A diagnosis may reappear only because the generator produced it itself.

## Models And Calls

- support generation: `openai/gpt-5.4`, reasoning disabled;
- semantic measurement: `openai/gpt-5.4-mini`, reasoning disabled;
- temperature: `0.5` for support generation and `0.0` for measurement;
- structured retries: zero;
- expected physical requests: exactly 10;
- OpenRouter run ceiling: `$0.50`;
- projected spend reserved by the ledger: `$0.15`;
- live account balance must be checked immediately before launch.

The ten calls are eight support generations, four per case, followed by one joint
semantic audit per case.

## Frozen Endpoints

The independent audit scores truth equivalence for all four supports. It also measures
semantic overlap in both directions between the original and duplicate final supports.
The case overlap is the worse directional fraction. Related diseases and broad parent
categories do not count as matches.

All gates must pass:

1. exactly 10 physical requests;
2. zero reasoning tokens and zero retries;
3. every support has size 12;
4. duplicate prompts are byte-for-byte equal;
5. neither selected stored source path contains the hidden target literally;
6. duplicate semantic overlap is at least `0.80` for both cases;
7. original-versus-duplicate truth-score gap is at most `0.05` for both cases;
8. no parser or runtime failure.

Truth-score changes along the path are descriptive and are not required to improve in
this serving smoke.

## Decision Rule

Pass authorizes a small fixed-slot one-step versus ordered two-step oracle opportunity
screen with truth used only after support generation. Failure stops this exact serving
interface; no larger policy or structural run follows.
