# HotpotQA Causal Belief-State Smoke Preregistration

Date frozen: 2026-07-25

## Purpose

Test the first causal link required for LLM-native non-myopic BED:
does revealing different first documents induce distinct natural-language
belief states, and does a continuation scorer that can access the path only
through those beliefs select the externally annotated enabling order?

This is a public opportunity-task smoke, not a held-out policy result.

## Frozen Task And Candidates

Source and opportunity split are frozen in
`HOTPOT_DIRECTIONAL_UNLOCK_PREREGISTRATION.md`.

After the zero-call opportunity result passed, smoke selection used this fixed
rule over the already released 500 rows:

1. strict directional unlock;
2. neither support title appears verbatim in the question;
3. title-only BM25 does not select the enabling root;
4. both support documents occur in the title-BM25 top four;
5. choose the first row in frozen opportunity order.

Selected task: `5ae0036a55429942ec259bdf`.

- Root candidates are original context indices `3, 4, 6, 1`, in title-BM25
  rank order.
- The hidden answer-bearing support is index `3`.
- The hidden enabling support is index `4`.
- Title-BM25 therefore selects the answer document, giving the intended
  answer-first myopic contrast.

The model receives the question, ten candidate titles, and only the paragraph
revealed by a simulated branch. It never receives the answer, supporting-fact
annotations, support roles, selection rule, or endpoint.

## Exact Ten-Call Interface

Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.

1. One initial call sees the question and ten titles. It emits eight
   natural-language answer/support-chain hypotheses and immediate usefulness
   scores for the four frozen roots.
2. Four independent branch-refresh calls each see the question, the initial
   hypotheses, and exactly one candidate root paragraph. Each emits eight
   refreshed unresolved answer/support-chain hypotheses and no scores.
3. Four independent continuation calls score one root's nine uninspected
   titles under three belief states in the same response:
   - the correctly aligned refreshed state;
   - the cyclic next-root refreshed state;
   - the unchanged initial state.
4. One final-answer call sees only the question and the two paragraphs selected
   by the model-aware policy. Its answer is diagnostic; support coverage is the
   primary external endpoint.

The continuation scorer never sees the question, root title, root paragraph,
answer, annotations, or branch ID. Its only path-dependent input is the belief
state plus neutral candidate-title IDs and title strings.

Correct and shuffled states are blinded as A/B with seed `24351`:

- root 1: aligned B;
- root 2: aligned A;
- root 3: aligned B;
- root 4: aligned A.

Each scorer also receives unchanged initial state C. This makes belief alignment
a within-response intervention while holding candidate titles and serving
noise fixed.

Responses use flat JSON only. Scores may be JSON integers or canonical decimal
strings from 0 through 100; this representation rule is frozen prospectively.
No trailing-text extraction, key repair, content retry, or response replacement
is allowed.

## Frozen Policies

All policies use the same four roots, revealed paragraphs, branch beliefs, and
continuation score calls.

- **Myopic receding:** root with maximum initial immediate score; then the
  aligned refreshed-state continuation for that root.
- **Fixed-support d2:** immediate root score plus maximum state-C continuation;
  then state-C argmax.
- **Model-aware d2:** immediate root score plus maximum aligned refreshed-state
  continuation; then aligned argmax.
- **Shuffled-belief d2:** immediate root score plus maximum cyclic-state
  continuation; then cyclic-state argmax.
- **Random receding:** root sampled with seed `24352`; then that root's aligned
  continuation.

All ties use earlier root/candidate order. Exact endpoint is the number of the
two annotated support titles retrieved by the selected root/follow-up pair.

## Gates

The smoke passes only if all hold:

### Mechanics

- exactly 10 physical requests and 10 HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- every response has exact keys and parses without repair;
- initial and every branch state contains eight nonempty unique hypotheses;
- all four refreshed states differ from the initial state and from one another;
- A/B alignment labels are exactly balanced;
- aligned continuation scores vary within at least three roots;
- aligned and shuffled score vectors differ on at least three roots;
- aligned and initial score vectors differ on at least three roots;
- total cost is at most `$0.50`.

### Scientific Signal

- on the enabling root, aligned continuation selects the answer support;
- model-aware d2 selects the enabling root;
- model-aware support coverage is exactly `2`;
- model-aware coverage exceeds myopic coverage;
- model-aware coverage is at least fixed-support and shuffled-belief coverage;
- final diagnostic answer has nonzero token F1 against the hidden answer.

This is conjunctive. Failure closes this exact causal-bottleneck interface
before development or holdout calls. Passing authorizes a separately frozen
multi-task development gate on public opportunity rows; the 100 development
and 5,318 holdout rows remain sealed.

## Budget

- Projected cost: `$0.15`.
- Hard run cap: `$0.50`.
- Live OpenRouter balance before calls: `$46.369749126`; protect `$25` through
  Monday.
- OatML remains paused; no cluster job is involved.
