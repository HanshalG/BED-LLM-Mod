# Bamboogle No-Search Memory Screen Result

## Decision

The frozen no-search screen passes every gate. Bamboogle is not saturated by
non-reasoning GPT-5.4 memory, so a separately committed cached-search mechanics
test is authorized.

## Mechanics

- Logical / physical requests: `25 / 25`.
- HTTP attempts: `25`.
- Retries / reasoning tokens / forced exits: `0 / 0 / 0`.
- Exact parsed responses: `25 / 25`.
- Search requests: `0`.
- Cost: `$0.0077025`.
- Public artifact SHA-256:
  `7f0a974fdfaaf17d08ba7cc7b0db232de331b4af78cbf97eb64522c0dfa5e321`.
- Live OpenRouter balance afterward: `$40.459511484`.
- OatML use: none.

The first shell invocation stopped before adapter construction because `.env`
variables were not exported. It made zero requests and changed no protocol
field. The successful invocation used shell auto-export only.

## Results

| Metric | Result | Gate |
| --- | ---: | ---: |
| Correct modal answers | `1/5` | `<=3/5` |
| Exact sample accuracy | `.20` | `<=.70` |
| Tasks below `.80` gold support | `4/5` | `>=2/5` |
| Mean answer entropy | `.5220` nats | diagnostic |

Four tasks had zero gold-answer support across all five samples. The one
correct task was correct on all five samples. One incorrect task produced five
distinct normalized answers; two produced two answers; one was confidently
wrong with a single answer.

## Interpretation

The benchmark leaves substantial room for evidence acquisition. It also
contains both epistemic diversity and confident error, so entropy reduction
alone may reward confidently wrong branches. The cached-search mechanics test
must therefore report target-blind entropy and external gold-support mass
separately, and must stop if evidence does not change generated support.

Opportunity, development, and holdout values remain sealed.
