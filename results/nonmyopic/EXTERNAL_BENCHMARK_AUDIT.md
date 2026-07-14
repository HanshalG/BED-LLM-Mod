# External Benchmark Endpoint Audit

Audited: 2026-07-15. This audit precedes any LLM pilot. It makes no model calls
and does not use either package's reported outcome as evidence.

## Sources Inspected

| Package | Pinned source inspected | Commit | Claimed 20Q datasets |
| --- | --- | --- | --- |
| UoT | `https://github.com/zhiyuanhubj/UoT` | `e08f17ff4ceec0f734da66e04d7d35df04ae6460` | `bigbench`, `common`, `thing` |
| MISQ-HF | `https://github.com/harshita-chopra/misq-hf` | `77421f21ce28fc5e7829207391ec7a653d865a11` | `bigbench`, `common`, `thing` |

The repositories were cloned read-only to `/tmp` for this audit. Both expose
target strings in their 20 Questions task modules, but neither supplies a
scripted truth-table answerer or an exact target-decode evaluator.

## Endpoint Findings

1. **The answerer is an LLM judge, not ground truth.** Both runners default to
   `--examiner_model gpt-4`. Each 20Q turn passes the hidden target and dialogue
   to that examiner, then parses free-text `Yes`/`No` prefixes. This makes
   response likelihood and task feedback model-mediated.
2. **The nominal success signal is a phrase emitted by that judge.** UoT marks
   success when the examiner output contains `guessed it` or `are right.`
   (`src/uot/method.py:116-118`); MISQ-HF uses the same test
   (`src/misq/method.py:343-344`). Neither compares a parsed final decode to the
   released target string.
3. **Both packages then overwrite the outcome to success whenever the dialogue
   ended before the maximum turn.** UoT assigns `state = 1` at
   `src/uot/method.py:137-139` (and its naive path at 192-194); MISQ-HF does the
   same at `src/misq/method.py:386-387` (and its naive path at 445-446). This
   means the saved success-rate endpoint is not a target-decode accuracy metric.
4. **Their planning supports are not fixed deployment supports.** In open-set
   mode, both packages ask an LLM to initialize and renew the possibility set;
   UoT documents a three-turn pre-ask stage and per-round possibility updates.
   This is an important research setting, but it cannot provide the exact
   branch/deployment equivalence required for the planned controlled comparison.

## Classification and Consequence

| Candidate source | Target ground truth | Answerer | Endpoint | Confirmatory status |
| --- | --- | --- | --- | --- |
| UoT 20Q as released | target string exists | GPT-4 free text | judge phrase plus early-stop overwrite | **Not eligible** |
| MISQ-HF 20Q as released | target string exists | GPT-4 free text | judge phrase plus early-stop overwrite | **Not eligible** |
| UCI Zoo frozen-matrix 20Q | released row identity | deterministic trait lookup | exact MAP identity decode | **Eligible for controlled pilot** |

UoT and MISQ-HF remain relevant prior work: they motivate planning-based
information seeking and supply a useful method comparison target for future work.
They will not be rerun or used as an arm in the immediate confirmatory program.
The next authorized route is the exact UCI Zoo matrix, where the LLM is
load-bearing only for constrained legal-candidate proposal and every answer,
posterior update, branch, and decode is deterministic and auditable.
