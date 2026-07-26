# InfoQuest Cached-Trajectory Opportunity Result

## Verdict

The V1 cached-trajectory audit **fails closed before aggregate metrics**. The
opportunity result is unmeasured; this is not evidence for or against a
non-myopic opportunity.

## Failure

The preregistered implementation first passed its eight local tests and was
committed at `d15f9b5`. A direct-script invocation then failed at Python import
resolution before opening any source. The unchanged registered module was
invoked correctly through `python -m`.

That scientific invocation validated the hash-pinned files and explicit ID
maps, then stopped at:

```text
run0.id132.world1.history[3].content is empty
```

The frozen schema required every released message to contain a nonempty
string. No aggregate metric, threshold result, or policy comparison was
written. The exact V1 route therefore closes without parser relaxation,
trajectory exclusion, threshold change, or rerun.

This is a released-baseline data-quality failure. It does not show that
InfoQuest lacks delayed discovery or path dependence. A distinct follow-up
would require:

1. a fresh content-blind split drawn from previously sealed records;
2. a rule frozen in advance that treats an empty later utterance as a
   zero-information action rather than silently dropping the trajectory;
3. an explicit low missingness cap;
4. the same delayed-gain, shifted-answer, and cross-run-variation gates.

Opportunity content was accessed only until the first violation. Development
and effective holdout were not read. No semantic content is present in the
failure artifact.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
