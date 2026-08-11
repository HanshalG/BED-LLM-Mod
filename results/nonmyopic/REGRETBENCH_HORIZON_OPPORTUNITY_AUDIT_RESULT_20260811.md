# RegretBench Horizon-Opportunity Audit Result

Status: **structural opportunity null**.

The exact source audit covered all `2,419` eligible official RegretBench OpenDomainQA test CIGs at the pinned source commit. All `6,286` published test-file checksums passed. No model response, policy endpoint, or saved endpoint outcome was opened; cost and model calls were zero.

## Result

Under a uniform prior over each task's official hidden intents, every semantic ask facet was evaluated as a deterministic partition by its official slot value. For each task, the audit compared:

- the depth-two information obtained after the greedy one-step first action; and
- the maximum depth-two information over all possible first actions, with an optimal different second action chosen separately in each first-answer branch.

The strictly positive depth-two first-action gain count was **0/2,419**, with maximum gain `0.0` nats. The result remained exactly zero in every stratum of 2--4 facets and 3--6 intents.

The same result held after excluding all four previously frozen 132-task cohorts. Among the `759` untouched tasks with exactly four executable canonical actions, the positive-gain count was **0/759** and the maximum gain was `0.0` nats.

## Interpretation

RegretBench's official fixed-support CIG provides no environment-grounded depth-two reason to choose a different first action from greedy. An LLM can still regenerate different hypotheses after hypothetical answers, but any resulting horizon preference would not correspond to an independent opportunity in the benchmark's true intent/facet structure. Given the repeated semantic-interface failures, pursuing that difference would risk measuring regeneration noise rather than non-myopic experimental design.

This closes RegretBench as the primary headline route. It does not show that non-myopic planning generally fails, and it does not invalidate RegretBench as a useful semantic-interface diagnostic. A successor environment must expose a real, source-verifiable horizon opportunity before any LLM call, while still requiring the LLM to generate semantic hypotheses or likelihoods.

## Provenance

- protocol SHA-256: `6f708ba0a2056d15acc602f10a62f4377bf794398dea9c0910753c8162611f61`;
- audit implementation SHA-256: `57138a6ff6f6ae678312569ceb7f322edc57b036d7ae479f5f115728ad9e7597`;
- result SHA-256: `b788ffa852936e70176e124c300dad50f921b7471ff8630666ce62467eb597f5`;
- focused test SHA-256: `c137b4e0772f8a802563b467a25472ce95e8cef4a1864cb387baf19b89c254eb`;
- focused audit test: `1/1` passed;
- typed-action regression tests: `26/26` passed.
