# HotpotQA Train Future-Uplift Serving Result

The preregistered exact ten-call open-record serving gate passed every
mechanics condition.

- run:
  `hotpot-train-future-uplift-serving-20260726T050726Z`;
- public artifact:
  `results/nonmyopic/hotpot_train_future_uplift_serving/hotpot-train-future-uplift-serving-20260726T050726Z/SERVING.json`;
- physical requests and HTTP attempts: `10` and `10`;
- transport retries, reasoning tokens, and forced exits: all `0`;
- structured responses parsed without repair: `10 / 10`;
- logical calls: one initial belief, four branch refreshes, four blinded
  continuation scorers, and one final answer;
- all refreshed states differed from the initial state and from one another;
- blinded score vectors varied across branches and alignments;
- public future-score spread: `22`;
- cost: `$0.0789575`, below the frozen `$0.15` cap.

The run used the already-open validation smoke record. It did not access or
report a scientific training endpoint, and its private raw responses remain
untracked. This is transport and mechanics evidence only. With the frozen
zero-call opportunity gate already passed, it authorizes the unchanged exact
100-call confirmation under its `$1.20` cap.
