# HotpotQA Train Future-Uplift Opportunity Result

The frozen zero-call opportunity gate passed every condition.

- public artifact:
  `results/nonmyopic/hotpot_train_future_uplift_opportunity/OPPORTUNITY.json`;
- opportunity split:
  1,000 records, ordered-ID SHA-256
  `2e63446b32879cae47876214261054ef696e153ebf621042fc5420bc7e1eb60c`;
- strict one-way directional unlocks: `320`, above the frozen `250`;
- qualifying rows with both supports in the title-BM25 top four and a
  non-enabling top root: `31`, above the frozen `20`;
- qualifying levels: 13 easy, 13 medium, and 5 hard;
- qualifying rows with neither support title in the question: `9`.

Every qualifying row has exact structural support coverage two for
enabling-first and one for answer-first. The reader materialized endpoint
columns for exactly the 1,000 opportunity IDs through an Arrow predicate.
Development, confirmation, and holdout endpoint rows remained unmaterialized.

The audit made zero model calls, cost `$0`, and used no cluster. This is
opportunity evidence only. It authorizes the frozen ten-call open-record
serving pass, not the scientific confirmation directly.
