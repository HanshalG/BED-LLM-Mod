# Animals Stratified Implicit-Prior Target Protocol

Status: **frozen before sampler responses**.

The iid implicit-prior sampler collapsed to 34 common names. This distinct
protocol defines an explicit equal-component semantic mixture over eight
taxonomic strata:

1. mammals;
2. birds;
3. reptiles;
4. amphibians;
5. fish;
6. insects;
7. arachnids and myriapods;
8. mollusks, crustaceans, echinoderms, and cnidarians.

One non-thinking Gemma 4 26B call at temperature `1.0` requests exactly 16
names from each stratum. Outputs receive unchanged structural cleanup,
case-insensitive deduplication, and animal-name validation. Shuffle seed
`24285` assigns the first 20 unique names to development, the next 60 to an
untouched holdout, and leaves any remainder unused.

Pass requires at least 80 unique validated names in this one attempt, zero
reasoning, and cost below `$0.25`. No replacement call, manual name editing, or
endpoint-conditioned selection is allowed. If it passes, branch belief
generation must use the same eight strata before policy evaluation.
