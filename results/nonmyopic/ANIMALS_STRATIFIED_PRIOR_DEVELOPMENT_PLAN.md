# Animals Stratified-Prior Development Plan

Status: **frozen before policy responses on the fixed development split**.

Seed `24286` uses the 20 development targets fixed in the independently
sampled seed-24285 prior pool. Target names never enter policy prompts.

Every initial and counterfactual belief generation uses one non-thinking Gemma
4 26B call for each of the same eight taxonomic mixture components used to
construct the target prior. Up to eight names per component are merged before
unchanged validation, deduplication, and history filtering. Three production
candidates and the unchanged target-blind branch-content ranker are used.

Development passes only if:

1. all 20 states complete;
2. at least 12 targets occur in a six-branch union;
3. at least six targets absent initially are recovered by a branch;
4. ranker candidate Spearman is positive and exceeds immediate EIG;
5. ranker selected coverage exceeds EIG;
6. ranker wins exceed losses;
7. serving uses zero reasoning and remains below `$4`.

Pass authorizes an independently preregistered run on the untouched 60-target
split. Failure stops this exact stratified Animals formulation.
