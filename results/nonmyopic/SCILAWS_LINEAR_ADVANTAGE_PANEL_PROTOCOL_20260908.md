# Linear-advantage interval panel

Predecessor workload e95c4ceace0c94647e98c4e461e802e4ab4d19bc02efedf44f530ec3cb10402c
passes all3 scenarios with differences<=4.06e-15. Evaluation counts old/new:
zero1200/1200,affine1680/1380,quadratic1620/1320. Five tests pass in1.09s.

Prospectively run exactly the prior24-case interval-refinement panel, replacing only
the independent terminal integral representation with explicit LinearAdvantageReference.
Same initial observations, all eight roots and continuation actions, outer4, posterior,
5second/100000 counted evaluation caps, single-thread BLAS,5e-5 terminal uncertainty,
1e-7 maximum reported reference error. No retry or subset exclusion. Preserve all
partial roots/failures. Full24 completion required for operational coverage.

Original panel remains untouched. This tests operational coverage, not outer
integration error, source calibration, or h3 efficacy. No source/model calls.
