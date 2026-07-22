# RockSample[15,15] Exact Scale Qualification

## Result

The preregistered exact structural gate passed on a frozen POBAX-generated
RockSample[15,15] diagnosis instance with 32,768 latent rock vectors. Across 100
paired 15-round trajectories, exhaustive receding-horizon d2 beat exhaustive d1
on entropy AUC by `+0.58884` nats (95% paired bootstrap CI
`[+0.58006,+0.59706]`; 100/0/0 wins/ties/losses).

The truth-anchored result agreed: d2 improved truth-log-posterior AUC by
`+0.59301` (`[+0.54702,+0.63960]`). It also reduced final entropy by `1.38824`
nats relative to d1 (`[+1.36546,+1.41002]`). All registered pairing, legality,
trace-length, and zero-LLM checks passed.

## Mechanism

At the initial belief, exact d1 value was only `0.00572`, whereas exact d2 value
was `0.02985`. The d1 policy never moved in 1,500 decisions and repeatedly
checked a weak nearby target. The d2 policy moved in 500/1,500 decisions, usually
following an information-enabling route before checking rocks with stronger
sensors. Thus the scale result preserves the intended non-myopic mechanism, not
merely an aggregate endpoint difference.

Mean entropy AUC was `10.35974` for d1 and `9.77090` for d2; mean final entropy
was `10.33102` and `8.94278`, respectively. D2 used two action sequences over the
100 trials; the dominant sequence occurred 96 times.

## Audit And Scope

The geometry, seed, endpoints, and gate were frozen before inspecting any BED
value or trajectory. The raw report is in
`rocksample_15_15_exact_qualification_20260722/REPORT.json`. A separate analyzer
reconstructed every paired endpoint and frozen-seed bootstrap interval from raw
traces; its audit passed without exceptions.

This establishes a classical planning gap at 15 rocks. It does not by itself show
that an LLM strategy proposer can recover that gap. The paid K4 h2 confirmation is
therefore separately preregistered and gated by a fresh serving smoke.
