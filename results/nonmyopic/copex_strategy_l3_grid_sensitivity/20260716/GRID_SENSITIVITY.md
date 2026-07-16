# COPEx L3 Grid-Resolution Sensitivity

This sensitivity makes zero LLM calls and reuses the formal seed schedule, truths, particles, initial positions, realized noise, and matched scorer budget.

| Angular resolution | Full d2 sequences | Evaluated sequences | Grid final entropy | Strategy gain | 95% paired CI | W / T / L |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 4 | 16 | 8 | 0.001422 | +0.001397 | [+0.000001, +0.003619] | 28 / 0 / 2 |
| 8 | 64 | 8 | 0.000175 | +0.000150 | [+0.000001, +0.000400] | 28 / 0 / 2 |
| 16 | 256 | 8 | 0.004803 | +0.004778 | [+0.000029, +0.012445] | 30 / 0 / 0 |
