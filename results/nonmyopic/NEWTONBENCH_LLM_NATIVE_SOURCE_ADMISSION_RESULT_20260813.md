# NewtonBench LLM-Native Source Admission Result

Date: 2026-08-13

Status: **source failed closed; exact release closed as a non-myopic LLM-native
route.**

## Result

NewtonBench is a complete interactive scientific-law benchmark: its official release
declares 324 tasks across 12 physics domains, three law difficulties, three law
versions, and three system complexities. Agents can request experiments and submit a
machine-evaluable Python law.

The frozen source audit passed the binding/population and native-interaction gates,
then failed the first horizon gate. The public prompt explicitly allows one
`run_experiment` action to contain arbitrarily many parameter settings; in the default
interface all measurements are noiseless, perfectly accurate, and deterministic.
There is therefore no native measurement budget forcing a first experiment to enable
or improve a later one.

Source inspection also found a complete three-by-three `LAW_REGISTRY` in every one of
the 12 domains. This corroborates that the evaluated target support is benchmark-owned
and finite, but the stop-at-first-failure protocol did not formally advance to the
open-support gate.

| Gate | Result |
|---|---|
| immutable source and 324-task population | pass |
| native interactive experiment | pass |
| finite-budget horizon | **fail** |
| open-support ownership | not evaluated |
| LLM irreducibility | not evaluated |

## Interpretation

NewtonBench is valuable for interactive law discovery and scientific reasoning. This
result does not evaluate any agent or law-recovery endpoint. It establishes only that
the exact released interface cannot identify a non-myopic first-link benefit: one
action can already acquire an arbitrary noiseless design, while all target laws are
listed in source.

No task was executed, no target law was invoked, no saved trajectory or result was
read, and no candidate law or endpoint was evaluated. Model calls, OpenRouter cost,
and cluster jobs were all zero. Adding an action limit after observing the interface
would create a new benchmark rather than evaluate the released one.

## Integrity

- protocol SHA-256:
  `b408a36f94c4750c352662f4c7e424cf0b23ef973608e5e53dd94c49178e5ad4`;
- aggregate result SHA-256:
  `82c0d6a948fe0f372e2855a4d807a71bcf6f68d2705d2eb5a88a821f524e3d1e`;
- implementation SHA-256:
  `db8d776343a02631b63ac0fcf9b1197c13610f573f3b1bb00e4016cdb1eefbc8`;
- focused test SHA-256:
  `391cc6e2c6b842b87ab51a90cec998ace6103a3d38506327cfcbd390ddc03a53`;
- official commit/tree:
  `912a4ba5f4356ddd06acc16e44460ca30be4abc2` /
  `88b68ea14e1ee6bc17a237642ab9eccde52faa48`.
