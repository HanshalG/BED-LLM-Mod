# Semantic Object Game Bitstring Serving Result

Date: 2026-07-29

## Decision

The frozen two-call cross-provider serving gate passed every condition and
authorizes exactly one fresh mechanics tree under the preregistered seeds and
method.

## Result

- accepted requests / HTTP attempts: `2 / 2`;
- retries / provider-error retries: `0 / 0`;
- reasoning tokens / forced exits: `0 / 0`;
- cost: `$0.00626555`;
- GPT-5.4 Mini valid unique concepts: `14 / 16`;
- Gemini 2.5 Flash valid unique concepts: `15 / 16`;
- scientific endpoint accessed: `false`.

GPT-5.4 Mini had one duplicate extension and one extension outside the frozen
3--29 member range. Gemini had one extension outside that range. The local
parser rejected those rows exactly as preregistered; both supports remained
above the minimum of 12.

## Artifacts

- public serving SHA-256:
  `975303f22c1854c20a46b3b0344ada2395e4c46b24e0b664a407a897ef8dc747`;
- private raw-response SHA-256:
  `5bc9515374865468c55538e7daf5e3d16fff917b19304b5b5c109451ec56a786`.

The conditional mechanics run must hash-bind this exact public result and
cannot alter the interface, models, prompts, seeds, thresholds, or cost cap.

OpenRouter only. OatML, Slurm, SSH, and cluster use: `0`.
