# Bongard OpenWorld August 10 Final Handoff Protocol

Frozen: 2026-08-09 Europe/London, before any August 10 Bongard model
response, candidate label, endpoint label, or terminal artifact was opened.

Status: **prospective orchestration-only protocol; authorizes no new model
request or scientific branch**.

## Purpose

The frozen August 10 paid wrapper, one-shot downstream postprocessor, and
random-strategy audit are individually fail closed, but their required ordering
still depends on multiple manual commands. This protocol binds those immutable
components behind one operational entrypoint so an interruption cannot cause a
paid stage to be repeated or a required zero-call analysis to be omitted.

It changes no model, prompt, response schema, task, image, split, seed, action,
policy, likelihood, endpoint, terminal updater, request count, retry rule, cost
cap, gate, claim tier, development authorization, or headline rule.

## Bound Components And Order

The finalizer must verify exact implementation hashes before any component is
called, then execute in this order:

1. the frozen August 10 wrapper, which alone owns serving and mechanics calls;
2. the frozen V2 one-shot postprocessor, which owns disposition and, only after
   an existing mechanics pass, the classical suite, path mediation, and the
   compute-matched audit;
3. the frozen random-strategy audit, only after a complete postprocessor result
   with downstream analyses opened.

The random audit must use the mechanics result referenced by the exact wrapper
and independently authorize that stage against the wrapper. A mechanics null or
failure must not create a random endpoint artifact.

## Interruption And Replay

The finalizer may be invoked again only to validate and finish zero-call work
after an interruption. The frozen paid wrapper decides whether an existing
serving or mechanics artifact is terminal and must never repeat a banked paid
component. The postprocessor retains its existing one-shot terminal semantics.
An existing random audit or final handoff must be independently reconstructed
and compared byte-for-value against the bound inputs before it is accepted.

A downstream failure is terminal for that exact prefix. It authorizes no rerun,
repair, paid request, development stage, or claim. The final handoff records the
exact hashes of every component it accepts.

## Interpretation

This is operational evidence only. It cannot alter or rescue any scientific
result. The random control remains descriptive and not compute matched. The
existing wrapper is the sole source of any development authorization; neither
the postprocessor, random audit, nor final handoff creates authorization.
