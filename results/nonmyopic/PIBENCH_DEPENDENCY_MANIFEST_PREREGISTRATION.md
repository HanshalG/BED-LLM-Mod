# pi-Bench Dependency-Task Manifest Preregistration

## Purpose

Freeze an untouched, persona-balanced partition of pi-Bench dependency-final
tasks before inspecting their hidden intent text, initial request text, objective
values, or task asset contents.

This manifest does not define or test a policy. It creates a clean boundary for
a subsequent zero-call mechanics study of whether prior-session or workspace
inspection can have delayed value for a later clarification decision.

## Source

- Repository: `https://github.com/Simplified-Reasoning/Pi-Bench`
- Commit: `383910b1698758a198b86037c63a111c8edc32ad`
- License: Apache-2.0
- Five personas and 100 tasks are required.
- Exactly six dependency-final tasks per persona are required.

Only episode task IDs and `depends_on` metadata determine the split. Task YAML
is loaded only to record counts and hashes. No initial request, hidden intent,
objective, or asset content is emitted.

## Split

Seed `24403` is applied in fixed persona order:

1. `Financier`
2. `law_trainee`
3. `marketer`
4. `pharmacist`
5. `researcher`

Within each persona, the first dependency-final task in episode order is the
disclosed mechanics task. The remaining five IDs are sorted, shuffled, and
assigned as two opportunity, one development, and two holdout tasks.

| Split | Size | Ordered ID SHA-256 |
| --- | ---: | --- |
| Mechanics | 5 | `ad526a0f6505eef7c3aee1580bc2261226967ea6a419d4ff1179cde70b160156` |
| Opportunity | 10 | `45fe47a534aef1a4754390450b2e76327cd5441c3ed07171f25e5edb829a43d5` |
| Development | 5 | `469755a9fb606d3d94b5d724d2d074ab7699ed88d43c94d878614034d5135947` |
| Holdout | 10 | `2eeb48e535b48e7a18b140ae4322c2c0cea2b251b7a0e421b5da311fd4588851` |

## Access Rules

- The five mechanics tasks may be inspected after the manifest is committed.
- Opportunity text and assets remain unread until a target-blind tree and
  prospective structural gates are separately frozen.
- Development and holdout text/assets/endpoints remain unread through the
  opportunity gate.
- Hidden intent text is an endpoint and must never be shown to a policy.
- A structural pass authorizes only a separately preregistered, low-cost
  development smoke.
- There are no OpenRouter calls and no OatML jobs in manifest creation.

## Rejection Rules

Stop before value access if the source revision, 100-task universe, per-persona
dependency counts, split sizes, ordered hashes, or required task files do not
reproduce. Do not repair the split after viewing hidden content.

This partition does not itself solve pi-Bench's lack of mutually exclusive
latent worlds. Any later construction must still demonstrate that the LLM is
load-bearing for semantic belief/support generation and that the early action
changes which later information action is best.
