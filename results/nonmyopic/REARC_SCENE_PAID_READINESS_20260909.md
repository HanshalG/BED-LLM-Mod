# Scene-interface paid readiness

The initial zero-call preflight stopped on the second task's inventory repair
prompt exceeding the unchanged 65536-byte message cap. Its record is preserved
in rearc_scene_preflight_20260909/result.json. No API attempt or future label was
opened. Before any model responses, inventory JSON serialization was changed to
compact separators, without dropping fields, changing values or changing any
scientific gate. The earlier preflight was not overwritten.

V2 rearc_scene_preflight_v2_20260909 passes all 24 public prompt checks, maximum
57829 request bytes using maximum-length synthetic plan fields and short identity
programs. Actual generated-program repair messages remain subject to exact byte
checks before dispatch. This preflight cannot guarantee all later prompts fit.
Public SHA remains49f36fa565f2f7d4c4eecba07d6003b10e82ab87aba5006478bbaad3d4d5e090.

The new scene runner binds the frozen protocol/source/public panel, limits calls
to24, reserves .08 per attempt and the full1.92 block before launch, authenticates
again immediately before each attempt, and accounts for uncertain requests.
31 focused tests pass including both arm orders, exact24-call pass/null replay,
interruptions before/at second arm, cap enforcement, reservation and dispatch
authorization failures, execution caching, public history and source binding.
No old closed study implementation was edited.

Live account before launch:245credits/222.126794559usage/22.873205441balance.
London-day conservative spend1.70851608,remaining3.29148392, full1.92 authorized.
Exact Luna medium OpenAI route and price ceilings revalidated. No paid calls yet.
Runner will execute only once after this readiness commit is pushed. A null is
terminal under the frozen protocol, not permission for extra repairs or depth.
