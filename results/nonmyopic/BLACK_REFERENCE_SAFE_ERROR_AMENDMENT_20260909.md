# Preserve safe-mode rejection as an observed response

V2 executed pre-fix source but stopped at the documented Python2 safe-mode
limitation: the formatter's Python3 AST equivalence check rejects legacy print
syntax. It fetched no fixed source and produced no reference comparison.
Keep its failed_closed record. Do not change fast=False or remove any inputs.

Before any fixed source/output is accessed, V3 adds precisely the observed
`cannot use --safe with this file; failed to parse source file` AssertionError
prefix as the response category SourceAstUnsupported. Other AssertionErrors and
unexpected exceptions still terminate. This makes the actual API's safe-mode
rejection observable, rather than treating it as formatted text or bypassing it.
It changes the observation schema transparently after a pre-fix mechanics failure,
not after a scientific reference comparison. Original domain, eligibility counts,
dependencies, modes, revisions and no-paid-call restrictions are unchanged.
V3 is the amended diagnostic; V1/V2 are not scientific nulls or extra trials.
