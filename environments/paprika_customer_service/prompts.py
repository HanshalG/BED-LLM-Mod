"""Prompts grounded in Paprika's released customer-service protocol."""

from __future__ import annotations

from collections.abc import Sequence

from .types import PaprikaAction


def transcript_text(history: Sequence[tuple[PaprikaAction, object]]) -> str:
    if not history:
        return "(no previous turns)"
    lines: list[str] = []
    for action, observation in history:
        reply = getattr(observation, "reply", str(observation))
        lines.extend((f"Agent: {action.query}", f"Customer: {reply}"))
    return "\n".join(lines)


def hypothesis_messages(scenario: str, count: int) -> list[dict[str, str]]:
    return [{"role": "system", "content": "Generate plausible hidden causes for customer-service troubleshooting. Return strict JSON only."}, {"role": "user", "content": f"Scenario: {scenario}\nReturn exactly {count} distinct hypotheses as {{\"hypotheses\":[...]}}. Each hypothesis must state a cause and remedy; do not assume access to the private benchmark solution."}]


def refinement_messages(scenario: str, beliefs: Sequence[str], history: Sequence[tuple[PaprikaAction, object]], count: int) -> list[dict[str, str]]:
    return [{"role": "system", "content": "Refine a troubleshooting differential from new customer evidence. Return strict JSON only."}, {"role": "user", "content": f"Scenario: {scenario}\nConversation:\n{transcript_text(history)}\nCurrent hypotheses:\n- " + "\n- ".join(beliefs) + f"\nReturn exactly {count} additional or corrected cause-and-remedy hypotheses as {{\"refined_hypotheses\":[...]}}. Use only public scenario and conversation evidence."}]


def filtering_messages(scenario: str, hypotheses: Sequence[str], history: Sequence[tuple[PaprikaAction, object]]) -> list[dict[str, str]]:
    numbered = "\n".join(f"{index}: {hypothesis}" for index, hypothesis in enumerate(hypotheses))
    return [{"role": "system", "content": "Filter troubleshooting hypotheses for consistency with observed evidence. Return strict JSON only."}, {"role": "user", "content": f"Scenario: {scenario}\nConversation:\n{transcript_text(history)}\nCandidate hypotheses:\n{numbered}\nReturn {{\"keep_indices\":[...]}} containing every zero-based index still plausibly consistent. Do not discard merely because evidence is absent; discard only contradictions."}]


def candidate_messages(scenario: str, beliefs: Sequence[str], history: Sequence[tuple[PaprikaAction, object]], count: int) -> list[dict[str, str]]:
    return [{"role": "system", "content": "Propose concise customer-service diagnostic questions or corrective solution attempts and a discrete answer space. Return strict JSON only."}, {"role": "user", "content": f"Scenario: {scenario}\nCurrent hypotheses:\n- " + "\n- ".join(beliefs) + f"\nConversation:\n{transcript_text(history)}\nReturn exactly {count} candidates as {{\"candidates\":[{{\"query\":\"...\",\"kind\":\"diagnostic\" or \"solution\",\"outcomes\":[\"...\",\"...\",\"...\"]}}]}}. Every candidate needs 3-5 mutually exclusive customer-observable replies to that exact query/action. Outcomes must describe what the customer reports or observes, never a recommended next action or an unobserved diagnosis. For a diagnostic query, outcomes directly answer the requested observation. For a solution attempt, outcomes describe the result AFTER trying it and must cover at least: problem resolved, action completed but problem unchanged, and unable to perform or determine. Do not substitute pre-action conditions (for example, whether a part was dirty) for post-action results (whether cleaning fixed the problem). Include a 'not checked / cannot determine' outcome whenever a non-technical customer may not know. Use kind=solution only when the query explicitly proposes a diagnosis or corrective action that could solve the issue; inspection and information-gathering are diagnostic."}]


def likelihood_messages(hypothesis: str, action: PaprikaAction) -> list[dict[str, str]]:
    outcomes = "\n".join(f"- {outcome}" for outcome in action.outcomes)
    return [{"role": "system", "content": "Estimate a categorical answer likelihood for troubleshooting. Return strict JSON only."}, {"role": "user", "content": f"Scenario: {action.scenario}\nAssumed hidden cause and remedy: {hypothesis}\nConversation so far:\n" + "\n".join(f"Agent: {q}\nCustomer: {a}" for q, a in action.transcript) + f"\nNext agent query/action: {action.query}\nPossible customer outcomes:\n{outcomes}\nReturn {{\"probabilities\":{{outcome: probability, ...}}}} using exactly those outcome strings. Values must sum to 1."}]


def customer_messages(action: PaprikaAction, solution: str) -> list[dict[str, str]]:
    transcript = "\n".join(f"Agent: {q}\nCustomer: {a}" for q, a in action.transcript)
    return [{"role": "system", "content": f"You are the customer in this scenario: {action.scenario}\nThe private solution is: {solution}\nOnly answer what the agent asks. Be concise and non-technical. Never reveal the private solution directly. If the latest proposed diagnosis/action solves the issue, reply exactly 'Goal reached'."}, {"role": "user", "content": f"Conversation so far:\n{transcript or '(none)'}\nAgent: {action.query}\nCustomer:"}]


def mapping_messages(
    reply: str,
    outcomes: Sequence[str],
    *,
    uncertainty_forbidden: bool = False,
) -> list[dict[str, str]]:
    constraint = (
        " This is a repair pass because an explicit observation was previously mapped to uncertainty. "
        "Map only if one listed outcome is directly supported; otherwise return null with clean=false."
        if uncertainty_forbidden
        else " Select a 'not checked / cannot determine' outcome only when the customer explicitly says they did not check, do not know, cannot tell, or cannot perform the check. Never map an explicit observation to uncertainty."
    )
    return [{"role": "system", "content": "Map a customer reply to the single semantically matching proposed outcome. Return strict JSON only."}, {"role": "user", "content": f"Reply: {reply}\nOutcomes:\n" + "\n".join(f"- {value}" for value in outcomes) + "\nReturn {\"outcome\": <exact outcome string or null>, \"clean\": true or false}. Consider only the part of the reply relevant to the query; extra troubleshooting context does not invalidate a direct match. Prefer the outcome directly confirmed or contradicted by the reply." + constraint + " Use clean=false if no outcome adequately represents the reply."}]


def judge_messages(scenario: str, solution: str, query: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": "Judge whether a customer-service agent solved the released task."}, {"role": "user", "content": f"Scenario: {scenario}\nPrivate solution: {solution}\nAgent response: {query}\nReply with <VALID> only if the response diagnoses or proposes the correct solution; otherwise reply <NOTVALID>."}]
