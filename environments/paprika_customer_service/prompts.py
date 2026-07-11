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


def candidate_messages(
    scenario: str,
    beliefs: Sequence[str],
    history: Sequence[tuple[PaprikaAction, object]],
    count: int,
    *,
    prompt_mode: str = "standard",
) -> list[dict[str, str]]:
    if prompt_mode == "standard":
        system = (
            "Propose concise customer-service diagnostic questions or corrective solution "
            "attempts and a discrete answer space. Return strict JSON only."
        )
        objective = ""
    elif prompt_mode == "best_n":
        system = (
            "Propose the best customer-service diagnostic questions or corrective solution "
            "attempts for resolving the issue quickly, with a discrete answer space. Return "
            "strict JSON only."
        )
        objective = (
            f"Return your {count} best distinct next actions for resolving the customer's "
            "issue as quickly as possible. "
        )
    else:
        raise ValueError(f"Unsupported Paprika candidate prompt mode: {prompt_mode}")
    return [{"role": "system", "content": system}, {"role": "user", "content": f"Scenario: {scenario}\nCurrent hypotheses:\n- " + "\n- ".join(beliefs) + f"\nConversation:\n{transcript_text(history)}\n{objective}Return exactly {count} candidates as {{\"candidates\":[{{\"query\":\"...\",\"kind\":\"diagnostic\" or \"solution\",\"outcomes\":[\"...\",\"...\",\"...\"]}}]}}. Every candidate must contain one atomic question or corrective action, never multiple checks joined by 'and' or 'or'. Every candidate needs 3-5 mutually exclusive customer-observable replies to that exact query/action. Outcomes must describe what the customer reports or observes, never a recommended next action or an unobserved diagnosis. For a diagnostic query, outcomes directly answer the requested observation. For a solution attempt, outcomes describe the result AFTER trying it and must cover at least: problem resolved, action completed but problem unchanged, and unable to perform or determine. Do not substitute pre-action conditions (for example, whether a part was dirty) for post-action results (whether cleaning fixed the problem). Always include a 'not attempted / cannot determine' outcome. Use kind=solution only when the query explicitly proposes a diagnosis or corrective action that could solve the issue; inspection and information-gathering are diagnostic."}]


def arbitration_candidate_messages(
    scenario: str,
    history: Sequence[tuple[PaprikaAction, object]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Act as the native customer-service troubleshooting policy. Propose your natural "
                "next action and two credible alternatives. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Scenario: {scenario}\nConversation:\n{transcript_text(history)}\n"
                "Return exactly 3 candidates as {\"candidates\":[{\"query\":\"...\","
                "\"kind\":\"diagnostic\" or \"solution\",\"outcomes\":[\"...\",\"...\","
                "\"...\"]}]}. Candidate 0 must be the single action you would naturally take "
                "next. Candidates 1 and 2 must be distinct credible alternatives. Each candidate "
                "must be atomic and have 3-5 mutually exclusive customer-observable outcomes, "
                "including not attempted / cannot determine."
            ),
        },
    ]


def likelihood_messages(hypothesis: str, action: PaprikaAction) -> list[dict[str, str]]:
    outcomes = "\n".join(f"- {outcome}" for outcome in action.outcomes)
    return [{"role": "system", "content": "Estimate a categorical answer likelihood for troubleshooting. Return strict JSON only."}, {"role": "user", "content": f"Scenario: {action.scenario}\nAssumed hidden cause and remedy: {hypothesis}\nConversation so far:\n" + "\n".join(f"Agent: {q}\nCustomer: {a}" for q, a in action.transcript) + f"\nNext agent query/action: {action.query}\nPossible customer outcomes:\n{outcomes}\nReturn {{\"probabilities\":{{outcome: probability, ...}}}} using exactly those outcome strings. Values must sum to 1."}]


def customer_messages(action: PaprikaAction, solution: str) -> list[dict[str, str]]:
    transcript = "\n".join(f"Agent: {q}\nCustomer: {a}" for q, a in action.transcript)
    return [
        {
            "role": "system",
            "content": (
                "You are going to role-play as a customer experiencing a specific issue.\n\n"
                "You must adhere to the following guidelines:\n"
                "1. Only respond to the agent's questions and provide relevant information when prompted.\n"
                "2. Your responses should be concise and reflect a typical, non-technical customer's knowledge level.\n"
                "3. You know the private solution, but never reveal it directly.\n"
                "4. Determine the result of each proposed check or remedy from the true underlying problem.\n"
                "5. Reply with 'Goal reached' and nothing more if the agent has reached or solved the issue.\n\n"
                f"Scenario: {action.scenario}\n"
                f"Private solution: {solution}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{transcript or '(none)'}\n"
                f"Agent: {action.query}\n\n"
                "Remember: decide the result using the private solution. If this suggestion fixes "
                "the issue or identifies the correct remedy, reply exactly 'Goal reached'. If it "
                "does not, give a truthful concise customer response and do not reveal the remedy.\n"
                "A merely plausible alternative fix is not enough: the latest action must directly "
                "implement or identify the specific private cause/remedy.\n"
                "Customer:"
            ),
        },
    ]


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


def success_judge_messages(
    scenario: str,
    solution: str,
    conversation: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    transcript = "\n".join(
        f"Agent: {query}\nCustomer: {reply}" for query, reply in conversation
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the judge of a customer-service agent trying to help a customer in a "
                "particular scenario. Verify whether the agent has solved the problem correctly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Scenario: {scenario}\nPrivate solution: {solution}\n\n"
                f"The complete conversation begins here:\n{transcript}\n"
                "(End of conversation)\n\n"
                "Reply with <VALID> only if the conversation shows that the agent identified or "
                "suggested the specific correct remedy and the customer confirms the issue is "
                "resolved. A promise to try, a rejected attempt, or an unconfirmed suggestion is "
                "<NOTVALID>."
            ),
        },
    ]


def faithfulness_messages(
    action: PaprikaAction,
    solution: str,
    reply: str,
) -> list[dict[str, str]]:
    transcript = "\n".join(f"Agent: {q}\nCustomer: {a}" for q, a in action.transcript)
    return [
        {
            "role": "system",
            "content": "Audit whether a simulated customer's reply is faithful to a private ground-truth solution. Return strict JSON only.",
        },
        {
            "role": "user",
            "content": (
                f"Scenario: {action.scenario}\nPrivate solution: {solution}\n"
                f"Previous conversation:\n{transcript or '(none)'}\n"
                f"Latest agent action: {action.query}\nSimulated customer reply: {reply}\n\n"
                "Return {\"consistent\": true or false}. A reply is inconsistent if it contradicts "
                "the private cause/remedy, claims an incorrect remedy solved the issue, says the "
                "specific correct remedy failed, reveals the private solution without being asked, "
                "or merely promises to try the correct remedy instead of returning 'Goal reached'."
            ),
        },
    ]


def terminal_faithfulness_messages(
    action: PaprikaAction,
    solution: str,
) -> list[dict[str, str]]:
    transcript = "\n".join(f"Agent: {q}\nCustomer: {a}" for q, a in action.transcript)
    return [
        {
            "role": "system",
            "content": (
                "Strictly audit a benchmark customer's terminal success claim against the private "
                "ground truth. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Scenario: {action.scenario}\nPrivate solution: {solution}\n"
                f"Previous conversation:\n{transcript or '(none)'}\n"
                f"Latest agent action: {action.query}\n\n"
                "The simulator wants to reply 'Goal reached'. Return "
                '{"terminal_consistent": true or false}. Use true only when the latest action '
                "directly implements or identifies the specific private cause/remedy. A different "
                "plausible fix for the same symptom is false. For example, straightening a drain "
                "hose is false when the private cause is a clogged hose; clearing that clog is "
                "true. Securing a connector is true when the private cause is a loose connector."
            ),
        },
    ]


def faithfulness_repair_messages(
    messages: Sequence[dict[str, str]],
    rejected_reply: str,
) -> list[dict[str, str]]:
    return list(messages) + [
        {"role": "assistant", "content": rejected_reply},
        {
            "role": "user",
            "content": (
                "That reply contradicted the private ground truth. Regenerate only the customer's "
                "reply. If the agent reached or suggested the correct remedy, reply exactly 'Goal "
                "reached'. Otherwise give a concise truthful result consistent with the private "
                "solution, without revealing that solution. Do not accept a merely plausible "
                "alternative fix; it must directly match the specific private cause/remedy."
            ),
        },
    ]
