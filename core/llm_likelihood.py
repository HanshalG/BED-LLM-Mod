"""Shared LLM-based binary observation likelihood for BED environments."""

from __future__ import annotations

import math
from typing import Any, Callable, Protocol, Sequence

import numpy as np


LikelihoodMessageBuilder = Callable[[Any, Any], list[dict[str, str]]]


class LLMBinaryLikelihoodMixin(Protocol):
    """Protocol for environments using batched Yes/No (or similar) LLM likelihoods."""

    observation_labels: tuple[Any, Any]
    config: Any

    def build_likelihood_messages(self, hypothesis: Any, action: Any) -> list[dict[str, str]]: ...

    def get_questioner(self) -> Any: ...


def log_likelihood_many_llm_binary(
    mixin: LLMBinaryLikelihoodMixin,
    hypotheses: Sequence[Any],
    action: Any,
    observation: Any,
    *,
    non_informative_observations: frozenset[Any] = frozenset(),
) -> np.ndarray:
    """Batched log p(observation | action, hypothesis) via ``chat_probabilities_messages_batched``."""
    if not hypotheses:
        return np.empty(0, dtype=float)
    positive_label, negative_label = mixin.observation_labels
    if observation in non_informative_observations or observation not in {positive_label, negative_label}:
        return np.zeros(len(hypotheses), dtype=float)

    questioner = mixin.get_questioner()
    labels = [positive_label, negative_label]
    conversations = [
        mixin.build_likelihood_messages(hypothesis, action) for hypothesis in hypotheses
    ]
    probabilities = questioner.chat_probabilities_messages_batched(
        conversations,
        labels,
        temperature=mixin.config.answer_temperature,
        block_size=mixin.config.batched_block_size,
    )
    log_likelihoods = np.empty(len(hypotheses), dtype=float)
    for idx, row in enumerate(probabilities):
        probability = max(row[observation], 1e-300)
        log_likelihoods[idx] = math.log(probability)
    return log_likelihoods
