"""Public surface for location-finding prompts.

The actual prompt-building functions remain defined in :mod:`location_finding`
for now because they are tightly coupled with the formatters, types, and
algorithm code in that module.  This file gives them a stable public name
under ``environments.location_finding.prompts``, which is where new code and
the future :class:`core.Environment` implementation will import them from.

When :mod:`location_finding` is decomposed into submodules (Phase E of the
refactor), the bodies of these functions will move here and the imports will
flip direction — ``location_finding.py`` will import from this module instead
of the other way around.  External callers of this module will not need to
change.
"""

from __future__ import annotations

# Re-export everything users of this module need.  We rebind the leading
# underscore on each name to advertise these as the supported public surface.
from environments.location_finding.runner import (
    _belief_generation_messages as belief_generation_messages,
    _belief_output_contract as belief_output_contract,
    _belief_system_prompt as belief_system_prompt,
    _candidate_generation_messages as candidate_generation_messages,
    _location_posterior_distribution_messages as location_posterior_distribution_messages,
    _naive_location_messages as naive_location_messages,
    _naive_source_estimate_messages as naive_source_estimate_messages,
    _naive_source_estimate_repair_messages as naive_source_estimate_repair_messages,
    _strategy_crossover_messages as strategy_crossover_messages,
    _strategy_diverse_messages as strategy_diverse_messages,
    _strategy_location_messages as strategy_location_messages,
    _strategy_mutation_messages as strategy_mutation_messages,
    _strategy_root_crossover_messages as strategy_root_crossover_messages,
    _strategy_root_diverse_messages as strategy_root_diverse_messages,
    _strategy_root_mutation_messages as strategy_root_mutation_messages,
    _strategy_root_system_preamble as strategy_root_system_preamble,
    _strategy_system_preamble as strategy_system_preamble,
)


__all__ = [
    "belief_generation_messages",
    "belief_output_contract",
    "belief_system_prompt",
    "candidate_generation_messages",
    "location_posterior_distribution_messages",
    "naive_location_messages",
    "naive_source_estimate_messages",
    "naive_source_estimate_repair_messages",
    "strategy_crossover_messages",
    "strategy_diverse_messages",
    "strategy_location_messages",
    "strategy_mutation_messages",
    "strategy_root_crossover_messages",
    "strategy_root_diverse_messages",
    "strategy_root_mutation_messages",
    "strategy_root_system_preamble",
    "strategy_system_preamble",
]
