"""Factored M-open support transitions for categorical ChemBench planning."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .mechanics import (
    MAX_PROPOSALS,
    ModelBank,
    PolicyLadderPlanner,
    ProposalCache,
    SpeculativeState,
    _logsumexp,
    proposal_key,
)


@dataclass(frozen=True)
class RegistrySignature:
    core: str
    modifiers: tuple[str, ...]

    @property
    def tags(self) -> frozenset[str]:
        return frozenset((f"core:{self.core}", *(f"modifier:{item}" for item in self.modifiers)))


def registry_signature(model_name: str) -> RegistrySignature:
    """Map every frozen ChemBench registry name to a canonical mechanism signature."""

    stem = re.sub(r"^c\d+_", "", str(model_name))
    if stem in {
        "michaelis_menten",
        "competitive_inhibition",
        "product_inhibition",
        "arrhenius_temperature",
        "uncompetitive_inhibition",
        "noncompetitive_inhibition",
    } or stem.startswith("mm_"):
        core = "michaelis_menten"
    elif stem == "pingpong_bisubstrate" or stem.startswith("pingpong_"):
        core = "pingpong"
    elif stem == "substrate_inhibition" or stem.startswith("sinh_"):
        core = "substrate_inhibition"
    elif stem == "hill_cooperativity" or stem.startswith("hill_"):
        core = "hill"
    elif stem.startswith("ordered_bi_bi"):
        core = "ordered_bi_bi"
    elif stem.startswith("allosteric_act"):
        core = "allosteric_activation"
    elif stem.startswith("anticoop"):
        core = "anticooperative_hill"
    elif stem.startswith("fractal"):
        core = "fractal_kinetics"
    elif stem.startswith("mixed_inh") or stem == "mixed_inhibition":
        core = "mixed_inhibition"
    elif stem.startswith("coop_inh") or stem == "coop_inhibition":
        core = "cooperative_inhibition"
    elif stem.startswith("monotonic_ph"):
        core = "monotonic_ph"
    elif stem.startswith("metal_act") or stem == "metal_activation":
        core = "metal_activation"
    elif stem.startswith("product_act") or stem == "product_activation":
        core = "product_activation"
    elif stem.startswith("two_sub_inh") or stem == "two_substrate_inhibition":
        core = "two_substrate_inhibition"
    elif stem.startswith("reversible_mm"):
        core = "reversible_michaelis_menten"
    else:
        raise ValueError(f"unknown ChemBench registry mechanism: {model_name}")

    modifiers: set[str] = set()
    if "arrhenius" in stem:
        modifiers.add("arrhenius")
    if stem.endswith("_ph") or "_arr_ph" in stem:
        modifiers.add("ph_bell_curve")
    if "uncompetitive" in stem:
        modifiers.add("uncompetitive_inhibition")
    elif "noncompetitive" in stem or "_noncomp" in stem:
        modifiers.add("noncompetitive_inhibition")
    elif "competitive" in stem:
        modifiers.add("competitive_inhibition")
    if stem == "product_inhibition" or (
        core in {"michaelis_menten", "hill", "substrate_inhibition"}
        and "_product" in stem
    ):
        modifiers.add("product_inhibition")
    if "feedback" in stem:
        modifiers.add("product_feedback")
    return RegistrySignature(core=core, modifiers=tuple(sorted(modifiers)))


@dataclass(frozen=True)
class RegistryEdit:
    parent: int
    candidate: int
    operation: str
    added_tags: tuple[str, ...]
    removed_tags: tuple[str, ...]
    core_family: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "parent": self.parent,
            "candidate": self.candidate,
            "operation": self.operation,
            "added_tags": list(self.added_tags),
            "removed_tags": list(self.removed_tags),
            "core_family": self.core_family,
        }


class RegistryEditCompiler:
    """Compile a registry candidate into a validated typed patch from live support."""

    OPERATIONS = frozenset({"add_factor", "remove_factor", "replace_factor", "replace_core"})

    def __init__(self, bank: ModelBank) -> None:
        if len(set(bank.model_names)) != bank.num_models:
            raise ValueError("registry model names must be unique")
        self.bank = bank
        self.signatures = tuple(registry_signature(name) for name in bank.model_names)

    def core_family(self, model: int) -> str:
        return self.signatures[int(model)].core

    def compile(self, represented: Sequence[int], candidate: int) -> RegistryEdit:
        candidate = int(candidate)
        parents = tuple(dict.fromkeys(int(item) for item in represented))
        if not parents:
            raise ValueError("typed edit requires a represented parent")
        if candidate < 0 or candidate >= self.bank.num_models:
            raise ValueError("typed edit candidate is outside the registry")
        if candidate in parents:
            raise ValueError("typed edit candidate is already represented")
        if not np.isfinite(self.bank.likelihoods[candidate]).all() or not np.isfinite(
            self.bank.target_features[candidate]
        ).all():
            raise ValueError("typed edit candidate is not executable")

        candidate_signature = self.signatures[candidate]

        def distance(parent: int) -> tuple[int, int]:
            parent_signature = self.signatures[parent]
            core_cost = 2 if parent_signature.core != candidate_signature.core else 0
            modifier_cost = len(
                set(parent_signature.modifiers).symmetric_difference(candidate_signature.modifiers)
            )
            return core_cost + modifier_cost, parent

        parent = min(parents, key=distance)
        parent_tags = self.signatures[parent].tags
        candidate_tags = candidate_signature.tags
        added = tuple(sorted(candidate_tags - parent_tags))
        removed = tuple(sorted(parent_tags - candidate_tags))
        if not added and not removed:
            raise ValueError("typed edit does not change the parent signature")
        if self.signatures[parent].core != candidate_signature.core:
            operation = "replace_core"
        elif added and removed:
            operation = "replace_factor"
        elif added:
            operation = "add_factor"
        else:
            operation = "remove_factor"
        edit = RegistryEdit(
            parent=parent,
            candidate=candidate,
            operation=operation,
            added_tags=added,
            removed_tags=removed,
            core_family=candidate_signature.core,
        )
        self.validate(edit)
        return edit

    def validate(self, edit: RegistryEdit) -> None:
        if edit.operation not in self.OPERATIONS:
            raise ValueError("unknown typed edit operation")
        if edit.parent == edit.candidate:
            raise ValueError("typed edit cannot target its parent")
        parent_tags = set(self.signatures[edit.parent].tags)
        if not set(edit.removed_tags).issubset(parent_tags):
            raise ValueError("typed edit removes a tag absent from its parent")
        roundtrip = (parent_tags - set(edit.removed_tags)) | set(edit.added_tags)
        if roundtrip != set(self.signatures[edit.candidate].tags):
            raise ValueError("typed edit does not round-trip to its candidate")
        if edit.core_family != self.signatures[edit.candidate].core:
            raise ValueError("typed edit core family mismatch")


@dataclass(frozen=True)
class FactoredState:
    history: tuple[tuple[int, int], ...]
    discovered: tuple[int, ...]
    live: tuple[int, ...]
    reserve: tuple[int, ...]
    tried: tuple[int, ...]
    edit_history: tuple[RegistryEdit, ...]
    represented_mass: tuple[float, ...]
    represented_weight: tuple[float, ...]
    outside_mass: float
    last_represented_probability: float | None
    last_surprise: float | None
    expansion_triggered: bool
    phase: str
    last_proposed: tuple[int, ...]
    last_pruned: tuple[int, ...]
    last_edits: tuple[RegistryEdit, ...]

    @property
    def represented_models(self) -> tuple[int, ...]:
        return self.live + self.reserve

    @property
    def represented_weights(self) -> np.ndarray:
        values = np.asarray(self.represented_weight, dtype=float)
        if not np.isfinite(values).all() or not math.isclose(
            float(values.sum()), 1.0, abs_tol=1e-12, rel_tol=0.0
        ):
            raise FloatingPointError("factored represented weights are not normalized")
        return values

    def public_key(self) -> dict[str, Any]:
        return {
            "history": [list(item) for item in self.history],
            "discovered": list(self.discovered),
            "live": list(self.live),
            "reserve": list(self.reserve),
            "tried": list(self.tried),
            "edit_history": [item.public_dict() for item in self.edit_history],
            "last_represented_probability": self.last_represented_probability,
            "last_surprise": self.last_surprise,
            "expansion_triggered": self.expansion_triggered,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class ExpansionDiagnostic:
    represented_probability: float
    surprise: float
    threshold: float
    triggered: bool


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("weighted quantile must be strictly between zero and one")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    cumulative /= cumulative[-1]
    return float(ordered_values[min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)])


class FactoredModelBank(ModelBank):
    """Categorical model bank with calibrated expansion and bounded support."""

    def __init__(
        self,
        likelihoods: np.ndarray,
        target_features: np.ndarray,
        model_names: Sequence[str],
        action_names: Sequence[str],
        action_groups: Sequence[str | None],
        initial_support: Sequence[int],
        *,
        outside_prior: float = 0.35,
        evidence_slots: int = 8,
        diversity_slots: int = 4,
        surprise_quantile: float = 0.90,
    ) -> None:
        if evidence_slots <= 0 or diversity_slots < 0:
            raise ValueError("factored support slots are invalid")
        super().__init__(
            likelihoods,
            target_features,
            model_names,
            action_names,
            action_groups,
            initial_support,
            outside_prior=outside_prior,
            live_cap=evidence_slots,
            reserve_cap=diversity_slots,
        )
        self.evidence_slots = int(evidence_slots)
        self.diversity_slots = int(diversity_slots)
        self.pool_cap = self.evidence_slots + self.diversity_slots
        self.surprise_quantile = float(surprise_quantile)
        self.compiler = RegistryEditCompiler(self)
        self.surprise_threshold, self.calibration_false_trigger_mass = self._calibrate_trigger()

    def _calibrate_trigger(self) -> tuple[float, float]:
        initial = np.asarray(self.initial_support)
        represented_predictive = np.mean(self.likelihoods[initial], axis=0)
        values: list[float] = []
        weights: list[float] = []
        normalizer = len(initial) * self.num_actions
        for generator in initial:
            for action in range(self.num_actions):
                for outcome in range(3):
                    probability = float(self.likelihoods[generator, action, outcome])
                    if probability <= 0:
                        continue
                    values.append(-math.log(max(float(represented_predictive[action, outcome]), 1e-300)))
                    weights.append(probability / normalizer)
        value_array = np.asarray(values, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        threshold = _weighted_quantile(value_array, weight_array, self.surprise_quantile)
        false_mass = float(weight_array[value_array > threshold].sum())
        return threshold, false_mass

    def state(
        self,
        history: Sequence[tuple[int, int]],
        discovered: Sequence[int],
        *,
        tried: Sequence[int] | None = None,
        edit_history: Sequence[RegistryEdit] = (),
        last_represented_probability: float | None = None,
        last_surprise: float | None = None,
        expansion_triggered: bool = False,
        last_proposed: Sequence[int] = (),
        last_edits: Sequence[RegistryEdit] = (),
    ) -> FactoredState:
        history_tuple = tuple((int(action), int(outcome)) for action, outcome in history)
        support = tuple(sorted(set(int(item) for item in discovered)))
        if not support or any(item < 0 or item >= self.num_models for item in support):
            raise ValueError("factored discovered support is invalid")
        tried_tuple = tuple(sorted(set(support if tried is None else (int(item) for item in tried))))
        if any(item < 0 or item >= self.num_models for item in tried_tuple):
            raise ValueError("factored tried support is invalid")
        scores_by_model = {
            model: math.log(1.0 - self.outside_prior)
            - math.log(len(support))
            + self.log_likelihood(model, history_tuple)
            for model in support
        }
        ranked = sorted(support, key=lambda model: (-scores_by_model[model], model))
        live = tuple(ranked[: self.evidence_slots])
        remaining = [model for model in ranked if model not in live]
        live_families = {self.compiler.core_family(model) for model in live}
        reserve: list[int] = []
        reserved_families: set[str] = set()
        for model in remaining:
            family = self.compiler.core_family(model)
            if family in live_families or family in reserved_families:
                continue
            reserve.append(model)
            reserved_families.add(family)
            if len(reserve) == self.diversity_slots:
                break
        if len(reserve) < self.diversity_slots:
            reserve.extend(
                model
                for model in remaining
                if model not in reserve
            )
            reserve = reserve[: self.diversity_slots]
        reserve_tuple = tuple(reserve)
        represented = live + reserve_tuple
        pruned = tuple(sorted(set(support) - set(represented)))
        represented_scores = np.asarray([scores_by_model[model] for model in represented], dtype=float)
        known_normalizer = _logsumexp(represented_scores)
        represented_weight_values = np.exp(represented_scores - known_normalizer)
        outside_score = math.log(self.outside_prior) - len(history_tuple) * math.log(3.0)
        difference = outside_score - known_normalizer
        if difference >= 0:
            ratio = math.exp(-difference) if difference < 746.0 else 0.0
            represented_total_mass = ratio / (1.0 + ratio)
        else:
            ratio = math.exp(difference) if difference > -746.0 else 0.0
            represented_total_mass = 1.0 / (1.0 + ratio)
        represented_total_mass = max(represented_total_mass, np.finfo(float).tiny)
        outside_mass = 1.0 - represented_total_mass
        represented_mass = tuple(
            float(represented_total_mass * value) for value in represented_weight_values
        )
        if not math.isclose(sum(represented_mass) + outside_mass, 1.0, abs_tol=1e-12):
            raise FloatingPointError("factored dynamic belief did not normalize")
        phase = "explore" if expansion_triggered else "refine"
        return FactoredState(
            history=history_tuple,
            discovered=tuple(sorted(represented)),
            live=live,
            reserve=reserve_tuple,
            tried=tried_tuple,
            edit_history=tuple(edit_history),
            represented_mass=represented_mass,
            represented_weight=tuple(float(value) for value in represented_weight_values),
            outside_mass=outside_mass,
            last_represented_probability=(
                None if last_represented_probability is None else float(last_represented_probability)
            ),
            last_surprise=None if last_surprise is None else float(last_surprise),
            expansion_triggered=bool(expansion_triggered),
            phase=phase,
            last_proposed=tuple(int(item) for item in last_proposed),
            last_pruned=pruned,
            last_edits=tuple(last_edits),
        )

    def initial_state(self) -> FactoredState:
        return self.state((), self.initial_support, tried=self.initial_support)

    def represented_predictive(self, state: FactoredState, action: int) -> np.ndarray:
        result = state.represented_weights @ self.likelihoods[
            np.asarray(state.represented_models), int(action), :
        ]
        total = float(result.sum())
        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError("represented predictive has no mass")
        return result / total

    def expansion_diagnostic(
        self, state: FactoredState, action: int, outcome: int
    ) -> ExpansionDiagnostic:
        probability = float(self.represented_predictive(state, action)[outcome])
        surprise = -math.log(max(probability, 1e-300))
        return ExpansionDiagnostic(
            represented_probability=probability,
            surprise=surprise,
            threshold=self.surprise_threshold,
            triggered=surprise > self.surprise_threshold,
        )

    def transition(
        self,
        state: FactoredState,
        action: int,
        outcome: int,
        proposal: Sequence[int],
    ) -> FactoredState:
        if action < 0 or action >= self.num_actions or outcome not in (0, 1, 2):
            raise ValueError("factored transition action or outcome is invalid")
        diagnostic = self.expansion_diagnostic(state, action, outcome)
        proposed = tuple(int(item) for item in proposal)
        if proposed and not diagnostic.triggered:
            raise ValueError("non-triggering transition cannot consume proposals")
        support = list(state.discovered)
        tried = set(state.tried)
        accepted_edits: list[RegistryEdit] = []
        seen = set(state.discovered)
        for candidate in proposed:
            if candidate in seen or candidate in tried or candidate < 0 or candidate >= self.num_models:
                continue
            try:
                edit = self.compiler.compile(state.represented_models, candidate)
            except ValueError:
                continue
            self.compiler.validate(edit)
            tried.add(candidate)
            seen.add(candidate)
            support.append(candidate)
            accepted_edits.append(edit)
            if len(accepted_edits) == MAX_PROPOSALS:
                break
        history = state.history + ((int(action), int(outcome)),)
        return self.state(
            history,
            support,
            tried=tuple(tried),
            edit_history=state.edit_history + tuple(accepted_edits),
            last_represented_probability=diagnostic.represented_probability,
            last_surprise=diagnostic.surprise,
            expansion_triggered=diagnostic.triggered,
            last_proposed=proposed[:MAX_PROPOSALS],
            last_edits=accepted_edits,
        )

    def residual_report(self, state: FactoredState, *, remaining_budget: int) -> dict[str, Any]:
        rows = []
        weights = state.represented_weights
        for position, model in enumerate(state.represented_models):
            innovations: dict[str, list[float]] = {}
            negative_log_likelihood = 0.0
            for action, outcome in state.history:
                probability = float(self.likelihoods[model, action, outcome])
                negative_log_likelihood -= math.log(max(probability, 1e-300))
                group = self.action_groups[action]
                if group is None:
                    continue
                predicted_mean = float(
                    self.likelihoods[model, action, :] @ np.arange(3, dtype=float)
                )
                innovations.setdefault(group, []).append(float(outcome) - predicted_mean)
            rows.append(
                {
                    "model_id": int(model),
                    "model_name": self.model_names[model],
                    "core_family": self.compiler.core_family(model),
                    "evidence_weight": round(float(weights[position]), 12),
                    "categorical_nll": round(negative_log_likelihood, 12),
                    "signed_innovation_by_group": {
                        group: round(float(np.mean(values)), 12)
                        for group, values in sorted(innovations.items())
                    },
                }
            )
        latest = None
        if state.history:
            action, outcome = state.history[-1]
            latest = {
                "action": action,
                "action_name": self.action_names[action],
                "action_group": self.action_groups[action],
                "outcome": outcome,
                "represented_probability": state.last_represented_probability,
                "surprise": state.last_surprise,
                "expansion_triggered": state.expansion_triggered,
            }
        return {
            "models": rows,
            "latest": latest,
            "tried_models": list(state.tried),
            "represented_models": list(state.represented_models),
            "phase": state.phase,
            "remaining_budget": int(remaining_budget),
        }


class TypedRegistryOracleProposer:
    mode = "typed_registry_oracle"

    def __init__(self, bank: FactoredModelBank) -> None:
        self.bank = bank

    def propose(self, state: FactoredState, action: int, outcome: int, seed: int) -> tuple[int, ...]:
        del seed
        history = state.history + ((int(action), int(outcome)),)
        missing = [model for model in range(self.bank.num_models) if model not in state.tried]
        missing.sort(key=lambda model: (-self.bank.log_likelihood(model, history), model))
        valid: list[int] = []
        for candidate in missing:
            try:
                self.bank.compiler.compile(state.represented_models, candidate)
            except ValueError:
                continue
            valid.append(candidate)
            if len(valid) == MAX_PROPOSALS:
                break
        return tuple(valid)


class FactoredPolicyLadderPlanner(PolicyLadderPlanner):
    """Policy ladder whose proposer is called only after a calibrated trigger."""

    bank: FactoredModelBank

    def __init__(
        self,
        bank: FactoredModelBank,
        proposal_cache: ProposalCache,
        particle_indices: Sequence[int],
        *,
        seed: int,
    ) -> None:
        super().__init__(bank, proposal_cache, particle_indices, seed=seed)
        self.transition_audit: dict[str, dict[str, Any]] = {}

    def transition(self, state: SpeculativeState, action: int, outcome: int) -> SpeculativeState:
        weights = state.weights()
        posterior = weights * self.bank.likelihoods[
            np.asarray(self.particle_indices), action, outcome
        ]
        total = float(posterior.sum())
        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError("factored speculative branch has zero posterior mass")
        posterior /= total
        diagnostic = self.bank.expansion_diagnostic(state.inference, action, outcome)
        seed = self._seed(state, action, outcome)
        proposal = (
            self.proposal_cache.get(state.inference, action, outcome, seed)
            if diagnostic.triggered
            else ()
        )
        inference = self.bank.transition(state.inference, action, outcome, proposal)
        report = self.bank.residual_report(inference, remaining_budget=0)
        report_sha = hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        audit_key = proposal_key("factored-transition-audit", state.inference, action, outcome, seed)
        families = {
            self.bank.compiler.core_family(model) for model in inference.represented_models
        }
        record = {
            "action": int(action),
            "outcome": int(outcome),
            "represented_probability": diagnostic.represented_probability,
            "surprise": diagnostic.surprise,
            "threshold": diagnostic.threshold,
            "triggered": diagnostic.triggered,
            "proposal_requested": diagnostic.triggered,
            "proposal": [int(item) for item in proposal],
            "proposal_novel_and_unique": len(proposal) == len(set(proposal))
            and all(int(item) not in state.inference.tried for item in proposal),
            "parent_tried_size": len(state.inference.tried),
            "accepted_edits": [item.public_dict() for item in inference.last_edits],
            "pruned": list(inference.last_pruned),
            "support_size": len(inference.represented_models),
            "core_family_count": len(families),
            "tried_size": len(inference.tried),
            "phase": inference.phase,
            "residual_report_sha256": report_sha,
        }
        previous = self.transition_audit.get(audit_key)
        if previous is not None and previous != record:
            raise AssertionError("factored transition audit is not deterministic")
        self.transition_audit[audit_key] = record
        return SpeculativeState(
            inference=inference,
            particle_weight=tuple(float(value) for value in posterior),
        )
