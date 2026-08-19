"""Public compositional edits for staged ChemBench structural discovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .mechanics import (
    BankedProposer,
    DynamicState,
    ModelBank,
    PolicyLadderPlanner,
    ProposalCache,
    SpeculativeState,
    _logsumexp,
    proposal_key,
)


@dataclass(frozen=True, order=True)
class CompoundSignature:
    substrate: str
    inhibitor: str = "none"
    temperature: str = "none"
    ph_dependence: str = "none"

    def __post_init__(self) -> None:
        if self.substrate not in {"mm", "hill", "substrate_inh", "pingpong"}:
            raise ValueError("unknown substrate core")
        if self.inhibitor not in {
            "none",
            "competitive",
            "uncompetitive",
            "noncompetitive",
            "product",
        }:
            raise ValueError("unknown inhibitor mechanism")
        if self.temperature not in {"none", "arrhenius"}:
            raise ValueError("unknown temperature mechanism")
        if self.ph_dependence not in {"none", "bell_curve"}:
            raise ValueError("unknown pH mechanism")

    @property
    def components(self) -> tuple[str, str, str, str]:
        return (
            self.substrate,
            self.inhibitor,
            self.temperature,
            self.ph_dependence,
        )


@dataclass(frozen=True)
class AtomicCompoundEdit:
    parent: str
    candidate: str
    operation: str
    component: str
    before: str
    after: str

    def public_dict(self) -> dict[str, str]:
        return {
            "parent": self.parent,
            "candidate": self.candidate,
            "operation": self.operation,
            "component": self.component,
            "before": self.before,
            "after": self.after,
        }


class CompositionalEditCompiler:
    """Validate one scientific edit between source-supported compositions."""

    def __init__(self, signatures: Mapping[str, CompoundSignature]) -> None:
        self.signatures = dict(signatures)
        if not self.signatures or len(set(self.signatures.values())) != len(self.signatures):
            raise ValueError("compositional signatures must be nonempty and unique")

    def compile(self, parent: str, candidate: str) -> AtomicCompoundEdit:
        if parent not in self.signatures or candidate not in self.signatures:
            raise ValueError("edit references an unknown structure")
        if parent == candidate:
            raise ValueError("atomic edit must change the structure")
        before = self.signatures[parent]
        after = self.signatures[candidate]
        fields = ("substrate", "inhibitor", "temperature", "ph_dependence")
        changed = [field for field in fields if getattr(before, field) != getattr(after, field)]
        if len(changed) != 1:
            raise ValueError("atomic edit must change exactly one component")
        component = changed[0]
        old = getattr(before, component)
        new = getattr(after, component)
        if component == "substrate":
            operation = "replace_core"
        elif component == "inhibitor":
            if old != "none" and new != "none":
                raise ValueError("changing inhibitor type requires remove then add")
            operation = "add_inhibitor" if old == "none" else "remove_inhibitor"
        else:
            if old != "none" and new != "none":
                raise ValueError("modifier replacement is not atomic")
            operation = f"add_{component}" if old == "none" else f"remove_{component}"
        return AtomicCompoundEdit(
            parent=parent,
            candidate=candidate,
            operation=operation,
            component=component,
            before=old,
            after=new,
        )

    def parents(
        self, represented: Sequence[str], candidate: str
    ) -> tuple[AtomicCompoundEdit, ...]:
        edits = []
        for parent in sorted(set(represented)):
            try:
                edits.append(self.compile(parent, candidate))
            except ValueError:
                continue
        return tuple(edits)


class StructureParticleIndex:
    """Bind structures to equal-size inference particles and held-out truths."""

    def __init__(
        self,
        particle_structure: Sequence[str],
        *,
        inference_particles: Mapping[str, Sequence[int]],
        truth_particles: Sequence[int],
    ) -> None:
        self.particle_structure = tuple(str(item) for item in particle_structure)
        self.inference_particles = {
            str(structure): tuple(int(item) for item in particles)
            for structure, particles in inference_particles.items()
        }
        self.truth_particles = frozenset(int(item) for item in truth_particles)
        if not self.particle_structure or not self.inference_particles:
            raise ValueError("particle index is empty")
        particle_count = len(self.particle_structure)
        all_inference = [
            particle for particles in self.inference_particles.values() for particle in particles
        ]
        if len(set(all_inference)) != len(all_inference):
            raise ValueError("inference particles overlap")
        if self.truth_particles.intersection(all_inference):
            raise ValueError("truth and inference particles overlap")
        if any(item < 0 or item >= particle_count for item in (*all_inference, *self.truth_particles)):
            raise ValueError("particle index is out of range")
        if any(
            self.particle_structure[particle] != structure
            for structure, particles in self.inference_particles.items()
            for particle in particles
        ):
            raise ValueError("inference particle structure mismatch")
        sizes = {len(value) for value in self.inference_particles.values()}
        if sizes != {3}:
            raise ValueError("every structure must have exactly three inference particles")

    def represented_structures(self, state: DynamicState) -> tuple[str, ...]:
        return tuple(
            sorted({self.particle_structure[particle] for particle in state.discovered})
        )


class AtomicStructureOracleProposer:
    """Choose one executable neighboring structure by marginal history evidence."""

    mode = "atomic_compositional_oracle"

    def __init__(
        self,
        bank: ModelBank,
        compiler: CompositionalEditCompiler,
        particles: StructureParticleIndex,
    ) -> None:
        self.bank = bank
        self.compiler = compiler
        self.particles = particles

    def _score(self, structure: str, history: Sequence[tuple[int, int]]) -> float:
        particle_ids = self.particles.inference_particles[structure]
        values = np.asarray(
            [self.bank.log_likelihood(particle, history) for particle in particle_ids],
            dtype=float,
        )
        return _logsumexp(values) - math.log(len(values))

    def propose(
        self, state: DynamicState, action: int, outcome: int, seed: int
    ) -> tuple[int, ...]:
        del seed
        represented = self.particles.represented_structures(state)
        history = state.history + ((int(action), int(outcome)),)
        candidates = []
        for structure in self.particles.inference_particles:
            if structure in represented:
                continue
            parents = self.compiler.parents(represented, structure)
            if not parents:
                continue
            candidates.append(
                (
                    -self._score(structure, history),
                    self.compiler.signatures[structure].components,
                    structure,
                    parents[0],
                )
            )
        if not candidates:
            return ()
        _, _, structure, _ = min(candidates)
        proposal = self.particles.inference_particles[structure]
        if self.particles.truth_particles.intersection(proposal):
            raise AssertionError("oracle attempted to propose a held-out truth particle")
        return proposal


class AuditedCompositionalPolicyPlanner(PolicyLadderPlanner):
    """Policy ladder with independently checkable atomic proposal transitions."""

    def __init__(
        self,
        bank: ModelBank,
        proposal_cache: ProposalCache,
        particle_indices: Sequence[int],
        *,
        compiler: CompositionalEditCompiler,
        particles: StructureParticleIndex,
        seed: int,
    ) -> None:
        super().__init__(bank, proposal_cache, particle_indices, seed=seed)
        self.compiler = compiler
        self.particles = particles
        self.transition_audit: dict[str, dict[str, Any]] = {}

    def transition(self, state: SpeculativeState, action: int, outcome: int) -> SpeculativeState:
        weights = state.weights()
        posterior = weights * self.bank.likelihoods[
            np.asarray(self.particle_indices), action, outcome
        ]
        total = float(posterior.sum())
        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError("compositional branch has zero posterior mass")
        posterior /= total
        seed = self._seed(state, action, outcome)
        proposal = self.proposal_cache.get(state.inference, action, outcome, seed)
        represented_before = self.particles.represented_structures(state.inference)
        proposal_structures = tuple(
            sorted({self.particles.particle_structure[item] for item in proposal})
        )
        edit = None
        valid = not proposal or (
            len(proposal_structures) == 1
            and proposal
            == self.particles.inference_particles.get(proposal_structures[0], ())
            and not self.particles.truth_particles.intersection(proposal)
        )
        if valid and proposal:
            parents = self.compiler.parents(represented_before, proposal_structures[0])
            valid = bool(parents)
            edit = parents[0] if parents else None
        if not valid:
            raise ValueError("proposal is not one complete atomic structure expansion")
        inference = self.bank.transition(state.inference, action, outcome, proposal)
        key = proposal_key(
            "compositional-transition-audit", state.inference, action, outcome, seed
        )
        record = {
            "action": int(action),
            "outcome": int(outcome),
            "represented_before": list(represented_before),
            "proposal": [int(item) for item in proposal],
            "proposal_structures": list(proposal_structures),
            "edit": edit.public_dict() if edit is not None else None,
            "truth_particle_proposed": bool(
                self.particles.truth_particles.intersection(proposal)
            ),
            "represented_after": list(self.particles.represented_structures(inference)),
        }
        record_once = getattr(self.transition_audit, "record_once", None)
        if record_once is not None:
            record_once(key, record)
        else:
            previous = self.transition_audit.get(key)
            if previous is not None and previous != record:
                raise AssertionError("compositional transition audit is not deterministic")
            self.transition_audit[key] = record
        return SpeculativeState(
            inference=inference,
            particle_weight=tuple(float(item) for item in posterior),
        )


def replay_proposer(source_mode: str, records: Mapping[str, Sequence[int]]) -> BankedProposer:
    return BankedProposer(source_mode, records)
