"""Coordinator for typed, bounded specialist-agent handoffs."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from .contracts import (
    AgentOutcome,
    AgentRole,
    AgentStatus,
    InvalidHandoff,
    OrchestrationState,
    UnknownAgent,
)


class SpecialistAgent(Protocol):
    role: AgentRole

    def handle(self, state: OrchestrationState) -> AgentOutcome: ...


class AgentRegistry:
    def __init__(self, agents: tuple[SpecialistAgent, ...] = ()) -> None:
        self._agents: dict[AgentRole, SpecialistAgent] = {}
        for agent in agents:
            self.register(agent)

    def register(self, agent: SpecialistAgent) -> None:
        if agent.role in self._agents:
            raise ValueError(f"agent role already registered: {agent.role.value}")
        self._agents[agent.role] = agent

    def get(self, role: AgentRole) -> SpecialistAgent:
        try:
            return self._agents[role]
        except KeyError as exc:
            raise UnknownAgent(f"agent role is not registered: {role.value}") from exc


class MultiAgentCoordinator:
    DEFAULT_HANDOFFS = {
        AgentRole.OBJECTIVE: {AgentRole.CONTEXT},
        AgentRole.CONTEXT: {AgentRole.BENCHMARK},
        AgentRole.BENCHMARK: {AgentRole.DIAGNOSIS, AgentRole.REPORTING},
        AgentRole.DIAGNOSIS: {AgentRole.KNOWLEDGE, AgentRole.BENCHMARK},
        AgentRole.KNOWLEDGE: {AgentRole.PLANNING},
        AgentRole.PLANNING: {AgentRole.EXECUTION, AgentRole.REPORTING},
        AgentRole.EXECUTION: {AgentRole.BENCHMARK, AgentRole.REPORTING},
        AgentRole.REPORTING: set(),
    }

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        allowed_handoffs: dict[AgentRole, set[AgentRole]] | None = None,
        max_steps: int = 32,
    ) -> None:
        self._registry = registry
        self._allowed = allowed_handoffs or self.DEFAULT_HANDOFFS
        self._max_steps = max_steps

    def run(self, state: OrchestrationState) -> OrchestrationState:
        current = replace(state, status=AgentStatus.RUNNING)
        steps = 0
        while current.status == AgentStatus.RUNNING:
            if steps >= self._max_steps:
                return replace(current, status=AgentStatus.FAILED)
            role = current.active_role
            outcome = self._registry.get(role).handle(current)
            self._validate_handoff(role, outcome)
            current = current.apply(role, outcome)
            steps += 1
        return current

    def resume_approval(
        self, state: OrchestrationState, *, approved: bool
    ) -> OrchestrationState:
        if state.status != AgentStatus.WAITING_APPROVAL or state.pending_approval is None:
            raise ValueError("orchestration is not waiting for approval")
        data = dict(state.data)
        data["approval"] = {
            "request": dict(state.pending_approval),
            "approved": approved,
        }
        resumed = replace(
            state,
            status=AgentStatus.RUNNING if approved else AgentStatus.FAILED,
            data=data,
            pending_approval=None,
        )
        return self.run(resumed) if approved else resumed

    def _validate_handoff(self, source: AgentRole, outcome: AgentOutcome) -> None:
        if outcome.handoff is None:
            return
        if outcome.handoff.source != source:
            raise InvalidHandoff("handoff source does not match active agent")
        if outcome.handoff.target not in self._allowed.get(source, set()):
            raise InvalidHandoff(
                f"handoff {source.value} -> {outcome.handoff.target.value} is not allowed"
            )

