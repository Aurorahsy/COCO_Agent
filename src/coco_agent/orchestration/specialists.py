"""Composable specialist implementation used by deterministic and LLM agents."""

from __future__ import annotations

from typing import Callable

from .contracts import AgentOutcome, AgentRole, OrchestrationState


class FunctionalSpecialist:
    def __init__(
        self,
        role: AgentRole,
        handler: Callable[[OrchestrationState], AgentOutcome],
    ) -> None:
        self.role = role
        self._handler = handler

    def handle(self, state: OrchestrationState) -> AgentOutcome:
        return self._handler(state)

