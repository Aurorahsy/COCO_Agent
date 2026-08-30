"""Runtime-independent contracts for COCO multi-agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class AgentRole(str, Enum):
    COORDINATOR = "coordinator"
    OBJECTIVE = "objective"
    CONTEXT = "context"
    BENCHMARK = "benchmark"
    DIAGNOSIS = "diagnosis"
    KNOWLEDGE = "knowledge"
    PLANNING = "planning"
    EXECUTION = "execution"
    REPORTING = "reporting"


class AgentStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkItem:
    work_id: str
    kind: str
    payload: dict[str, Any]
    experiment_id: str | None = None


@dataclass(frozen=True)
class Handoff:
    source: AgentRole
    target: AgentRole
    work: WorkItem
    reason: str


@dataclass(frozen=True)
class AgentOutcome:
    status: AgentStatus
    updates: dict[str, Any] = field(default_factory=dict)
    handoff: Handoff | None = None
    message: str | None = None
    approval_request: dict[str, Any] | None = None


@dataclass(frozen=True)
class OrchestrationEvent:
    sequence: int
    role: AgentRole
    work_id: str
    status: AgentStatus
    message: str | None = None


@dataclass(frozen=True)
class OrchestrationState:
    task_id: str
    status: AgentStatus
    active_role: AgentRole
    current_work: WorkItem
    revision: int = 0
    data: dict[str, Any] = field(default_factory=dict)
    history: tuple[OrchestrationEvent, ...] = ()
    pending_approval: dict[str, Any] | None = None

    def apply(self, role: AgentRole, outcome: AgentOutcome) -> "OrchestrationState":
        data = dict(self.data)
        data.update(outcome.updates)
        event = OrchestrationEvent(
            sequence=self.revision + 1,
            role=role,
            work_id=self.current_work.work_id,
            status=outcome.status,
            message=outcome.message,
        )
        next_role = outcome.handoff.target if outcome.handoff else role
        next_work = outcome.handoff.work if outcome.handoff else self.current_work
        return replace(
            self,
            status=outcome.status,
            active_role=next_role,
            current_work=next_work,
            revision=self.revision + 1,
            data=data,
            history=self.history + (event,),
            pending_approval=outcome.approval_request,
        )


class OrchestrationError(RuntimeError):
    pass


class InvalidHandoff(OrchestrationError):
    pass


class UnknownAgent(OrchestrationError):
    pass

