from coco_agent.orchestration import (
    AgentOutcome,
    AgentRegistry,
    AgentRole,
    AgentStatus,
    FunctionalSpecialist,
    Handoff,
    MultiAgentCoordinator,
    OrchestrationState,
    WorkItem,
)
from coco_agent.orchestration.contracts import InvalidHandoff


def handoff(source, target, state, *, updates=None, kind=None):
    return AgentOutcome(
        status=AgentStatus.RUNNING,
        updates=updates or {},
        handoff=Handoff(
            source=source,
            target=target,
            work=WorkItem(
                work_id=f"{state.task_id}:{target.value}",
                kind=kind or target.value,
                payload={},
                experiment_id=state.current_work.experiment_id,
            ),
            reason=f"{source.value} completed",
        ),
    )


def initial_state():
    return OrchestrationState(
        task_id="task-1",
        status=AgentStatus.READY,
        active_role=AgentRole.OBJECTIVE,
        current_work=WorkItem("task-1:objective", "objective", {}),
    )


def test_coordinator_routes_typed_specialists_to_report():
    sequence = [
        (AgentRole.OBJECTIVE, AgentRole.CONTEXT, "objective"),
        (AgentRole.CONTEXT, AgentRole.BENCHMARK, "context"),
        (AgentRole.BENCHMARK, AgentRole.DIAGNOSIS, "baseline"),
        (AgentRole.DIAGNOSIS, AgentRole.KNOWLEDGE, "hypothesis"),
        (AgentRole.KNOWLEDGE, AgentRole.PLANNING, "features"),
        (AgentRole.PLANNING, AgentRole.EXECUTION, "candidate"),
        (AgentRole.EXECUTION, AgentRole.BENCHMARK, "change"),
    ]
    agents = []
    for source, target, key in sequence:
        agents.append(
            FunctionalSpecialist(
                source,
                lambda state, s=source, t=target, k=key: handoff(
                    s, t, state, updates={k: True}
                ),
            )
        )

    benchmark_calls = {"count": 0}

    def benchmark(state):
        benchmark_calls["count"] += 1
        if benchmark_calls["count"] == 1:
            return handoff(
                AgentRole.BENCHMARK,
                AgentRole.DIAGNOSIS,
                state,
                updates={"baseline": True},
            )
        return handoff(
            AgentRole.BENCHMARK,
            AgentRole.REPORTING,
            state,
            updates={"candidate_run": True},
        )

    agents = [agent for agent in agents if agent.role != AgentRole.BENCHMARK]
    agents.extend(
        [
            FunctionalSpecialist(AgentRole.BENCHMARK, benchmark),
            FunctionalSpecialist(
                AgentRole.REPORTING,
                lambda state: AgentOutcome(
                    AgentStatus.COMPLETED,
                    updates={"report": "complete"},
                ),
            ),
        ]
    )
    result = MultiAgentCoordinator(AgentRegistry(tuple(agents))).run(initial_state())
    assert result.status is AgentStatus.COMPLETED
    assert result.data["report"] == "complete"
    assert benchmark_calls["count"] == 2
    assert [event.role for event in result.history][0] is AgentRole.OBJECTIVE


def test_execution_agent_can_pause_and_resume_for_approval():
    calls = {"count": 0}

    def execution(state):
        calls["count"] += 1
        if "approval" not in state.data:
            return AgentOutcome(
                AgentStatus.WAITING_APPROVAL,
                approval_request={"candidate_id": "candidate-1", "risk": "high"},
            )
        return handoff(AgentRole.EXECUTION, AgentRole.REPORTING, state)

    registry = AgentRegistry(
        (
            FunctionalSpecialist(AgentRole.EXECUTION, execution),
            FunctionalSpecialist(
                AgentRole.REPORTING,
                lambda state: AgentOutcome(AgentStatus.COMPLETED, {"report": True}),
            ),
        )
    )
    coordinator = MultiAgentCoordinator(registry)
    state = OrchestrationState(
        task_id="task-approval",
        status=AgentStatus.READY,
        active_role=AgentRole.EXECUTION,
        current_work=WorkItem("work-1", "apply_candidate", {}),
    )
    waiting = coordinator.run(state)
    assert waiting.status is AgentStatus.WAITING_APPROVAL
    completed = coordinator.resume_approval(waiting, approved=True)
    assert completed.status is AgentStatus.COMPLETED
    assert calls["count"] == 2


def test_coordinator_rejects_cross_boundary_handoff():
    registry = AgentRegistry(
        (
            FunctionalSpecialist(
                AgentRole.OBJECTIVE,
                lambda state: handoff(AgentRole.OBJECTIVE, AgentRole.EXECUTION, state),
            ),
        )
    )
    try:
        MultiAgentCoordinator(registry).run(initial_state())
    except InvalidHandoff as error:
        assert "objective -> execution" in str(error)
    else:
        raise AssertionError("invalid handoff must be rejected")

