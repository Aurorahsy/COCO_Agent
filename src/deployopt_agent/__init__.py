"""coco_agent v4 implementation.

The package is intentionally separate from the legacy ``coco_agent`` package.
"""

from .domain.contracts import GoalSpec, TaskStatus

__all__ = ["GoalSpec", "TaskStatus"]
