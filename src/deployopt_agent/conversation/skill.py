"""Local tuning skill loader."""

from pathlib import Path


def load_tuning_skill() -> str:
    path = Path(__file__).parents[1] / "skills" / "tuning" / "SKILL.md"
    return path.read_text(encoding="utf-8").strip()
