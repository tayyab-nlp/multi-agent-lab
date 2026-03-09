"""Agent object builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clean(value: Any, fallback: str) -> str:
    """Normalize free-form UI values into safe plain text."""
    if value is None:
        return fallback
    if isinstance(value, list):
        text = "\n".join(str(item) for item in value if item is not None).strip()
    elif isinstance(value, dict):
        text = str(value).strip()
    else:
        text = str(value).strip()
    return text if text else fallback


@dataclass
class AgentProfile:
    """Simple runtime representation of an agent."""

    agent_id: str
    name: str
    role: str
    instruction: str
    specialization: str = ""
    is_main: bool = False


def create_main_agent(name: str, role: str, instruction: str) -> AgentProfile:
    """Build main coordinator agent profile."""
    return AgentProfile(
        agent_id="main",
        name=_clean(name, "Coordinator"),
        role=_clean(role, "Task planner"),
        instruction=_clean(
            instruction,
            "Break tasks into subtasks and delegate them to the best sub-agent.",
        ),
        is_main=True,
    )


def create_sub_agent(index: int, name: str, specialization: str, instruction: str) -> AgentProfile:
    """Build one sub-agent profile."""
    slot = f"sub{index}"
    default_name = f"Sub-agent {index}"
    return AgentProfile(
        agent_id=slot,
        name=_clean(name, default_name),
        role="Subtask executor",
        specialization=_clean(specialization, "General support"),
        instruction=_clean(
            instruction,
            "Complete assigned subtasks clearly and return concise output.",
        ),
        is_main=False,
    )
