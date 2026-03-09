"""Tool configuration builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import AVAILABLE_TOOLS


def _clean(value: Any, fallback: str) -> str:
    """Normalize UI values into plain text safely."""
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
class ToolConfig:
    """User-selected tool configuration."""

    tool_id: str
    name: str
    assigned_agent_ids: list[str]


def create_tools(tool_rows: list[dict], valid_agent_ids: set[str]) -> list[ToolConfig]:
    """Build active tools from UI rows."""
    tools: list[ToolConfig] = []
    labels = {item["id"]: item["name"] for item in AVAILABLE_TOOLS}

    for row in tool_rows:
        if not row.get("enabled"):
            continue

        tool_id = row["tool_id"]
        assigned_raw = row.get("assigned_agent_ids") or []
        assigned = [slot for slot in assigned_raw if slot in valid_agent_ids]
        if not assigned:
            continue

        tools.append(
            ToolConfig(
                tool_id=tool_id,
                name=_clean(row.get("name"), labels.get(tool_id, tool_id)),
                assigned_agent_ids=assigned,
            )
        )
    return tools
