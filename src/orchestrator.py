"""Workflow orchestration logic."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.agent_builder import AgentProfile
from src.config import MAX_SUBTASKS
from src.gemini_client import GeminiClient
from src.tool_builder import ToolConfig
from src.tools import TOOL_REGISTRY, ToolResult


@dataclass
class WorkflowResult:
    """Final payload for UI rendering."""

    architecture_summary: str
    agent_plan: str
    execution_trace: str
    sources_used: str
    final_answer: str


def _agent_lookup(main_agent: AgentProfile, sub_agents: list[AgentProfile]) -> dict[str, AgentProfile]:
    lookup = {main_agent.agent_id: main_agent}
    for agent in sub_agents:
        lookup[agent.agent_id] = agent
    return lookup


def _resolve_assignee(raw_name: str, main_agent: AgentProfile, sub_agents: list[AgentProfile]) -> AgentProfile:
    name = (raw_name or "").strip().lower()
    for agent in sub_agents:
        if agent.name.lower() == name or agent.agent_id.lower() == name:
            return agent
    if sub_agents:
        for agent in sub_agents:
            if agent.name.lower() in name or name in agent.name.lower():
                return agent
        return sub_agents[0]
    return main_agent


def _fallback_plan(main_agent: AgentProfile, sub_agents: list[AgentProfile], task: str) -> dict[str, Any]:
    if not sub_agents:
        return {
            "plan_overview": "Handle the task directly with the main agent.",
            "subtasks": [
                {
                    "title": f"Analyze and answer: {task}",
                    "assigned_agent": main_agent.name,
                    "reason": "No sub-agents configured.",
                }
            ],
        }

    subtasks = []
    titles = [
        f"Research key points for: {task}",
        f"Draft response for: {task}",
        f"Refine and quality-check response for: {task}",
    ]
    for idx, agent in enumerate(sub_agents):
        if idx >= len(titles):
            break
        subtasks.append(
            {
                "title": titles[idx],
                "assigned_agent": agent.name,
                "reason": f"Matched specialization: {agent.specialization}.",
            }
        )
    return {
        "plan_overview": "Split work across configured specialists and synthesize final output.",
        "subtasks": subtasks,
    }


def _format_architecture(main_agent: AgentProfile, sub_agents: list[AgentProfile], tools: list[ToolConfig]) -> str:
    lines = [
        "### Agent Architecture",
        f"- **Main Agent:** {main_agent.name} ({main_agent.role})",
    ]
    if sub_agents:
        lines.append("- **Sub-Agents:**")
        for agent in sub_agents:
            lines.append(f"  - {agent.name} | {agent.specialization}")
    else:
        lines.append("- **Sub-Agents:** None configured")

    if tools:
        lines.append("- **Enabled Tools:**")
        for tool in tools:
            assigned = ", ".join(tool.assigned_agent_ids)
            lines.append(f"  - {tool.name} (assigned to: {assigned})")
    else:
        lines.append("- **Enabled Tools:** None")
    return "\n".join(lines)


def _format_plan(plan: dict[str, Any]) -> str:
    lines = ["### Agent Plan", plan.get("plan_overview", "No overview provided."), ""]
    subtasks = plan.get("subtasks", [])
    if not subtasks:
        lines.append("- No subtasks produced.")
        return "\n".join(lines)

    for idx, item in enumerate(subtasks, start=1):
        lines.append(
            f"{idx}. **{item.get('title', 'Untitled')}**  \n"
            f"   Assigned: {item.get('assigned_agent', 'Unknown')}  \n"
            f"   Reason: {item.get('reason', 'No reason provided.')}"
        )
    return "\n".join(lines)


def _format_trace(trace_steps: list[str]) -> str:
    lines = ["### Execution Trace"]
    if not trace_steps:
        lines.append("- No trace available.")
    else:
        for idx, step in enumerate(trace_steps, start=1):
            lines.append(f"{idx}. {step}")
    return "\n".join(lines)


def _extract_expression(text: str) -> str:
    match = re.search(r"[-+/*()0-9.\s]{3,}", text or "")
    return match.group(0).strip() if match else ""


def _run_tool(tool: ToolConfig, subtask_title: str, full_task: str) -> ToolResult:
    fn = TOOL_REGISTRY.get(tool.tool_id)
    if not fn:
        return ToolResult(tool.name, "Tool implementation not found.", [])

    if tool.tool_id == "calculator":
        expression = _extract_expression(subtask_title) or _extract_expression(full_task)
        return fn(expression)  # type: ignore[misc]
    if tool.tool_id == "text_summarizer":
        return fn(full_task)  # type: ignore[misc]
    return fn(subtask_title or full_task)  # type: ignore[misc]


def run_workflow(
    api_key: str,
    model_id: str,
    main_agent: AgentProfile,
    sub_agents: list[AgentProfile],
    tools: list[ToolConfig],
    task: str,
) -> WorkflowResult:
    """Execute complete multi-agent workflow."""
    client = GeminiClient(api_key=api_key, model_id=model_id)
    trace: list[str] = []
    collected_sources: list[str] = []
    subtask_outputs: list[dict[str, str]] = []

    architecture = _format_architecture(main_agent, sub_agents, tools)
    trace.append(f"{main_agent.name} received task.")

    planning_system = (
        f"You are {main_agent.name}, role: {main_agent.role}. "
        f"Instruction: {main_agent.instruction}."
    )
    available_agents = [
        {"name": main_agent.name, "slot": main_agent.agent_id, "specialization": main_agent.role}
    ] + [
        {"name": agent.name, "slot": agent.agent_id, "specialization": agent.specialization}
        for agent in sub_agents
    ]

    planning_user = (
        f"Task:\n{task}\n\n"
        "Available agents:\n"
        f"{json.dumps(available_agents, indent=2)}\n\n"
        "Return JSON only with this schema:\n"
        "{"
        '"plan_overview":"...",'
        '"subtasks":[{"title":"...","assigned_agent":"...","reason":"..."}]'
        "}\n"
        f"Limit to {MAX_SUBTASKS} subtasks."
    )

    try:
        plan = client.generate_json(planning_system, planning_user)
    except Exception:  # pylint: disable=broad-except
        plan = _fallback_plan(main_agent, sub_agents, task)
        trace.append("Planner JSON parse failed; fallback planning used.")
    else:
        trace.append("Main agent created plan and subtasks.")

    subtasks = plan.get("subtasks", [])[:MAX_SUBTASKS]
    if not subtasks:
        plan = _fallback_plan(main_agent, sub_agents, task)
        subtasks = plan["subtasks"]
        trace.append("No subtasks returned; fallback subtask created.")

    for item in subtasks:
        title = item.get("title", "Untitled subtask")
        assignee = _resolve_assignee(item.get("assigned_agent", ""), main_agent, sub_agents)
        trace.append(f"Delegated '{title}' to {assignee.name}.")

        assigned_tools = [tool for tool in tools if assignee.agent_id in tool.assigned_agent_ids]
        tool_context_parts: list[str] = []
        if assigned_tools:
            for tool in assigned_tools:
                tool_result = _run_tool(tool, title, task)
                trace.append(f"{assignee.name} used tool: {tool.name}.")
                tool_context_parts.append(f"[{tool.name}]\n{tool_result.output}")
                for src in tool_result.sources:
                    if src not in collected_sources:
                        collected_sources.append(src)

        tool_context = "\n\n".join(tool_context_parts) if tool_context_parts else "No tool outputs."
        agent_system = (
            f"You are {assignee.name}. "
            f"Specialization: {assignee.specialization or assignee.role}. "
            f"Instruction: {assignee.instruction}"
        )
        agent_user = (
            f"Original task:\n{task}\n\n"
            f"Assigned subtask:\n{title}\n\n"
            f"Tool context:\n{tool_context}\n\n"
            "Return concise markdown with key findings only."
        )

        result_text = client.generate_text(agent_system, agent_user)
        subtask_outputs.append({"agent": assignee.name, "subtask": title, "result": result_text})
        trace.append(f"{assignee.name} completed subtask.")

    synthesis_system = (
        f"You are {main_agent.name}. Role: {main_agent.role}. "
        "Synthesize final answer from sub-agent outputs."
    )
    synthesis_user = (
        f"Original task:\n{task}\n\n"
        f"Plan:\n{json.dumps(plan, indent=2)}\n\n"
        f"Sub-agent outputs:\n{json.dumps(subtask_outputs, indent=2)}\n\n"
        "Return a clean final answer in markdown with:\n"
        "- Short heading\n"
        "- Core answer\n"
        "- Actionable bullets (if useful)\n"
        "- Keep it readable and concise."
    )
    final_answer = client.generate_text(synthesis_system, synthesis_user)
    trace.append("Main agent generated final synthesis.")

    if collected_sources:
        source_lines = ["### Sources Used"] + [f"- {src}" for src in collected_sources]
    else:
        source_lines = ["### Sources Used", "- No external sources were used."]

    return WorkflowResult(
        architecture_summary=architecture,
        agent_plan=_format_plan(plan),
        execution_trace=_format_trace(trace),
        sources_used="\n".join(source_lines),
        final_answer=final_answer,
    )
