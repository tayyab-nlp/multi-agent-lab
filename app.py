"""Multi-Agent Orchestrator Gradio application."""

from __future__ import annotations

import os
import queue
import threading
from typing import Any

import gradio as gr

from src.agent_builder import AgentProfile, create_main_agent, create_sub_agent
from src.config import (
    AGENT_SLOT_CHOICES,
    APP_DESCRIPTION,
    APP_TITLE,
    AVAILABLE_TOOLS,
    EXAMPLE_TASKS,
    MODEL_ID,
)
from src.orchestrator import WorkflowResult, run_workflow
from src.tool_builder import ToolConfig, create_tools

LEFT_LABEL_TO_SLOT = {label: slot for label, slot in AGENT_SLOT_CHOICES}

APP_CSS = """
body, .gradio-container { background: #f5f7fb !important; }
.mao-shell { max-width: 1380px; margin: 0 auto; }
.mao-card {
  border: 1px solid #dbe4f0;
  border-radius: 14px;
  background: #fff;
  padding: 14px;
}
#agent-row { display: flex; flex-wrap: wrap; gap: 12px; }
#agent-row > div { flex: 1 1 260px; min-width: 260px; }
#config-row { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }
#config-row > div { flex: 1 1 320px; min-width: 320px; }
.status-ok { color: #0f766e; font-weight: 700; }
.status-run { color: #1d4ed8; font-weight: 700; }
.status-err { color: #b91c1c; font-weight: 700; }
"""


def _parse_assignments(values: list[str]) -> list[str]:
    return [LEFT_LABEL_TO_SLOT[value] for value in (values or []) if value in LEFT_LABEL_TO_SLOT]


def _make_tool_rows(
    tool1_enabled: bool,
    tool1_name: str,
    tool1_assigned: list[str],
    tool2_enabled: bool,
    tool2_name: str,
    tool2_assigned: list[str],
    tool3_enabled: bool,
    tool3_name: str,
    tool3_assigned: list[str],
    tool4_enabled: bool,
    tool4_name: str,
    tool4_assigned: list[str],
    tool5_enabled: bool,
    tool5_name: str,
    tool5_assigned: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "tool_id": AVAILABLE_TOOLS[0]["id"],
            "enabled": tool1_enabled,
            "name": tool1_name,
            "assigned_agent_ids": _parse_assignments(tool1_assigned),
        },
        {
            "tool_id": AVAILABLE_TOOLS[1]["id"],
            "enabled": tool2_enabled,
            "name": tool2_name,
            "assigned_agent_ids": _parse_assignments(tool2_assigned),
        },
        {
            "tool_id": AVAILABLE_TOOLS[2]["id"],
            "enabled": tool3_enabled,
            "name": tool3_name,
            "assigned_agent_ids": _parse_assignments(tool3_assigned),
        },
        {
            "tool_id": AVAILABLE_TOOLS[3]["id"],
            "enabled": tool4_enabled,
            "name": tool4_name,
            "assigned_agent_ids": _parse_assignments(tool4_assigned),
        },
        {
            "tool_id": AVAILABLE_TOOLS[4]["id"],
            "enabled": tool5_enabled,
            "name": tool5_name,
            "assigned_agent_ids": _parse_assignments(tool5_assigned),
        },
    ]


def _build_architecture_preview(main_agent: AgentProfile, sub_agents: list[AgentProfile], tools: list[ToolConfig]) -> str:
    lines = [
        "### Agent Architecture",
        f"- **Main Agent:** {main_agent.name} ({main_agent.role})",
    ]
    if sub_agents:
        lines.append("- **Sub-Agents:**")
        for agent in sub_agents:
            lines.append(f"  - {agent.name} | {agent.specialization}")
    else:
        lines.append("- **Sub-Agents:** None")
    if tools:
        lines.append("- **Tools:**")
        for tool in tools:
            lines.append(f"  - {tool.name} -> {', '.join(tool.assigned_agent_ids)}")
    else:
        lines.append("- **Tools:** None")
    return "\n".join(lines)


def _format_live_trace(steps: list[str]) -> str:
    lines = ["### Execution Trace"]
    if not steps:
        lines.append("- Waiting for workflow start.")
        return "\n".join(lines)
    for idx, step in enumerate(steps, start=1):
        lines.append(f"{idx}. {step}")
    return "\n".join(lines)


def _error_outputs(message: str) -> tuple[str, str, str, str, str, str]:
    status = f'<span class="status-err">{message}</span>'
    return (
        status,
        "### Agent Architecture\n-",
        "### Agent Plan\n-",
        "### Execution Trace\n-",
        "### Sources Used\n-",
        "### Final Answer\n-",
    )


def run_orchestration_stream(
    api_key: str,
    model_id: str,
    main_name: str,
    main_role: str,
    main_instruction: str,
    sub1_enabled: bool,
    sub1_name: str,
    sub1_spec: str,
    sub1_instruction: str,
    sub2_enabled: bool,
    sub2_name: str,
    sub2_spec: str,
    sub2_instruction: str,
    sub3_enabled: bool,
    sub3_name: str,
    sub3_spec: str,
    sub3_instruction: str,
    tool1_enabled: bool,
    tool1_name: str,
    tool1_assigned: list[str],
    tool2_enabled: bool,
    tool2_name: str,
    tool2_assigned: list[str],
    tool3_enabled: bool,
    tool3_name: str,
    tool3_assigned: list[str],
    tool4_enabled: bool,
    tool4_name: str,
    tool4_assigned: list[str],
    tool5_enabled: bool,
    tool5_name: str,
    tool5_assigned: list[str],
    task: str,
):
    """Stream workflow progress and outputs."""
    key = (api_key or "").strip()
    user_task = (task or "").strip()
    if not key:
        yield _error_outputs("Please enter your Gemini API key.")
        return
    if not user_task:
        yield _error_outputs("Please enter a task before running.")
        return

    main_agent = create_main_agent(main_name, main_role, main_instruction)
    sub_agents: list[AgentProfile] = []
    if sub1_enabled:
        sub_agents.append(create_sub_agent(1, sub1_name, sub1_spec, sub1_instruction))
    if sub2_enabled:
        sub_agents.append(create_sub_agent(2, sub2_name, sub2_spec, sub2_instruction))
    if sub3_enabled:
        sub_agents.append(create_sub_agent(3, sub3_name, sub3_spec, sub3_instruction))

    valid_slots = {"main"} | {agent.agent_id for agent in sub_agents}
    tool_rows = _make_tool_rows(
        tool1_enabled,
        tool1_name,
        tool1_assigned,
        tool2_enabled,
        tool2_name,
        tool2_assigned,
        tool3_enabled,
        tool3_name,
        tool3_assigned,
        tool4_enabled,
        tool4_name,
        tool4_assigned,
        tool5_enabled,
        tool5_name,
        tool5_assigned,
    )
    tools = create_tools(tool_rows, valid_slots)

    architecture_preview = _build_architecture_preview(main_agent, sub_agents, tools)
    live_steps: list[str] = []
    trace_md = _format_live_trace(live_steps)
    waiting = "### Pending\n- Waiting for completion."

    yield (
        '<span class="status-run">Starting workflow...</span>',
        architecture_preview,
        waiting,
        trace_md,
        "### Sources Used\n- Running...",
        "### Final Answer\n- Running...",
    )

    events: queue.Queue = queue.Queue()
    result_ref: dict[str, WorkflowResult] = {}
    error_ref: dict[str, str] = {}

    def _on_step(step: str) -> None:
        events.put(("step", step))

    def _worker() -> None:
        try:
            result = run_workflow(
                api_key=key,
                model_id=model_id,
                main_agent=main_agent,
                sub_agents=sub_agents,
                tools=tools,
                task=user_task,
                progress_callback=_on_step,
            )
            result_ref["value"] = result
            events.put(("done", "ok"))
        except Exception as exc:  # pylint: disable=broad-except
            error_ref["value"] = str(exc)
            events.put(("done", "error"))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    done = False
    while not done:
        try:
            kind, payload = events.get(timeout=0.25)
        except queue.Empty:
            continue

        if kind == "step":
            live_steps.append(payload)
            trace_md = _format_live_trace(live_steps)
            yield (
                f'<span class="status-run">Running: {payload}</span>',
                architecture_preview,
                waiting,
                trace_md,
                "### Sources Used\n- Running...",
                "### Final Answer\n- Running...",
            )
        elif kind == "done":
            done = True

    if "value" in error_ref:
        trace_md = _format_live_trace(live_steps)
        yield (
            f'<span class="status-err">Workflow failed: {error_ref["value"]}</span>',
            architecture_preview,
            waiting,
            trace_md,
            "### Sources Used\n- No sources captured.",
            "### Final Answer\n- No final answer generated.",
        )
        return

    result = result_ref["value"]
    yield (
        '<span class="status-ok">Workflow completed successfully.</span>',
        result.architecture_summary,
        result.agent_plan,
        result.execution_trace,
        result.sources_used,
        result.final_answer,
    )


def build_demo() -> gr.Blocks:
    """Create Gradio interface."""
    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Column(elem_classes="mao-shell"):
            gr.Markdown(f"# {APP_TITLE}\n{APP_DESCRIPTION}")

            # Row 1: four agent cards (responsive).
            with gr.Row(elem_id="agent-row"):
                with gr.Column(elem_classes="mao-card"):
                    gr.Markdown("### Main Agent")
                    main_name = gr.Textbox(label="Name", value="Coordinator")
                    main_role = gr.Textbox(label="Role", value="Task planner")
                    main_instruction = gr.Textbox(
                        label="Instruction",
                        lines=4,
                        value="Break tasks into subtasks and delegate them to the best sub-agent.",
                    )

                with gr.Column(elem_classes="mao-card"):
                    gr.Markdown("### Sub-agent 1")
                    sub1_enabled = gr.Checkbox(label="Enable", value=True)
                    sub1_name = gr.Textbox(label="Name", value="Researcher")
                    sub1_spec = gr.Textbox(label="Specialization", value="information gathering")
                    sub1_instruction = gr.Textbox(
                        label="Instruction",
                        lines=3,
                        value="Search and summarize factual information.",
                    )

                with gr.Column(elem_classes="mao-card"):
                    gr.Markdown("### Sub-agent 2")
                    sub2_enabled = gr.Checkbox(label="Enable", value=True)
                    sub2_name = gr.Textbox(label="Name", value="Writer")
                    sub2_spec = gr.Textbox(label="Specialization", value="writing and synthesis")
                    sub2_instruction = gr.Textbox(
                        label="Instruction",
                        lines=3,
                        value="Draft clear and structured responses from findings.",
                    )

                with gr.Column(elem_classes="mao-card"):
                    gr.Markdown("### Sub-agent 3")
                    sub3_enabled = gr.Checkbox(label="Enable", value=False)
                    sub3_name = gr.Textbox(label="Name", value="Critic")
                    sub3_spec = gr.Textbox(label="Specialization", value="quality review")
                    sub3_instruction = gr.Textbox(
                        label="Instruction",
                        lines=3,
                        value="Review responses for clarity, accuracy, and completeness.",
                    )

            # Row 2: tools + runtime config + task.
            slot_labels = [label for label, _ in AGENT_SLOT_CHOICES]
            with gr.Row(elem_id="config-row"):
                with gr.Column(elem_classes="mao-card"):
                    gr.Markdown("### Tools")
                    with gr.Accordion(AVAILABLE_TOOLS[0]["name"], open=False):
                        tool1_enabled = gr.Checkbox(label="Enable", value=True)
                        tool1_name = gr.Textbox(label="Tool Name", value=AVAILABLE_TOOLS[0]["name"])
                        tool1_assigned = gr.CheckboxGroup(label="Assign To", choices=slot_labels, value=["Sub-agent 1"])
                    with gr.Accordion(AVAILABLE_TOOLS[1]["name"], open=False):
                        tool2_enabled = gr.Checkbox(label="Enable", value=True)
                        tool2_name = gr.Textbox(label="Tool Name", value=AVAILABLE_TOOLS[1]["name"])
                        tool2_assigned = gr.CheckboxGroup(
                            label="Assign To",
                            choices=slot_labels,
                            value=["Sub-agent 1", "Sub-agent 2"],
                        )
                    with gr.Accordion(AVAILABLE_TOOLS[2]["name"], open=False):
                        tool3_enabled = gr.Checkbox(label="Enable", value=False)
                        tool3_name = gr.Textbox(label="Tool Name", value=AVAILABLE_TOOLS[2]["name"])
                        tool3_assigned = gr.CheckboxGroup(label="Assign To", choices=slot_labels, value=["Sub-agent 2"])
                    with gr.Accordion(AVAILABLE_TOOLS[3]["name"], open=False):
                        tool4_enabled = gr.Checkbox(label="Enable", value=False)
                        tool4_name = gr.Textbox(label="Tool Name", value=AVAILABLE_TOOLS[3]["name"])
                        tool4_assigned = gr.CheckboxGroup(label="Assign To", choices=slot_labels, value=["Sub-agent 2"])
                    with gr.Accordion(AVAILABLE_TOOLS[4]["name"], open=False):
                        tool5_enabled = gr.Checkbox(label="Enable", value=False)
                        tool5_name = gr.Textbox(label="Tool Name", value=AVAILABLE_TOOLS[4]["name"])
                        tool5_assigned = gr.CheckboxGroup(label="Assign To", choices=slot_labels, value=["Main Agent"])

                with gr.Column(elem_classes="mao-card"):
                    gr.Markdown("### LLM Configuration")
                    api_key = gr.Textbox(label="Gemini API Key", type="password", placeholder="Paste API key")
                    model = gr.Dropdown(label="Model", choices=[MODEL_ID], value=MODEL_ID, interactive=False)
                    run_button = gr.Button("Run Multi-Agent Workflow", variant="primary")

                with gr.Column(elem_classes="mao-card"):
                    gr.Markdown("### Task")
                    task = gr.Textbox(
                        label="Task Input",
                        lines=7,
                        placeholder="Example: Explain the EU AI Act and its impact on startups.",
                    )
                    gr.Examples(examples=[[item] for item in EXAMPLE_TASKS], inputs=[task], label="Example Tasks")

            # Row 3: execution results.
            with gr.Column(elem_classes="mao-card"):
                gr.Markdown("## Execution Results")
                status = gr.HTML('<span class="status-run">Idle. Configure and run workflow.</span>')
                trace = gr.Markdown("### Execution Trace\n- Waiting for execution.")
                with gr.Tabs():
                    with gr.Tab("Architecture"):
                        architecture = gr.Markdown("### Agent Architecture\n-")
                    with gr.Tab("Plan"):
                        plan = gr.Markdown("### Agent Plan\n-")
                    with gr.Tab("Sources"):
                        sources = gr.Markdown("### Sources Used\n-")
                    with gr.Tab("Final Answer"):
                        final_answer = gr.Markdown("### Final Answer\n-")

            run_button.click(
                fn=run_orchestration_stream,
                inputs=[
                    api_key,
                    model,
                    main_name,
                    main_role,
                    main_instruction,
                    sub1_enabled,
                    sub1_name,
                    sub1_spec,
                    sub1_instruction,
                    sub2_enabled,
                    sub2_name,
                    sub2_spec,
                    sub2_instruction,
                    sub3_enabled,
                    sub3_name,
                    sub3_spec,
                    sub3_instruction,
                    tool1_enabled,
                    tool1_name,
                    tool1_assigned,
                    tool2_enabled,
                    tool2_name,
                    tool2_assigned,
                    tool3_enabled,
                    tool3_name,
                    tool3_assigned,
                    tool4_enabled,
                    tool4_name,
                    tool4_assigned,
                    tool5_enabled,
                    tool5_name,
                    tool5_assigned,
                    task,
                ],
                outputs=[status, architecture, plan, trace, sources, final_answer],
            )

    return demo


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7862")))
    build_demo().queue().launch(server_name=server_name, server_port=server_port, css=APP_CSS)
