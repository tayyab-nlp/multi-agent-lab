"""Multi-Agent Orchestrator Gradio application."""

from __future__ import annotations

import html
import os
import queue
import re
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

TAB_LIVE = "live"
TAB_TRACE = "trace"
TAB_IO = "agent-io"
TAB_ARCH = "arch"
TAB_PLAN = "plan"
TAB_SOURCES = "sources"
TAB_FINAL = "final"

APP_CSS = """
body, .gradio-container { background: #f5f7fb !important; }
.mao-shell { max-width: 1400px; margin: 0 auto; }
.mao-panel {
  border: 1px solid #dbe4f0;
  border-radius: 14px;
  background: #fff;
  padding: 14px;
}
#left-panel {
  position: sticky;
  top: 16px;
  align-self: flex-start;
}
.status-ok { color: #0f766e; font-weight: 700; }
.status-run { color: #1d4ed8; font-weight: 700; }
.status-err { color: #b91c1c; font-weight: 700; }
.results-pane .prose h1,
.results-pane .prose h2,
.results-pane .prose h3,
.results-pane .prose h4,
.results-pane .prose h5,
.results-pane .prose h6 { border-bottom: none !important; padding-bottom: 0 !important; }
.results-pane .prose hr { display: none !important; }
.agent-io-wrap { display: grid; gap: 12px; }
.agent-io-card {
  border: 1px solid #dbe4f0;
  border-radius: 12px;
  background: #fbfdff;
  padding: 12px;
}
.agent-io-title { font-weight: 700; margin-bottom: 6px; }
.agent-io-meta { color: #475569; font-size: 13px; margin-bottom: 8px; }
.agent-io-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.agent-io-block {
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #ffffff;
  padding: 10px;
  line-height: 1.4;
  white-space: normal;
  word-break: break-word;
}
.agent-io-label { font-weight: 700; margin-bottom: 6px; }
@media (max-width: 1100px) {
  #left-panel { position: static; }
}
@media (max-width: 860px) {
  .agent-io-grid { grid-template-columns: 1fr; }
}
"""


def _parse_assignments(values: list[str] | None) -> list[str]:
    return [LEFT_LABEL_TO_SLOT[v] for v in (values or []) if v in LEFT_LABEL_TO_SLOT]


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


def _architecture_preview(main_agent: AgentProfile, sub_agents: list[AgentProfile], tools: list[ToolConfig]) -> str:
    lines = [
        f"- Main Agent: **{main_agent.name}** ({main_agent.role})",
    ]
    if sub_agents:
        lines.append("- Sub-Agents:")
        for agent in sub_agents:
            lines.append(f"  - {agent.name} | {agent.specialization}")
    else:
        lines.append("- Sub-Agents: None")

    if tools:
        lines.append("- Enabled Tools:")
        for tool in tools:
            lines.append(f"  - {tool.name} -> {', '.join(tool.assigned_agent_ids)}")
    else:
        lines.append("- Enabled Tools: None")
    return "\n".join(lines)


def _trace_md(steps: list[str]) -> str:
    if not steps:
        return "- Waiting for workflow start."
    lines = []
    for idx, step in enumerate(steps, start=1):
        lines.append(f"{idx}. {step}")
    return "\n".join(lines)


def _live_status_md(steps: list[str]) -> str:
    lines = [f"- Steps completed: **{len(steps)}**"]
    if steps:
        lines.append(f"- Latest update: **{steps[-1]}**")
        lines.append("- Recent updates:")
        for item in steps[-6:]:
            lines.append(f"  - {item}")
    else:
        lines.append("- Latest update: waiting...")
    return "\n".join(lines)


def _drop_title_line(text: str) -> str:
    lines = (text or "").splitlines()
    while lines and lines[0].strip().startswith("#"):
        lines.pop(0)
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def _clean_markdown(text: str) -> str:
    """Normalize markdown from model output for cleaner rendering."""
    raw = _drop_title_line((text or "").replace("\r\n", "\n")).strip()
    if not raw:
        return "-"

    fixed: list[str] = []
    for line in raw.split("\n"):
        current = line.rstrip()
        stripped = current.strip()

        if stripped in {"---", "***", "___"}:
            continue

        current = re.sub(r"^(#{1,6})(\S)", r"\1 \2", current)

        # Split malformed in-line bullets into separate bullets.
        if stripped.startswith("* ") and " * " in stripped:
            for part in stripped.split(" * "):
                part = part.strip()
                if not part:
                    continue
                fixed.append(part if part.startswith("*") else f"* {part}")
            continue

        fixed.append(current)

    cleaned = "\n".join(fixed)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() or "-"


def _format_block_html(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "<em>None</em>"
    return html.escape(cleaned).replace("\n", "<br>")


def _agent_io_html(agent_io: list[dict[str, Any]] | None) -> str:
    if not agent_io:
        return "<div class='agent-io-card'>Agent input/output will appear after sub-agents complete their steps.</div>"

    cards: list[str] = ["<div class='agent-io-wrap'>"]
    for idx, item in enumerate(agent_io, start=1):
        agent = html.escape(str(item.get("agent", "Agent")))
        subtask = html.escape(str(item.get("subtask", "Untitled subtask")))
        tools = item.get("tools") or []
        tools_text = html.escape(", ".join(str(tool) for tool in tools)) if tools else "None"
        input_html = _format_block_html(str(item.get("input", "")))
        output_html = _format_block_html(str(item.get("output", "")))
        cards.append(
            """
            <div class='agent-io-card'>
              <div class='agent-io-title'>Step {idx}: {agent}</div>
              <div class='agent-io-meta'>Subtask: {subtask}<br>Tools: {tools_text}</div>
              <div class='agent-io-grid'>
                <div class='agent-io-block'>
                  <div class='agent-io-label'>Agent Input</div>
                  {input_html}
                </div>
                <div class='agent-io-block'>
                  <div class='agent-io-label'>Agent Output</div>
                  {output_html}
                </div>
              </div>
            </div>
            """.format(
                idx=idx,
                agent=agent,
                subtask=subtask,
                tools_text=tools_text,
                input_html=input_html,
                output_html=output_html,
            )
        )
    cards.append("</div>")
    return "\n".join(cards)


def _error_outputs(message: str) -> tuple[Any, ...]:
    return (
        f'<span class="status-err">{message}</span>',
        gr.Tabs(selected=TAB_LIVE),
        "- Workflow failed before start.",
        "-",
        _agent_io_html([]),
        "-",
        "-",
        "-",
        "-",
    )


def run_orchestration_stream(
    api_key: str,
    model_id: str,
    main_name: str,
    main_role: str,
    main_instruction: str,
    task: str,
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
):
    """Stream workflow progress and outputs in real time."""
    key = (api_key or "").strip()
    user_task = (task or "").strip()
    if not key:
        yield _error_outputs("Please enter your Gemini API key in the API Config tab.")
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
    architecture_preview = _architecture_preview(main_agent, sub_agents, tools)

    live_steps: list[str] = []
    yield (
        '<span class="status-run">Starting workflow...</span>',
        gr.Tabs(selected=TAB_LIVE),
        _live_status_md(live_steps),
        _trace_md(live_steps),
        _agent_io_html([]),
        architecture_preview,
        "Planning in progress...",
        "Running...",
        "Running...",
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
            yield (
                f'<span class="status-run">Running: {payload}</span>',
                gr.Tabs(selected=TAB_LIVE),
                _live_status_md(live_steps),
                _trace_md(live_steps),
                _agent_io_html([]),
                architecture_preview,
                "Planning/execution in progress...",
                "Running...",
                "Running...",
            )
        elif kind == "done":
            done = True

    if "value" in error_ref:
        yield (
            f'<span class="status-err">Workflow failed: {error_ref["value"]}</span>',
            gr.Tabs(selected=TAB_LIVE),
            _live_status_md(live_steps),
            _trace_md(live_steps),
            _agent_io_html([]),
            architecture_preview,
            "Could not complete planning.",
            "No sources captured.",
            "No final answer generated.",
        )
        return

    result = result_ref["value"]
    yield (
        '<span class="status-ok">Workflow completed successfully.</span>',
        gr.Tabs(selected=TAB_FINAL),
        _live_status_md(live_steps),
        _clean_markdown(result.execution_trace),
        _agent_io_html(result.agent_io),
        _clean_markdown(result.architecture_summary),
        _clean_markdown(result.agent_plan),
        _clean_markdown(result.sources_used),
        _clean_markdown(result.final_answer),
    )


def build_demo() -> gr.Blocks:
    """Create Gradio interface."""
    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Column(elem_classes="mao-shell"):
            gr.Markdown(f"# {APP_TITLE}\n{APP_DESCRIPTION}")

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=420, elem_classes="mao-panel", elem_id="left-panel"):
                    with gr.Tabs():
                        with gr.Tab("Task & Run"):
                            with gr.Accordion("Task", open=True):
                                task = gr.Textbox(
                                    label="Task Input",
                                    lines=6,
                                    placeholder="Example: Explain the EU AI Act and its impact on startups.",
                                )
                                gr.Examples(examples=[[item] for item in EXAMPLE_TASKS], inputs=[task], label="Example Tasks")

                            with gr.Accordion("Model", open=True):
                                model = gr.Dropdown(label="Model", choices=[MODEL_ID], value=MODEL_ID, interactive=False)

                            run_button = gr.Button("Run Multi-Agent Workflow", variant="primary")

                        with gr.Tab("Main Agent"):
                            with gr.Accordion("Main Agent Configuration", open=True):
                                main_name = gr.Textbox(label="Agent Name", value="Coordinator")
                                main_role = gr.Textbox(label="Agent Role", value="Task planner")
                                main_instruction = gr.Textbox(
                                    label="Agent Instruction",
                                    lines=3,
                                    value="Break tasks into subtasks and delegate them to the best sub-agent.",
                                )

                        with gr.Tab("Sub-Agents"):
                            with gr.Accordion("Sub-agent 1", open=True):
                                sub1_enabled = gr.Checkbox(label="Enable", value=True)
                                sub1_name = gr.Textbox(label="Name", value="Researcher")
                                sub1_spec = gr.Textbox(label="Specialization", value="information gathering")
                                sub1_instruction = gr.Textbox(
                                    label="Instruction",
                                    lines=3,
                                    value="Search and summarize factual information.",
                                )
                            with gr.Accordion("Sub-agent 2", open=False):
                                sub2_enabled = gr.Checkbox(label="Enable", value=True)
                                sub2_name = gr.Textbox(label="Name", value="Writer")
                                sub2_spec = gr.Textbox(label="Specialization", value="writing and synthesis")
                                sub2_instruction = gr.Textbox(
                                    label="Instruction",
                                    lines=3,
                                    value="Draft clear and structured responses from findings.",
                                )
                            with gr.Accordion("Sub-agent 3", open=False):
                                sub3_enabled = gr.Checkbox(label="Enable", value=False)
                                sub3_name = gr.Textbox(label="Name", value="Critic")
                                sub3_spec = gr.Textbox(label="Specialization", value="quality review")
                                sub3_instruction = gr.Textbox(
                                    label="Instruction",
                                    lines=3,
                                    value="Review responses for clarity, accuracy, and completeness.",
                                )

                        with gr.Tab("Tools"):
                            slot_labels = [label for label, _ in AGENT_SLOT_CHOICES]
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

                        with gr.Tab("API Config"):
                            with gr.Accordion("Gemini API Key", open=True):
                                api_key = gr.Textbox(label="Gemini API Key", type="password", placeholder="Paste API key")
                                gr.Markdown("- API key is used only for the current run.\n- It is not stored or logged.")

                with gr.Column(scale=2, min_width=560, elem_classes="mao-panel"):
                    gr.Markdown("## Results")
                    status = gr.HTML('<span class="status-run">Idle. Configure workflow and run.</span>')

                    with gr.Tabs(selected=TAB_LIVE) as results_tabs:
                        with gr.Tab("Live Progress", id=TAB_LIVE):
                            live_status = gr.Markdown("- Waiting for workflow start.", elem_classes=["results-pane"])
                        with gr.Tab("Execution Trace", id=TAB_TRACE):
                            trace = gr.Markdown("- Waiting for execution.", elem_classes=["results-pane"])
                        with gr.Tab("Agent I/O", id=TAB_IO):
                            agent_io_view = gr.HTML(_agent_io_html([]))
                        with gr.Tab("Architecture", id=TAB_ARCH):
                            architecture = gr.Markdown("-", elem_classes=["results-pane"])
                        with gr.Tab("Plan", id=TAB_PLAN):
                            plan = gr.Markdown("-", elem_classes=["results-pane"])
                        with gr.Tab("Sources", id=TAB_SOURCES):
                            sources = gr.Markdown("-", elem_classes=["results-pane"])
                        with gr.Tab("Final Answer", id=TAB_FINAL):
                            final_answer = gr.Markdown("-", elem_classes=["results-pane"])

            run_button.click(
                fn=run_orchestration_stream,
                inputs=[
                    api_key,
                    model,
                    main_name,
                    main_role,
                    main_instruction,
                    task,
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
                ],
                outputs=[
                    status,
                    results_tabs,
                    live_status,
                    trace,
                    agent_io_view,
                    architecture,
                    plan,
                    sources,
                    final_answer,
                ],
            )

    return demo


if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7862")))
    build_demo().queue().launch(
        server_name=server_name,
        server_port=server_port,
        theme=gr.themes.Default(),
        css=APP_CSS,
    )
