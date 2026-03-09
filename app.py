"""Multi-Agent Orchestrator Gradio application."""

from __future__ import annotations

import os
import queue
import re
import threading
import json
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
    MODEL_OPTIONS,
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
body, .gradio-container { background: #ffffff !important; }
:root, .gradio-container {
  --body-background-fill: #ffffff !important;
  --block-background-fill: #ffffff !important;
  --block-background-fill-dark: #ffffff !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: #ffffff !important;
  --panel-background-fill: #ffffff !important;
  --color-accent-soft: #ffffff !important;
  --input-background-fill: #ffffff !important;
  --input-background-fill-focus: #ffffff !important;
}
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain {
  background: #ffffff !important;
}
.mao-shell { max-width: 1400px; margin: 0 auto; }
.mao-panel {
  border: 1px solid #dbe4f0;
  border-radius: 14px;
  background: #fff;
  padding: 12px;
}
#left-panel {
  position: sticky;
  top: 16px;
  align-self: flex-start;
}
.left-tabs .tab-nav {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 4px !important;
  overflow: visible !important;
}
.left-tabs .tab-nav button {
  flex: 0 1 auto;
  min-width: 0 !important;
  padding: 7px 9px !important;
  font-size: 14px !important;
}
.left-tabs .tab-nav button[aria-haspopup="menu"] {
  display: none !important;
}
.left-tab-body {
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #ffffff;
  padding: 10px;
}
#left-panel .prose,
#left-panel .prose p,
#left-panel .prose li,
#left-panel .markdown {
  background: transparent !important;
}
#left-panel .block,
#left-panel .gr-group,
#left-panel .styler,
#left-panel .form {
  background: #ffffff !important;
  border-color: #dbe4f0 !important;
  box-shadow: none !important;
}
#left-panel .block {
  border-width: 1px !important;
}
#left-panel .examples,
#left-panel .examples .table-wrap,
#left-panel .examples table,
#left-panel .examples table tr,
#left-panel .examples table td,
#left-panel .examples table th {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
#left-panel .gallery,
#left-panel .gallery-item,
#left-panel [class*="gallery"] {
  background: #ffffff !important;
  border: none !important;
  box-shadow: none !important;
}
#left-panel .examples {
  padding: 0 !important;
}
.example-task-btn {
  justify-content: flex-start !important;
  text-align: left !important;
  border: 1px solid #dbe4f0 !important;
  background: #ffffff !important;
}
.example-task-btn button,
#left-panel .example-task-btn button {
  background: #ffffff !important;
  border: 1px solid #dbe4f0 !important;
  box-shadow: none !important;
  color: #111827 !important;
}
#left-panel details,
#left-panel .accordion {
  border: 1px solid #dbe4f0 !important;
  box-shadow: none !important;
  outline: none !important;
}
#left-panel details > summary,
#left-panel .accordion summary {
  border-bottom: 1px solid #e2e8f0 !important;
}
.status-ok { color: #0f766e; font-weight: 700; }
.status-run { color: #1d4ed8; font-weight: 700; }
.status-err { color: #b91c1c; font-weight: 700; }
.results-tabs .tabitem {
  padding-top: 10px !important;
}
.results-pane .prose h1,
.results-pane .prose h2,
.results-pane .prose h3,
.results-pane .prose h4,
.results-pane .prose h5,
.results-pane .prose h6 { border-bottom: none !important; padding-bottom: 0 !important; }
.results-pane .prose hr { display: none !important; }
.result-markdown {
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #ffffff;
  padding: 12px 14px !important;
}
@media (max-width: 1100px) {
  #left-panel { position: static; }
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
    lines = ["### Live Status", f"- Steps completed: **{len(steps)}**"]
    if steps:
        lines.append(f"- Latest update: **{steps[-1]}**")
        lines.append("")
        lines.append("### Recent Updates")
        for idx, item in enumerate(steps[-6:], start=1):
            lines.append(f"{idx}. {item}")
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
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.split("\n")
        if len(lines) >= 2:
            lines = lines[1:-1]
            if lines and lines[0].strip().lower() == "markdown":
                lines = lines[1:]
            raw = "\n".join(lines).strip()

    if raw and ((raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'"))):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                raw = parsed.strip()
        except Exception:  # pylint: disable=broad-except
            raw = raw[1:-1].strip()

    raw = raw.replace("\\n", "\n").replace("\\t", "\t")
    raw = re.sub(r'^"+', "", raw)
    raw = re.sub(r'"+$', "", raw)

    if not raw:
        return "-"

    fixed: list[str] = []
    in_code_block = False
    for line in raw.split("\n"):
        current = line.rstrip("\n")
        if current.strip().startswith("```"):
            in_code_block = not in_code_block
            fixed.append(current.strip())
            continue

        if in_code_block:
            fixed.append(current.rstrip())
            continue

        current = current.strip()
        stripped = current.strip()

        if stripped in {"---", "***", "___"}:
            continue

        current = current.replace("•", "- ").replace("◦", "- ").replace("○", "- ").replace("●", "- ")
        current = re.sub(r"^\s*[oO]\s+", "- ", current)
        current = re.sub(r"^\s*(#{1,6})\s*(.+)$", r"\1 \2", current)

        # Split malformed in-line bullets into separate bullets.
        if " * " in stripped:
            for part in stripped.split(" * "):
                part = part.strip()
                if not part:
                    continue
                fixed.append(part if part.startswith("*") else f"* {part}")
            continue

        fixed.append(current)

    padded: list[str] = []
    for line in fixed:
        if re.match(r"^#{1,6}\s", line) and padded and padded[-1].strip():
            padded.append("")
        padded.append(line)

    cleaned = "\n".join(padded)
    cleaned = re.sub(r"([.!?])\s+(#{1,6}\s)", r"\1\n\n\2", cleaned)
    cleaned = re.sub(r"([.!?])\s+\*\s+", r"\1\n* ", cleaned)
    cleaned = re.sub(r"(?m)^-\s{2,}", "- ", cleaned)
    cleaned = re.sub(r"\n[-*]\s+\*\*", "\n* **", cleaned)
    cleaned = re.sub(r'(?m)^\s*"+\s*(#{1,6}\s)', r"\1", cleaned)
    cleaned = re.sub(r'(?m)^\s*"+\s*[-*]\s+', "* ", cleaned)
    cleaned = re.sub(r'(?m)\s*"+\s*$', "", cleaned)
    cleaned = re.sub(r"(?m)^(#{1,6}\s.+)$", r"\n\1\n", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() or "-"


def _agent_slot_md(
    agent_io: list[dict[str, Any]] | None,
    enabled: bool,
    agent_name: str,
    slot_label: str,
) -> str:
    if not enabled:
        return f"### {slot_label}\n- Disabled for this run."

    display_name = (agent_name or slot_label).strip()
    if not agent_io:
        return f"### {display_name}\n- Waiting for completed steps."

    items = [item for item in agent_io if str(item.get("agent", "")).strip().lower() == display_name.lower()]
    if not items:
        return f"### {display_name}\n- No steps completed yet."

    lines = [f"### {display_name}"]
    for idx, item in enumerate(items, start=1):
        lines.append(f"#### Step {idx}: {item.get('subtask', 'Untitled subtask')}")
        tools = item.get("tools") or []
        lines.append(f"- Tools: {', '.join(str(tool) for tool in tools) if tools else 'None'}")
        lines.append("")
        lines.append("**Input**")
        lines.append("```text")
        lines.append(str(item.get("input", "")).replace("```", "'''").strip())
        lines.append("```")
        lines.append("")
        lines.append("**Output**")
        lines.append("```text")
        lines.append(str(item.get("output", "")).replace("```", "'''").strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines).strip()


def _error_outputs(message: str) -> tuple[Any, ...]:
    return (
        f'<span class="status-err">{message}</span>',
        gr.Tabs(selected=TAB_LIVE),
        "### Live Status\n- Workflow failed before start.",
        "-",
        "-",
        "### Sub-agent 1\n- No data.",
        "### Sub-agent 2\n- No data.",
        "### Sub-agent 3\n- No data.",
        "-",
        "-",
        "-",
        "",
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
        yield _error_outputs("Please enter your Gemini API key in API Config.")
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
        "Planning in progress...",
        _trace_md(live_steps),
        _agent_slot_md([], sub1_enabled, sub1_name, "Sub-agent 1"),
        _agent_slot_md([], sub2_enabled, sub2_name, "Sub-agent 2"),
        _agent_slot_md([], sub3_enabled, sub3_name, "Sub-agent 3"),
        "Running...",
        architecture_preview,
        "Running...",
        "",
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
                "Planning/execution in progress...",
                _trace_md(live_steps),
                _agent_slot_md([], sub1_enabled, sub1_name, "Sub-agent 1"),
                _agent_slot_md([], sub2_enabled, sub2_name, "Sub-agent 2"),
                _agent_slot_md([], sub3_enabled, sub3_name, "Sub-agent 3"),
                "Running...",
                architecture_preview,
                "Running...",
                "",
            )
        elif kind == "done":
            done = True

    if "value" in error_ref:
        yield (
            f'<span class="status-err">Workflow failed: {error_ref["value"]}</span>',
            gr.Tabs(selected=TAB_LIVE),
            _live_status_md(live_steps),
            "Could not complete planning.",
            _trace_md(live_steps),
            _agent_slot_md([], sub1_enabled, sub1_name, "Sub-agent 1"),
            _agent_slot_md([], sub2_enabled, sub2_name, "Sub-agent 2"),
            _agent_slot_md([], sub3_enabled, sub3_name, "Sub-agent 3"),
            "No sources captured.",
            architecture_preview,
            "No final answer generated.",
            "",
        )
        return

    result = result_ref["value"]
    final_answer_clean = _clean_markdown(result.final_answer)
    yield (
        '<span class="status-ok">Workflow completed successfully.</span>',
        gr.Tabs(selected=TAB_FINAL),
        _live_status_md(live_steps),
        _clean_markdown(result.agent_plan),
        _clean_markdown(result.execution_trace),
        _agent_slot_md(result.agent_io, sub1_enabled, sub1_name, "Sub-agent 1"),
        _agent_slot_md(result.agent_io, sub2_enabled, sub2_name, "Sub-agent 2"),
        _agent_slot_md(result.agent_io, sub3_enabled, sub3_name, "Sub-agent 3"),
        _clean_markdown(result.sources_used),
        _clean_markdown(result.architecture_summary),
        final_answer_clean,
        final_answer_clean,
    )


def build_demo() -> gr.Blocks:
    """Create Gradio interface."""
    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Column(elem_classes="mao-shell"):
            gr.Markdown(
                f"# {APP_TITLE}\n"
                f"{APP_DESCRIPTION}\n\n"
                "[GitHub Repository](https://github.com/tayyab-nlp/multi-agent-lab)"
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=420, elem_classes="mao-panel", elem_id="left-panel"):
                    with gr.Tabs(elem_classes=["left-tabs"]):
                        with gr.Tab("Task"):
                            with gr.Group(elem_classes=["left-tab-body"]):
                                model = gr.Dropdown(label="Model", choices=MODEL_OPTIONS, value=MODEL_ID, interactive=True)
                                task = gr.Textbox(
                                    label="Task Input",
                                    lines=6,
                                    placeholder="Example: Explain the EU AI Act and its impact on startups.",
                                )
                                gr.Markdown("Example Tasks")
                                example_btn_1 = gr.Button(EXAMPLE_TASKS[0], elem_classes=["example-task-btn"])
                                example_btn_2 = gr.Button(EXAMPLE_TASKS[1], elem_classes=["example-task-btn"])
                                example_btn_3 = gr.Button(EXAMPLE_TASKS[2], elem_classes=["example-task-btn"])
                                run_button = gr.Button("Run Multi-Agent Workflow", variant="primary")

                        with gr.Tab("Agent"):
                            with gr.Group(elem_classes=["left-tab-body"]):
                                main_name = gr.Textbox(label="Agent Name", value="Coordinator")
                                main_role = gr.Textbox(label="Agent Role", value="Task planner")
                                main_instruction = gr.Textbox(
                                    label="Agent Instruction",
                                    lines=3,
                                    value="Break tasks into subtasks and delegate them to the best sub-agent.",
                                )

                        with gr.Tab("Sub-Agents"):
                            with gr.Tabs():
                                with gr.Tab("Sub-agent 1"):
                                    with gr.Group(elem_classes=["left-tab-body"]):
                                        sub1_enabled = gr.Checkbox(label="Enable", value=True)
                                        sub1_name = gr.Textbox(label="Name", value="Researcher")
                                        sub1_spec = gr.Textbox(label="Specialization", value="information gathering")
                                        sub1_instruction = gr.Textbox(
                                            label="Instruction",
                                            lines=3,
                                            value="Search and summarize factual information.",
                                        )
                                with gr.Tab("Sub-agent 2"):
                                    with gr.Group(elem_classes=["left-tab-body"]):
                                        sub2_enabled = gr.Checkbox(label="Enable", value=True)
                                        sub2_name = gr.Textbox(label="Name", value="Writer")
                                        sub2_spec = gr.Textbox(label="Specialization", value="writing and synthesis")
                                        sub2_instruction = gr.Textbox(
                                            label="Instruction",
                                            lines=3,
                                            value="Draft clear and structured responses from findings.",
                                        )
                                with gr.Tab("Sub-agent 3"):
                                    with gr.Group(elem_classes=["left-tab-body"]):
                                        sub3_enabled = gr.Checkbox(label="Enable", value=False)
                                        sub3_name = gr.Textbox(label="Name", value="Critic")
                                        sub3_spec = gr.Textbox(label="Specialization", value="quality review")
                                        sub3_instruction = gr.Textbox(
                                            label="Instruction",
                                            lines=3,
                                            value="Review responses for clarity, accuracy, and completeness.",
                                        )

                        with gr.Tab("Tools"):
                            with gr.Group(elem_classes=["left-tab-body"]):
                                slot_labels = [label for label, _ in AGENT_SLOT_CHOICES]
                                with gr.Accordion(AVAILABLE_TOOLS[0]["name"], open=True):
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
                            with gr.Group(elem_classes=["left-tab-body"]):
                                api_key = gr.Textbox(label="Gemini API Key", type="password", placeholder="Paste API key")
                                gr.Markdown("API key is used only for this run and is not stored or logged.")

                with gr.Column(scale=2, min_width=560, elem_classes="mao-panel"):
                    gr.Markdown("## Results")
                    status = gr.HTML('<span class="status-run">Idle. Configure workflow and run.</span>')

                    with gr.Tabs(selected=TAB_LIVE, elem_classes=["results-tabs"]) as results_tabs:
                        with gr.Tab("Live Progress", id=TAB_LIVE):
                            live_status = gr.Markdown(
                                "### Live Status\n- Waiting for workflow start.",
                                elem_classes=["results-pane", "result-markdown"],
                            )
                        with gr.Tab("Plan", id=TAB_PLAN):
                            plan = gr.Markdown("-", elem_classes=["results-pane", "result-markdown"])
                        with gr.Tab("Execution Trace", id=TAB_TRACE):
                            trace = gr.Markdown("-", elem_classes=["results-pane", "result-markdown"])
                        with gr.Tab("Agent I/O", id=TAB_IO):
                            with gr.Tabs():
                                with gr.Tab("Sub-agent 1"):
                                    agent_io_sub1 = gr.Markdown(
                                        "### Sub-agent 1\n- Waiting for workflow run.",
                                        elem_classes=["results-pane", "result-markdown"],
                                    )
                                with gr.Tab("Sub-agent 2"):
                                    agent_io_sub2 = gr.Markdown(
                                        "### Sub-agent 2\n- Waiting for workflow run.",
                                        elem_classes=["results-pane", "result-markdown"],
                                    )
                                with gr.Tab("Sub-agent 3"):
                                    agent_io_sub3 = gr.Markdown(
                                        "### Sub-agent 3\n- Waiting for workflow run.",
                                        elem_classes=["results-pane", "result-markdown"],
                                    )
                        with gr.Tab("Sources", id=TAB_SOURCES):
                            sources = gr.Markdown("-", elem_classes=["results-pane", "result-markdown"])
                        with gr.Tab("Architecture", id=TAB_ARCH):
                            architecture = gr.Markdown("-", elem_classes=["results-pane", "result-markdown"])
                        with gr.Tab("Final Answer", id=TAB_FINAL):
                            final_answer = gr.Markdown("-", elem_classes=["results-pane", "result-markdown"])
                            copy_final_btn = gr.Button("Copy Final Answer", variant="secondary")
                            copy_status = gr.Markdown("", elem_classes=["results-pane"])
                            final_answer_copy = gr.Textbox(
                                label="Copy-ready Final Answer",
                                lines=6,
                                interactive=False,
                            )

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
                    plan,
                    trace,
                    agent_io_sub1,
                    agent_io_sub2,
                    agent_io_sub3,
                    sources,
                    architecture,
                    final_answer,
                    final_answer_copy,
                ],
            )
            copy_final_btn.click(
                fn=lambda text: "Copied final answer." if (text or "").strip() else "No final answer to copy yet.",
                inputs=final_answer_copy,
                outputs=copy_status,
                js="""
                (text) => {
                  if (!text || !text.trim()) { return "No final answer to copy yet."; }
                  navigator.clipboard.writeText(text);
                  return "Copied final answer.";
                }
                """,
            )
            example_btn_1.click(fn=lambda: EXAMPLE_TASKS[0], outputs=task)
            example_btn_2.click(fn=lambda: EXAMPLE_TASKS[1], outputs=task)
            example_btn_3.click(fn=lambda: EXAMPLE_TASKS[2], outputs=task)

    return demo


if __name__ == "__main__":
    server_name = "0.0.0.0"
    server_port = int(os.getenv("PORT", "7860"))
    build_demo().queue().launch(
        server_name=server_name,
        server_port=server_port,
        theme=gr.themes.Default(),
        css=APP_CSS,
    )
