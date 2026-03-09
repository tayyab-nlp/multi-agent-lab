"""Multi-Agent Orchestrator Gradio application."""

from __future__ import annotations

import os

import gradio as gr

from src.agent_builder import create_main_agent, create_sub_agent
from src.config import (
    AGENT_SLOT_CHOICES,
    APP_DESCRIPTION,
    APP_TITLE,
    AVAILABLE_TOOLS,
    EXAMPLE_TASKS,
    MODEL_ID,
)
from src.orchestrator import run_workflow
from src.tool_builder import create_tools

LEFT_LABEL_TO_SLOT = {label: slot for label, slot in AGENT_SLOT_CHOICES}

APP_CSS = """
body, .gradio-container { background: #f6f8fc !important; }
.pvl-shell { max-width: 1300px; margin: 0 auto; }
.pvl-panel {
  border: 1px solid #dbe4f0;
  border-radius: 14px;
  background: #ffffff;
  padding: 14px;
}
#left-config { position: sticky; top: 12px; align-self: flex-start; }
.status-ok { color: #0f766e; font-weight: 700; }
.status-err { color: #b91c1c; font-weight: 700; }
"""


def _parse_assignments(values: list[str]) -> list[str]:
    slots = []
    for value in values or []:
        if value in LEFT_LABEL_TO_SLOT:
            slots.append(LEFT_LABEL_TO_SLOT[value])
    return slots


def run_orchestration(
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
    """Orchestrate configured multi-agent workflow and return UI outputs."""
    key = (api_key or "").strip()
    user_task = (task or "").strip()

    if not key:
        msg = '<span class="status-err">Please enter your Gemini API key.</span>'
        return msg, "", "", "", "", ""
    if not user_task:
        msg = '<span class="status-err">Please enter a task before running.</span>'
        return msg, "", "", "", "", ""

    main_agent = create_main_agent(main_name, main_role, main_instruction)

    sub_agents = []
    if sub1_enabled:
        sub_agents.append(create_sub_agent(1, sub1_name, sub1_spec, sub1_instruction))
    if sub2_enabled:
        sub_agents.append(create_sub_agent(2, sub2_name, sub2_spec, sub2_instruction))
    if sub3_enabled:
        sub_agents.append(create_sub_agent(3, sub3_name, sub3_spec, sub3_instruction))

    valid_slots = {"main"} | {agent.agent_id for agent in sub_agents}

    tool_rows = [
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

    tools = create_tools(tool_rows, valid_slots)

    try:
        result = run_workflow(
            api_key=key,
            model_id=model_id,
            main_agent=main_agent,
            sub_agents=sub_agents,
            tools=tools,
            task=user_task,
        )
    except Exception as exc:  # pylint: disable=broad-except
        status = f'<span class="status-err">Workflow failed: {exc}</span>'
        return status, "", "", "", "", ""

    status = '<span class="status-ok">Workflow completed successfully.</span>'
    return (
        status,
        result.architecture_summary,
        result.agent_plan,
        result.execution_trace,
        result.sources_used,
        result.final_answer,
    )


def build_demo() -> gr.Blocks:
    """Create Gradio interface."""
    with gr.Blocks(title=APP_TITLE) as demo:
        with gr.Column(elem_classes="pvl-shell"):
            gr.Markdown(f"# {APP_TITLE}\n{APP_DESCRIPTION}")

            with gr.Row():
                with gr.Column(scale=1, elem_classes="pvl-panel", elem_id="left-config"):
                    gr.Markdown("## Configuration")

                    api_key = gr.Textbox(
                        label="Gemini API Key",
                        placeholder="Paste API key",
                        type="password",
                    )
                    model = gr.Dropdown(
                        label="Model",
                        value=MODEL_ID,
                        choices=[MODEL_ID],
                        interactive=False,
                    )

                    gr.Markdown("### Main Agent")
                    main_name = gr.Textbox(label="Agent Name", value="Coordinator")
                    main_role = gr.Textbox(label="Agent Role", value="Task planner")
                    main_instruction = gr.Textbox(
                        label="Agent Instruction",
                        lines=3,
                        value="Break tasks into subtasks and delegate them to the best sub-agent.",
                    )

                    gr.Markdown("### Sub-Agents (1–3)")
                    with gr.Accordion("Sub-agent 1", open=True):
                        sub1_enabled = gr.Checkbox(label="Enable", value=True)
                        sub1_name = gr.Textbox(label="Name", value="Researcher")
                        sub1_spec = gr.Textbox(label="Specialization", value="information gathering")
                        sub1_instruction = gr.Textbox(
                            label="Instruction",
                            lines=2,
                            value="Search and summarize factual information.",
                        )
                    with gr.Accordion("Sub-agent 2", open=True):
                        sub2_enabled = gr.Checkbox(label="Enable", value=True)
                        sub2_name = gr.Textbox(label="Name", value="Writer")
                        sub2_spec = gr.Textbox(label="Specialization", value="writing and synthesis")
                        sub2_instruction = gr.Textbox(
                            label="Instruction",
                            lines=2,
                            value="Draft clear and structured responses from findings.",
                        )
                    with gr.Accordion("Sub-agent 3", open=False):
                        sub3_enabled = gr.Checkbox(label="Enable", value=False)
                        sub3_name = gr.Textbox(label="Name", value="Critic")
                        sub3_spec = gr.Textbox(label="Specialization", value="quality review")
                        sub3_instruction = gr.Textbox(
                            label="Instruction",
                            lines=2,
                            value="Review responses for clarity, accuracy, and completeness.",
                        )

                    gr.Markdown("### Tools")
                    slot_labels = [label for label, _ in AGENT_SLOT_CHOICES]
                    with gr.Accordion(AVAILABLE_TOOLS[0]["name"], open=False):
                        tool1_enabled = gr.Checkbox(label="Enable", value=True)
                        tool1_name = gr.Textbox(label="Tool Name", value=AVAILABLE_TOOLS[0]["name"])
                        tool1_assigned = gr.CheckboxGroup(
                            label="Assign To",
                            choices=slot_labels,
                            value=["Sub-agent 1"],
                        )
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
                        tool3_assigned = gr.CheckboxGroup(
                            label="Assign To",
                            choices=slot_labels,
                            value=["Sub-agent 2"],
                        )
                    with gr.Accordion(AVAILABLE_TOOLS[3]["name"], open=False):
                        tool4_enabled = gr.Checkbox(label="Enable", value=False)
                        tool4_name = gr.Textbox(label="Tool Name", value=AVAILABLE_TOOLS[3]["name"])
                        tool4_assigned = gr.CheckboxGroup(
                            label="Assign To",
                            choices=slot_labels,
                            value=["Sub-agent 2"],
                        )
                    with gr.Accordion(AVAILABLE_TOOLS[4]["name"], open=False):
                        tool5_enabled = gr.Checkbox(label="Enable", value=False)
                        tool5_name = gr.Textbox(label="Tool Name", value=AVAILABLE_TOOLS[4]["name"])
                        tool5_assigned = gr.CheckboxGroup(
                            label="Assign To",
                            choices=slot_labels,
                            value=["Main Agent"],
                        )

                    task = gr.Textbox(
                        label="Task",
                        lines=5,
                        placeholder="Example: Explain retrieval augmented generation.",
                    )
                    gr.Examples(
                        examples=[[t] for t in EXAMPLE_TASKS],
                        inputs=[task],
                        label="Example Tasks",
                    )

                    run_button = gr.Button("Run Multi-Agent Workflow", variant="primary")

                with gr.Column(scale=1, elem_classes="pvl-panel"):
                    gr.Markdown("## Execution")
                    status = gr.HTML("Status: idle")
                    architecture = gr.Markdown("### Agent Architecture\nWaiting for execution.")
                    plan = gr.Markdown("### Agent Plan\nWaiting for execution.")
                    trace = gr.Markdown("### Execution Trace\nWaiting for execution.")
                    sources = gr.Markdown("### Sources Used\nWaiting for execution.")
                    final_answer = gr.Markdown("### Final Answer\nWaiting for execution.")

            run_button.click(
                fn=run_orchestration,
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
