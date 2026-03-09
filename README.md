# Multi-Agent Orchestrator for LLM Workflows

A Gradio app that lets users configure a coordinator agent, sub-agents, and safe tools, then run an orchestrated workflow with visible execution trace.

## Key Capabilities

- Main coordinator agent for planning and synthesis
- Up to 3 configurable sub-agents
- Safe built-in tools (no user Python execution)
- Step-by-step execution trace
- Structured output sections:
  - Agent architecture
  - Agent plan
  - Execution trace
  - Sources used
  - Final answer

## Stack

- Python
- Gradio
- LangChain
- Gemini (`gemini-3.1-flash-lite-preview`)

## Project Structure

```text
multi-agent-orchestrator/
├── app.py
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── config.py
    ├── agent_builder.py
    ├── tool_builder.py
    ├── orchestrator.py
    ├── gemini_client.py
    └── tools.py
```

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL shown in terminal.

## API Key Handling

- Gemini API key is entered in the app UI.
- Key is used only for runtime requests.
- Key is not saved to disk by this implementation.
