"""Central configuration for Multi-Agent Orchestrator."""

APP_TITLE = "Multi-Agent Orchestrator for LLM Workflows"
APP_DESCRIPTION = (
    "Configure a coordinator, up to three sub-agents, and safe tools. "
    "Run a task and inspect visible multi-agent orchestration with traceable steps."
)

MODEL_ID = "gemini-3.1-flash-lite-preview"
MAX_SUBAGENTS = 3
MAX_SUBTASKS = 6

AGENT_SLOT_CHOICES = [
    ("Main Agent", "main"),
    ("Sub-agent 1", "sub1"),
    ("Sub-agent 2", "sub2"),
    ("Sub-agent 3", "sub3"),
]

AVAILABLE_TOOLS = [
    {
        "id": "wikipedia_search",
        "name": "Wikipedia Search",
        "description": "Find short factual context from Wikipedia search results.",
    },
    {
        "id": "keyword_extractor",
        "name": "Keyword Extractor",
        "description": "Extract top keywords from text using frequency.",
    },
    {
        "id": "text_summarizer",
        "name": "Text Summarizer",
        "description": "Summarize long text into a few sentences.",
    },
    {
        "id": "word_counter",
        "name": "Word Counter",
        "description": "Count words, characters, and estimated reading time.",
    },
    {
        "id": "calculator",
        "name": "Calculator",
        "description": "Evaluate simple arithmetic expressions safely.",
    },
]

EXAMPLE_TASKS = [
    "Explain the EU AI Act and its impact on startups.",
    "Compare transformers and RNNs for sequence modeling.",
    "Write a short LinkedIn post about retrieval-augmented generation.",
]
