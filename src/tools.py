"""Safe and simple tool implementations."""

from __future__ import annotations

import ast
import operator as op
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import quote

import requests

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "have",
    "your",
    "their",
    "they",
    "will",
    "were",
    "been",
    "what",
    "when",
    "where",
    "which",
    "also",
    "than",
}


@dataclass
class ToolResult:
    """Standardized result format for tools."""

    tool_name: str
    output: str
    sources: list[str]


def wikipedia_search(query: str, limit: int = 3) -> ToolResult:
    """Search Wikipedia and return top snippets."""
    text = (query or "").strip()
    if not text:
        return ToolResult("Wikipedia Search", "No query provided.", [])

    endpoint = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&list=search&srsearch={quote(text)}&utf8=1&format=json&srlimit={limit}"
    )
    try:
        resp = requests.get(endpoint, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        items = payload.get("query", {}).get("search", [])
    except Exception as exc:  # pylint: disable=broad-except
        return ToolResult("Wikipedia Search", f"Tool error: {exc}", [])

    if not items:
        return ToolResult("Wikipedia Search", "No results found.", [])

    lines = []
    sources = []
    for item in items:
        title = item.get("title", "Unknown")
        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip()
        page_url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        lines.append(f"- {title}: {snippet}")
        sources.append(page_url)

    return ToolResult("Wikipedia Search", "\n".join(lines), sources)


def keyword_extractor(text: str, top_k: int = 10) -> ToolResult:
    """Extract top keywords by frequency."""
    tokens = re.findall(r"[a-zA-Z']+", (text or "").lower())
    words = [w for w in tokens if len(w) > 2 and w not in STOPWORDS]
    if not words:
        return ToolResult("Keyword Extractor", "No keywords detected.", [])

    counts = Counter(words).most_common(top_k)
    result = ", ".join(f"{word}({count})" for word, count in counts)
    return ToolResult("Keyword Extractor", result, [])


def text_summarizer(text: str, max_sentences: int = 3) -> ToolResult:
    """Summarize text by selecting first N sentences."""
    raw = (text or "").strip()
    if not raw:
        return ToolResult("Text Summarizer", "No text to summarize.", [])

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
    if not sentences:
        return ToolResult("Text Summarizer", raw[:300], [])

    summary = " ".join(sentences[:max_sentences])
    return ToolResult("Text Summarizer", summary, [])


def word_counter(text: str) -> ToolResult:
    """Count words, characters, and estimated reading time."""
    raw = text or ""
    words = re.findall(r"\S+", raw)
    word_count = len(words)
    char_count = len(raw)
    read_time = round(word_count / 200, 2)
    out = (
        f"Words: {word_count}\n"
        f"Characters: {char_count}\n"
        f"Estimated reading time: {read_time} min"
    )
    return ToolResult("Word Counter", out, [])


_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval(node):  # noqa: ANN001
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression.")


def calculator(expression: str) -> ToolResult:
    """Safely evaluate basic arithmetic expression."""
    expr = (expression or "").strip()
    if not expr:
        return ToolResult("Calculator", "No expression provided.", [])
    try:
        parsed = ast.parse(expr, mode="eval").body
        value = _safe_eval(parsed)
        return ToolResult("Calculator", f"{expr} = {value}", [])
    except Exception as exc:  # pylint: disable=broad-except
        return ToolResult("Calculator", f"Invalid expression: {exc}", [])


TOOL_REGISTRY = {
    "wikipedia_search": wikipedia_search,
    "keyword_extractor": keyword_extractor,
    "text_summarizer": text_summarizer,
    "word_counter": word_counter,
    "calculator": calculator,
}
