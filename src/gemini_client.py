"""Gemini + LangChain client wrapper."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiClient:
    """Lightweight text generation client with consistent prompts."""

    def __init__(self, api_key: str, model_id: str):
        key = (api_key or "").strip()
        if not key:
            raise ValueError("Gemini API key is required.")

        self._llm = ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=key,
            temperature=0.2,
        )

    def generate_text(self, system_instruction: str, user_prompt: str) -> str:
        """Generate plain text response."""
        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=user_prompt),
                ]
            )
            text = self._content_to_text(response.content if response else "")
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Gemini generation failed: {exc}") from exc

        output = (text or "").strip()
        if not output:
            raise RuntimeError("Gemini returned an empty response.")
        return output

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Normalize LangChain response content to plain text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    # Gemini-style block can include {"type":"text","text":"..."}
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                    else:
                        chunks.append(str(item))
                else:
                    chunks.append(str(item))
            return "\n".join(chunks).strip()
        if isinstance(content, dict):
            text = content.get("text")
            return text if isinstance(text, str) else str(content)
        return str(content)

    def generate_json(self, system_instruction: str, user_prompt: str) -> dict[str, Any]:
        """Generate and parse JSON response with fallback extraction."""
        raw = self.generate_text(system_instruction=system_instruction, user_prompt=user_prompt)
        block = self._extract_json_block(raw)
        try:
            return json.loads(block)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini returned invalid JSON: {exc}") from exc

    @staticmethod
    def _extract_json_block(text: str) -> str:
        # Accept fenced JSON or raw JSON.
        cleaned = text.strip()
        fence_match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        obj_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if obj_match:
            return obj_match.group(1).strip()
        return cleaned
