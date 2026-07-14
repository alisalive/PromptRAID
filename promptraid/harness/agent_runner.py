"""Sends a payload to a target LLM agent via the Anthropic API.

Implements a basic tool-use loop so indirect injection can be simulated: a
tool's handler can return content that itself embeds an injected payload
(e.g. a "read_webpage" or "search" tool whose output contains the attack
string), letting the agent ingest the injection indirectly rather than
receiving it as the direct user message.

Reads ANTHROPIC_API_KEY from the environment. Requires a live API key to run;
not covered by the automated test suite (see tests/ for the mocked pieces).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only when dependency missing
    anthropic = None


DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class ToolDefinition:
    """A tool exposed to the target agent during a run."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], str]


@dataclass
class RunTranscript:
    """Full record of a single agent run."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "messages": self.messages,
            "final_text": self.final_text,
            "tool_calls": self.tool_calls,
        }


class AgentRunner:
    """Runs a payload against a target Anthropic model through a tool-use loop."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[ToolDefinition]] = None,
        max_turns: int = 6,
    ):
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it in your environment before "
                "running live agent tests."
            )
        if anthropic is None:
            raise RuntimeError(
                "The 'anthropic' package is not installed. Run `pip install anthropic`."
            )
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_turns = max_turns

    def _tool_schema(self) -> List[Dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in self.tools
        ]

    def _find_tool(self, name: str) -> Optional[ToolDefinition]:
        return next((t for t in self.tools if t.name == name), None)

    def run(self, user_payload: str) -> RunTranscript:
        """Send `user_payload` as the initial user message and run the tool-use loop
        until the model stops calling tools or `max_turns` is reached."""
        transcript = RunTranscript()
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_payload}]
        transcript.messages = messages

        response = None
        for _ in range(self.max_turns):
            kwargs: Dict[str, Any] = dict(
                model=self.model,
                max_tokens=1024,
                system=self.system_prompt,
                messages=messages,
            )
            if self.tools:
                kwargs["tools"] = self._tool_schema()

            response = self.client.messages.create(**kwargs)
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                text_blocks = [b.text for b in response.content if b.type == "text"]
                transcript.final_text = "\n".join(text_blocks)
                break

            tool_results = []
            for block in tool_use_blocks:
                tool = self._find_tool(block.name)
                output = (
                    tool.handler(block.input) if tool else f"Error: unknown tool {block.name}"
                )
                transcript.tool_calls.append(
                    {"name": block.name, "input": block.input, "output": output}
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
            messages.append({"role": "user", "content": tool_results})
        else:
            if response is not None:
                text_blocks = [
                    b.text for b in getattr(response, "content", []) if getattr(b, "type", None) == "text"
                ]
                if text_blocks:
                    transcript.final_text = "\n".join(text_blocks)

        return transcript
