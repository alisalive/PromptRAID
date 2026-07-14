"""Sends a payload to a target LLM agent through a pluggable provider backend.

Implements a basic tool-use loop so indirect injection can be simulated: a
tool's handler can return content that itself embeds an injected payload
(e.g. a "read_webpage" or "search" tool whose output contains the attack
string), letting the agent ingest the injection indirectly rather than
receiving it as the direct user message.

Backed by any `BaseProvider` implementation from `providers.py` (Anthropic,
Gemini, Groq, OpenRouter). Requires a live API key to run; not covered by the
automated test suite (see tests/ for the mocked pieces).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from promptraid.harness.providers import AnthropicProvider, BaseProvider

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
    """Runs a payload against a target model, through any `BaseProvider`, via a
    tool-use loop."""

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[ToolDefinition]] = None,
        max_turns: int = 6,
    ):
        # Preserves the original Anthropic-only constructor path when no
        # provider is supplied.
        self.provider = provider or AnthropicProvider(model=model, api_key=api_key)
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
        history: List[Dict[str, Any]] = [{"role": "user", "content": user_payload}]
        transcript.messages = history

        tool_schema = self._tool_schema() if self.tools else None

        response = None
        for _ in range(self.max_turns):
            response = self.provider.send(history, tools=tool_schema, system_prompt=self.system_prompt)

            if not response.tool_calls:
                transcript.final_text = response.text
                history.append({"role": "assistant", "text": response.text, "tool_calls": []})
                break

            history.append(
                {
                    "role": "assistant",
                    "text": response.text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "input": tc.input} for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                tool = self._find_tool(tc.name)
                output = tool.handler(tc.input) if tool else f"Error: unknown tool {tc.name}"
                transcript.tool_calls.append({"name": tc.name, "input": tc.input, "output": output})
                history.append(
                    {"role": "tool_result", "tool_call_id": tc.id, "name": tc.name, "content": output}
                )
        else:
            if response is not None and not transcript.final_text:
                transcript.final_text = response.text

        return transcript
