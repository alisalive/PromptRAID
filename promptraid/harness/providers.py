"""Unified LLM provider interface so AgentRunner can target multiple backends
(Anthropic, Gemini, Groq, OpenRouter) through a single `--target` string like
"anthropic:claude-sonnet-4-6" or "groq:llama-3.3-70b-versatile".

Each provider translates a generic conversation history into its native
request format and normalizes the response back into a `ProviderResponse`,
so `AgentRunner` never needs to know which backend it is talking to.

Generic message history format (list of dicts) used by `BaseProvider.send`:
  - {"role": "user", "content": "text"}
  - {"role": "assistant", "text": "...", "tool_calls": [{"id", "name", "input"}]}
  - {"role": "tool_result", "tool_call_id": "...", "name": "...", "content": "..."}

Generic tool schema (list of dicts) passed as `tools=`:
  - {"name": "...", "description": "...", "input_schema": {...JSON schema...}}
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import anthropic
except ImportError:  # pragma: no cover - exercised only when dependency missing
    anthropic = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only when dependency missing
    OpenAI = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised only when dependency missing
    genai = None
    genai_types = None


@dataclass
class ToolCall:
    """A normalized tool call request emitted by a target model."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ProviderResponse:
    """Normalized response from any provider's `send` call."""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None


class BaseProvider(ABC):
    """Common interface every target-model backend must implement."""

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def send(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> ProviderResponse:
        """Send the full generic conversation history and return a normalized response."""

    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider/model combination can be given tool schemas."""


class AnthropicProvider(BaseProvider):
    """Targets Anthropic models directly via the `anthropic` SDK."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
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

    def supports_tools(self) -> bool:
        return True

    def send(self, messages, tools=None, system_prompt=None) -> ProviderResponse:
        native_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "user":
                native_messages.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                content: List[Dict[str, Any]] = []
                if m.get("text"):
                    content.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls", []):
                    content.append(
                        {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
                    )
                native_messages.append({"role": "assistant", "content": content})
            elif role == "tool_result":
                native_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m["tool_call_id"],
                                "content": m["content"],
                            }
                        ],
                    }
                )

        kwargs: Dict[str, Any] = dict(model=self.model, max_tokens=1024, messages=native_messages)
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
                for t in tools
            ]

        response = self.client.messages.create(**kwargs)
        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input) for b in response.content if b.type == "tool_use"
        ]
        return ProviderResponse(text="\n".join(text_blocks), tool_calls=tool_calls, raw=response)


class OpenAICompatProvider(BaseProvider):
    """Shared base for OpenAI-compatible chat-completions APIs (Groq, OpenRouter)."""

    def __init__(self, model: str, api_key: str, base_url: str):
        super().__init__(model)
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. Run `pip install openai`."
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def supports_tools(self) -> bool:
        return True

    def send(self, messages, tools=None, system_prompt=None) -> ProviderResponse:
        native_messages: List[Dict[str, Any]] = []
        if system_prompt:
            native_messages.append({"role": "system", "content": system_prompt})

        for m in messages:
            role = m["role"]
            if role == "user":
                native_messages.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                msg: Dict[str, Any] = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])},
                        }
                        for tc in m["tool_calls"]
                    ]
                native_messages.append(msg)
            elif role == "tool_result":
                native_messages.append(
                    {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
                )

        kwargs: Dict[str, Any] = dict(model=self.model, messages=native_messages)
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]

        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0].message
        tool_calls = []
        if getattr(choice, "tool_calls", None):
            for tc in choice.tool_calls:
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments))
                )
        return ProviderResponse(text=choice.content or "", tool_calls=tool_calls, raw=response)


class GroqProvider(OpenAICompatProvider):
    """Targets Groq's OpenAI-compatible API."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it in your environment before running "
                "live agent tests."
            )
        super().__init__(model, api_key=resolved_key, base_url="https://api.groq.com/openai/v1")


class OpenRouterProvider(OpenAICompatProvider):
    """Targets OpenRouter's OpenAI-compatible API."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it in your environment before "
                "running live agent tests."
            )
        super().__init__(model, api_key=resolved_key, base_url="https://openrouter.ai/api/v1")


class GeminiProvider(BaseProvider):
    """Targets Google AI Studio's Gemini models via the `google-genai` SDK."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it in your environment before running "
                "live agent tests."
            )
        if genai is None:
            raise RuntimeError(
                "The 'google-genai' package is not installed. Run `pip install google-genai`."
            )
        self.client = genai.Client(api_key=resolved_key)

    def supports_tools(self) -> bool:
        return True

    def send(self, messages, tools=None, system_prompt=None) -> ProviderResponse:
        contents = []
        for m in messages:
            role = m["role"]
            if role == "user":
                contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=m["content"])]))
            elif role == "assistant":
                parts = []
                if m.get("text"):
                    parts.append(genai_types.Part(text=m["text"]))
                for tc in m.get("tool_calls", []):
                    parts.append(
                        genai_types.Part(function_call=genai_types.FunctionCall(name=tc["name"], args=tc["input"]))
                    )
                contents.append(genai_types.Content(role="model", parts=parts))
            elif role == "tool_result":
                contents.append(
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                function_response=genai_types.FunctionResponse(
                                    name=m.get("name", ""), response={"result": m["content"]}
                                )
                            )
                        ],
                    )
                )

        config_kwargs: Dict[str, Any] = {}
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
        if tools:
            declarations = [
                genai_types.FunctionDeclaration(
                    name=t["name"], description=t["description"], parameters=t["input_schema"]
                )
                for t in tools
            ]
            config_kwargs["tools"] = [genai_types.Tool(function_declarations=declarations)]
        config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        response = self.client.models.generate_content(model=self.model, contents=contents, config=config)

        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        candidate = response.candidates[0] if getattr(response, "candidates", None) else None
        if candidate and candidate.content and candidate.content.parts:
            for i, part in enumerate(candidate.content.parts):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc:
                    tool_calls.append(ToolCall(id=f"{fc.name}_{i}", name=fc.name, input=dict(fc.args or {})))

        return ProviderResponse(text="\n".join(text_parts), tool_calls=tool_calls, raw=response)


_PROVIDER_FACTORIES = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
}


def get_provider(target: str) -> BaseProvider:
    """Parse a "provider:model" string (e.g. "groq:llama-3.3-70b-versatile") and
    return a configured provider instance."""
    if ":" not in target:
        raise ValueError(
            f"Invalid target {target!r}; expected format 'provider:model', "
            f"e.g. 'anthropic:claude-sonnet-4-6'."
        )
    provider_name, model = target.split(":", 1)
    provider_name = provider_name.strip().lower()
    model = model.strip()
    if not model:
        raise ValueError(f"Invalid target {target!r}; model portion is empty.")

    try:
        provider_cls = _PROVIDER_FACTORIES[provider_name]
    except KeyError:
        raise ValueError(
            f"Unknown provider {provider_name!r}. Valid providers: "
            f"{sorted(_PROVIDER_FACTORIES)}"
        ) from None

    return provider_cls(model)
