"""Safe DeepSeek client wrapper for future agent workflows.

This module intentionally starts small. It centralizes credential handling and
request-shape defaults before agent execution logic is added elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import time
from typing import Any
from urllib import error, request

from project.agent_settings import DeepSeekSettings, load_deepseek_settings


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


class DeepSeekClient:
    """Minimal DeepSeek chat-completions client."""

    def __init__(self, settings: DeepSeekSettings | None = None):
        self.settings = settings or load_deepseek_settings()

    @property
    def configured(self) -> bool:
        return self.settings.configured

    def require_configured(self) -> None:
        self.settings.require_api_key()

    def build_chat_payload(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.1,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.require_configured()
        payload: dict[str, Any] = {
            "model": model or self.settings.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        return payload

    def complete_chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.1,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.build_chat_payload(
            messages,
            model=model,
            response_format=response_format,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
        body = json.dumps(payload).encode("utf-8")
        endpoint = self.settings.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            req = request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.settings.timeout_seconds) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (
                error.HTTPError,
                error.URLError,
                TimeoutError,
                socket.timeout,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(
            "DeepSeek chat completion failed; check network access, base URL, "
            "model name, and DEEPSEEK_API_KEY."
        ) from last_error

    def complete_chat_json(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        completion = self.complete_chat(
            messages,
            model=model,
            response_format=response_format,
            temperature=temperature,
        )
        try:
            content = completion["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek response did not contain message content.") from exc
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("DeepSeek response content was not valid JSON.") from exc

    def complete_chat_tool_calls(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = "required",
        temperature: float = 0.0,
    ) -> list[ChatToolCall]:
        completion = self.complete_chat(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        try:
            message = completion["choices"][0]["message"]
            raw_calls = message.get("tool_calls") or []
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek response did not contain tool_calls.") from exc

        calls: list[ChatToolCall] = []
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function") or {}
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            raw_arguments = function.get("arguments")
            try:
                arguments = json.loads(raw_arguments or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"DeepSeek tool call arguments for {name} were not valid JSON."
                ) from exc
            if not isinstance(arguments, dict):
                raise RuntimeError(
                    f"DeepSeek tool call arguments for {name} must decode to an object."
                )
            calls.append(
                ChatToolCall(
                    id=str(item.get("id") or ""),
                    name=name,
                    arguments=arguments,
                )
            )

        if not calls:
            raise RuntimeError("DeepSeek response did not contain usable tool calls.")
        return calls
