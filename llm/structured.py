"""
llm/structured.py — Validated structured output with bounded repair.

The model is asked for JSON matching a pydantic schema (natively constrained
on Ollama). Output is always parsed and validated here; on failure the model
gets one repair attempt. If validation still fails the caller receives
`value=None` and must take an explicit fallback path — a failed parse is
never treated as a valid decision.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from llm.base import ChatModel, LLMError, LLMResult, Message

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class StructuredResult(Generic[T]):
    value: T | None
    attempts: int
    error: str | None = None
    raw_outputs: list[str] = field(default_factory=list)
    llm_results: list[LLMResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.value is not None


def extract_json(text: str) -> str:
    """Return the first top-level JSON object in `text` (tolerates fences/prose)."""
    cleaned = _FENCE_RE.sub("", text.strip())
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start: i + 1]
    raise ValueError("unterminated JSON object")


def parse_model(text: str, schema: type[T]) -> T:
    return schema.model_validate(json.loads(extract_json(text)))


def complete_structured(
    model: ChatModel,
    messages: list[Message],
    schema: type[T],
    *,
    repair_attempts: int = 1,
    timeout_s: float | None = None,
) -> StructuredResult[T]:
    json_schema = schema.model_json_schema()
    convo: list[Message] = list(messages)
    result: StructuredResult[T] = StructuredResult(value=None, attempts=0)

    for attempt in range(1 + repair_attempts):
        result.attempts = attempt + 1
        try:
            llm_result = model.complete(convo, json_schema=json_schema, timeout_s=timeout_s)
        except LLMError as e:
            result.error = f"{type(e).__name__}: {e}"
            return result  # backend failure: do not retry blindly
        result.llm_results.append(llm_result)
        result.raw_outputs.append(llm_result.text)
        try:
            result.value = parse_model(llm_result.text, schema)
            result.error = None
            return result
        except (ValueError, ValidationError) as e:
            result.error = f"invalid structured output: {str(e)[:300]}"
            convo = convo + [
                {"role": "assistant", "content": llm_result.text},
                {
                    "role": "user",
                    "content": (
                        "That response was not valid JSON for the required schema. "
                        f"Error: {str(e)[:300]}\n"
                        "Reply with ONLY a JSON object matching this schema:\n"
                        f"{json.dumps(json_schema)}"
                    ),
                },
            ]
    return result
