"""
Uniform LLM interface for Anthropic, OpenAI, Google Gemini, and Ollama.

All providers are called through a single `judge()` function that returns a
`JudgeResponse` with token counts, latency, cost, and optionally a parsed
Pydantic object when a schema is supplied.
"""

import time
import os
import json
from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai
import ollama
from google import genai
from google.genai import types as genai_types


# Model prices per 1M tokens: (price_in, price_out)
PRICES: dict[str, tuple[float, float]] = {
    "o1":                        (15.00, 60.00),
    "claude-sonnet-4-6":         ( 3.00, 15.00),
    "claude-haiku-4-5":          ( 1.00,  1.25),
    "gemini-2.5-pro":            ( 2.50, 10.00),
    "gpt-4.1":                   ( 2.00,  8.00),
    "o3-mini":                   ( 1.10,  4.40),
    "gpt-5":                     ( 1.25, 10.00),
    "grok-4.3":                  ( 1.25,  2.50),
}


@dataclass
class JudgeResponse:
    """Holds the raw and parsed output of a single LLM call, plus usage stats."""

    raw_text: str
    model_id: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    parsed: Any = None       # populated when a Pydantic schema is passed to judge()
    cost_usd: float = field(init=False)

    def __post_init__(self):
        price_in, price_out = PRICES.get(self.model_id, (0.0, 0.0))
        self.cost_usd = (
            self.tokens_in * price_in
            + self.tokens_out * price_out
        ) / 1_000_000


def _call_anthropic(model: str, system_prompt: str, user_prompt: str, schema: type | None = None) -> JudgeResponse:
    """Call the Anthropic API. If schema is given, appends JSON instructions to the system prompt."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sys = system_prompt
    if schema is not None:
        sys += f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema.model_json_schema(), indent=2)}"
    t0 = time.perf_counter()
    response = client.messages.create(
        model=model,
        system=sys,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency = time.perf_counter() - t0
    raw = response.content[0].text
    usage = response.usage
    return JudgeResponse(
        raw_text=raw,
        model_id=model,
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
        latency_s=latency,
        parsed=schema.model_validate_json(raw) if schema is not None else None,
    )


def _call_openai(model: str, system_prompt: str, user_prompt: str, schema: type | None = None) -> JudgeResponse:
    """Call the OpenAI API. Uses the structured-output beta endpoint when a schema is given."""
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    t0 = time.perf_counter()
    if schema is not None:
        response = client.beta.chat.completions.parse(
            model=model, messages=messages, response_format=schema,
        )
        latency = time.perf_counter() - t0
        choice = response.choices[0].message
        return JudgeResponse(
            raw_text=choice.content or "",
            model_id=model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_s=latency,
            parsed=choice.parsed,
        )
    response = client.chat.completions.create(model=model, messages=messages)
    latency = time.perf_counter() - t0
    return JudgeResponse(
        raw_text=response.choices[0].message.content,
        model_id=model,
        tokens_in=response.usage.prompt_tokens,
        tokens_out=response.usage.completion_tokens,
        latency_s=latency,
    )


def _call_gemini(model: str, system_prompt: str, user_prompt: str, schema: type | None = None) -> JudgeResponse:
    """Call the Google Gemini API. Enables native JSON mode when a schema is given."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json" if schema is not None else None,
        response_schema=schema if schema is not None else None,
    )
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=config,
    )
    latency = time.perf_counter() - t0
    raw = response.text
    usage = response.usage_metadata
    return JudgeResponse(
        raw_text=raw,
        model_id=model,
        tokens_in=usage.prompt_token_count or 0,
        tokens_out=usage.candidates_token_count or 0,
        latency_s=latency,
        parsed=schema.model_validate_json(raw) if schema is not None else None,
    )


def _call_grok(model: str, system_prompt: str, user_prompt: str, schema: type | None = None) -> JudgeResponse:
    """Call the xAI Grok API (OpenAI-compatible). Uses structured output when a schema is given."""
    client = openai.OpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    t0 = time.perf_counter()
    if schema is not None:
        response = client.beta.chat.completions.parse(
            model=model, messages=messages, response_format=schema,
        )
        latency = time.perf_counter() - t0
        choice = response.choices[0].message
        return JudgeResponse(
            raw_text=choice.content or "",
            model_id=model,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_s=latency,
            parsed=choice.parsed,
        )
    response = client.chat.completions.create(model=model, messages=messages)
    latency = time.perf_counter() - t0
    return JudgeResponse(
        raw_text=response.choices[0].message.content,
        model_id=model,
        tokens_in=response.usage.prompt_tokens,
        tokens_out=response.usage.completion_tokens,
        latency_s=latency,
    )


def _call_ollama(model: str, system_prompt: str, user_prompt: str, schema: type | None = None) -> JudgeResponse:
    """Call a local Ollama model. Passes the JSON schema as a format constraint when given."""
    host = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    client = ollama.Client(host=host)
    t0 = time.perf_counter()
    response = client.chat(
        model=model,
        format=schema.model_json_schema() if schema is not None else None,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    latency = time.perf_counter() - t0
    raw = response.message.content
    return JudgeResponse(
        raw_text=raw,
        model_id=model,
        tokens_in=response.prompt_eval_count or 0,
        tokens_out=response.eval_count or 0,
        latency_s=latency,
        parsed=schema.model_validate_json(raw) if schema is not None else None,
    )


def judge(model: str, system_prompt: str, user_prompt: str, schema: type | None = None) -> JudgeResponse:
    """Route a call to the right provider based on the model name prefix.

    Pass a Pydantic BaseModel as `schema` to get structured output in `response.parsed`.
    Falls back to Ollama for any model name that doesn't match a known prefix.
    """
    if model.startswith("claude"):
        return _call_anthropic(model, system_prompt, user_prompt, schema)
    elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return _call_openai(model, system_prompt, user_prompt, schema)
    elif model.startswith("gemini"):
        return _call_gemini(model, system_prompt, user_prompt, schema)
    elif model.startswith("grok"):
        return _call_grok(model, system_prompt, user_prompt, schema)
    else:
        return _call_ollama(model, system_prompt, user_prompt, schema)
