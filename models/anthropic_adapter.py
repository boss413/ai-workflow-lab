"""
models/anthropic_adapter.py

Anthropic model adapter. Implements BaseModel using the Anthropic Python SDK.
API key is read from the ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from models.base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)

_COST_PER_1K: dict[str, dict[str, float]] = {
    "claude-opus-4-5":            {"input": 0.015,   "output": 0.075},
    "claude-sonnet-4-5":          {"input": 0.003,   "output": 0.015},
    "claude-haiku-4-5":           {"input": 0.0008,  "output": 0.004},
    "claude-opus-4":              {"input": 0.015,   "output": 0.075},
    "claude-sonnet-4":            {"input": 0.003,   "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003,   "output": 0.015},
    "claude-3-5-haiku-20241022":  {"input": 0.0008,  "output": 0.004},
    "claude-3-opus-20240229":     {"input": 0.015,   "output": 0.075},
    "claude-3-sonnet-20240229":   {"input": 0.003,   "output": 0.015},
    "claude-3-haiku-20240307":    {"input": 0.00025, "output": 0.00125},
}

_DEFAULT_COST: dict[str, float] = {"input": 0.003, "output": 0.015}
_ENV_KEY = "ANTHROPIC_API_KEY"


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_PER_1K.get(model, _DEFAULT_COST)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000


class AnthropicAdapter(BaseModel):
    """
    Adapter for Anthropic Claude models.

    Args:
        model_name: Anthropic model identifier (e.g. 'claude-sonnet-4-5').
        temperature: Default sampling temperature (overridden by generation_params).
        max_tokens: Default max output tokens (overridden by generation_params).
        top_p: Default top_p (overridden by generation_params).
        api_key: Optional API key override. Defaults to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._api_key = api_key or os.environ.get(_ENV_KEY)

        if not self._api_key:
            raise EnvironmentError(
                f"Anthropic API key not found. Set the {_ENV_KEY!r} environment variable."
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "anthropic"

    def run(
        self,
        prompt: str | dict[str, str],
        generation_params: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """
        Send a prompt to the Anthropic messages API and return a ModelResponse.

        Args:
            prompt: A plain string (user message) or normalized schema:
                    {"system": "...", "user": "..."}
            generation_params: Normalized params with keys temperature, top_p,
                               max_tokens. Falls back to adapter defaults.
        """
        user_text, system_text = self._parse_prompt(prompt)
        params = self._resolve_params(generation_params)
        client = self._build_client()

        logger.info("Calling Anthropic model='%s'.", self._model_name)

        api_kwargs: dict[str, Any] = dict(
            model=self._model_name,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            messages=[{"role": "user", "content": user_text}],
        )
        if system_text:
            api_kwargs["system"] = system_text

        try:
            with self._measure_latency() as timer:
                response = client.messages.create(**api_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Anthropic API call failed for model '{self._model_name}': {exc}"
            ) from exc

        return self._parse_response(response, timer.elapsed)

    def _parse_prompt(self, prompt: str | dict[str, str]) -> tuple[str, str]:
        """Return (user_text, system_text) from a string or normalized schema."""
        if isinstance(prompt, str):
            self._validate_prompt(prompt)
            return prompt, ""
        if not isinstance(prompt, dict) or "user" not in prompt:
            raise ValueError("Prompt dict must contain a 'user' key.")
        if not str(prompt["user"]).strip():
            raise ValueError("prompt['user'] must not be empty.")
        return prompt["user"], prompt.get("system", "")

    def _resolve_params(self, generation_params: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "temperature": generation_params.get("temperature", self.temperature)
                           if generation_params else self.temperature,
            "top_p": generation_params.get("top_p", self.top_p)
                     if generation_params else self.top_p,
            "max_tokens": generation_params.get("max_tokens", self.max_tokens)
                          if generation_params else self.max_tokens,
        }

    def _build_client(self) -> Any:
        try:
            from anthropic import Anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' library is required. Install it with: pip install anthropic"
            ) from exc
        return Anthropic(api_key=self._api_key)

    def _parse_response(self, response: Any, latency: float) -> ModelResponse:
        try:
            text = response.content[0].text if response.content else ""
            usage = response.usage
            input_tokens = int(usage.input_tokens)
            output_tokens = int(usage.output_tokens)
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Anthropic response structure: {exc}") from exc

        cost = _calculate_cost(self._model_name, input_tokens, output_tokens)
        logger.info(
            "Anthropic model='%s' tokens=(%d in, %d out) cost=$%.6f latency=%.3fs.",
            self._model_name, input_tokens, output_tokens, cost, latency,
        )
        return ModelResponse(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens,
            cost=cost, latency=latency,
        )