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

# ---------------------------------------------------------------------------
# Cost table — USD per 1 000 tokens (input, output)
# ---------------------------------------------------------------------------

_COST_PER_1K: dict[str, dict[str, float]] = {
    "claude-opus-4-5":          {"input": 0.015,   "output": 0.075},
    "claude-sonnet-4-5":        {"input": 0.003,   "output": 0.015},
    "claude-haiku-4-5":         {"input": 0.0008,  "output": 0.004},
    "claude-opus-4":            {"input": 0.015,   "output": 0.075},
    "claude-sonnet-4":          {"input": 0.003,   "output": 0.015},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022":  {"input": 0.0008,"output": 0.004},
    "claude-3-opus-20240229":   {"input": 0.015,   "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003,   "output": 0.015},
    "claude-3-haiku-20240307":  {"input": 0.00025, "output": 0.00125},
}

_DEFAULT_COST: dict[str, float] = {"input": 0.003, "output": 0.015}

_ENV_KEY = "ANTHROPIC_API_KEY"


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a single Anthropic API call."""
    rates = _COST_PER_1K.get(model, _DEFAULT_COST)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000


class AnthropicAdapter(BaseModel):
    """
    Adapter for Anthropic Claude models.

    Args:
        model_name: Anthropic model identifier (e.g. 'claude-sonnet-4-5').
        temperature: Sampling temperature passed to the API (0.0–1.0).
        max_tokens: Maximum output tokens (required by the Anthropic API).
        api_key: Optional API key override. Defaults to ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
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

    def run(self, prompt: str) -> ModelResponse:
        """
        Send a prompt to the Anthropic messages API and return a ModelResponse.

        Args:
            prompt: Fully rendered prompt string.

        Returns:
            ModelResponse with text, token counts, cost, and latency.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If the Anthropic API call fails.
        """
        self._validate_prompt(prompt)

        client = self._build_client()

        logger.info("Calling Anthropic model='%s'.", self._model_name)

        try:
            with self._measure_latency() as timer:
                response = client.messages.create(
                    model=self._model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
        except Exception as exc:
            raise RuntimeError(
                f"Anthropic API call failed for model '{self._model_name}': {exc}"
            ) from exc

        return self._parse_response(response, timer.elapsed)

    def _build_client(self) -> Any:
        """Instantiate the Anthropic client. Separated for easy mocking in tests."""
        try:
            from anthropic import Anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' library is required. Install it with: pip install anthropic"
            ) from exc
        return Anthropic(api_key=self._api_key)

    def _parse_response(self, response: Any, latency: float) -> ModelResponse:
        """Extract fields from an Anthropic Message response object."""
        try:
            text = response.content[0].text if response.content else ""
            usage = response.usage
            input_tokens = int(usage.input_tokens)
            output_tokens = int(usage.output_tokens)
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Anthropic response structure: {exc}"
            ) from exc

        cost = _calculate_cost(self._model_name, input_tokens, output_tokens)

        logger.info(
            "Anthropic model='%s' tokens=(%d in, %d out) cost=$%.6f latency=%.3fs.",
            self._model_name, input_tokens, output_tokens, cost, latency,
        )

        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency=latency,
        )