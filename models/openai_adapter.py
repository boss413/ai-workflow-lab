"""
models/openai_adapter.py

OpenAI model adapter. Implements BaseModel using the OpenAI Python SDK.
API key is read from the OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from models.base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table — USD per 1 000 tokens (input, output)
# Update as OpenAI pricing changes.
# ---------------------------------------------------------------------------

_COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4.1":            {"input": 0.002,   "output": 0.008},
    "gpt-4.1-mini":       {"input": 0.0004,  "output": 0.0016},
    "gpt-4o":             {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":        {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":        {"input": 0.01,    "output": 0.03},
    "gpt-4":              {"input": 0.03,    "output": 0.06},
    "gpt-3.5-turbo":      {"input": 0.0005,  "output": 0.0015},
}

_DEFAULT_COST: dict[str, float] = {"input": 0.005, "output": 0.015}

_ENV_KEY = "OPENAI_API_KEY"


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a single API call."""
    rates = _COST_PER_1K.get(model, _DEFAULT_COST)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000


class OpenAIAdapter(BaseModel):
    """
    Adapter for OpenAI chat completion models.

    Args:
        model_name: OpenAI model identifier (e.g. 'gpt-4o', 'gpt-4.1-mini').
        temperature: Sampling temperature passed to the API.
        max_tokens: Maximum output tokens. None lets the API use its default.
        api_key: Optional API key override. Defaults to OPENAI_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int | None = 1024,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get(_ENV_KEY)

        if not self._api_key:
            raise EnvironmentError(
                f"OpenAI API key not found. Set the {_ENV_KEY!r} environment variable."
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "openai"

    def run(self, prompt: str) -> ModelResponse:
        """
        Send a prompt to the OpenAI chat completions API and return a ModelResponse.

        Args:
            prompt: Fully rendered prompt string.

        Returns:
            ModelResponse with text, token counts, cost, and latency.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If the OpenAI API call fails.
        """
        self._validate_prompt(prompt)

        client = self._build_client()
        messages = [{"role": "user", "content": prompt}]

        logger.info("Calling OpenAI model='%s'.", self._model_name)

        try:
            with self._measure_latency() as timer:
                response = client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=self.temperature,
                    **({"max_tokens": self.max_tokens} if self.max_tokens is not None else {}),
                )
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI API call failed for model '{self._model_name}': {exc}"
            ) from exc

        return self._parse_response(response, timer.elapsed)

    def _build_client(self) -> Any:
        """Instantiate the OpenAI client. Separated for easy mocking in tests."""
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' library is required. Install it with: pip install openai"
            ) from exc
        return OpenAI(api_key=self._api_key)

    def _parse_response(self, response: Any, latency: float) -> ModelResponse:
        """Extract fields from an OpenAI ChatCompletion response object."""
        try:
            text = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = int(usage.prompt_tokens)
            output_tokens = int(usage.completion_tokens)
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected OpenAI response structure: {exc}"
            ) from exc

        cost = _calculate_cost(self._model_name, input_tokens, output_tokens)

        logger.info(
            "OpenAI model='%s' tokens=(%d in, %d out) cost=$%.6f latency=%.3fs.",
            self._model_name, input_tokens, output_tokens, cost, latency,
        )

        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency=latency,
        )