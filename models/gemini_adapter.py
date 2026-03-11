"""
models/gemini_adapter.py

Google Gemini model adapter. Implements BaseModel using the google-genai Python SDK.
API key is read from the GEMINI_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from models.base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table — USD per 1 000 tokens (input, output)
# Gemini pricing as of March 2026.
# ---------------------------------------------------------------------------

_COST_PER_1K: dict[str, dict[str, float]] = {
    "gemini-2.5-pro":         {"input": 0.00125, "output": 0.010},
    "gemini-2.5-flash":       {"input": 0.000075,"output": 0.0003},
    "gemini-2.0-flash":       {"input": 0.0001,  "output": 0.0004},
    "gemini-2.0-flash-lite":  {"input": 0.000075,"output": 0.0003},
    "gemini-1.5-pro":         {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash":       {"input": 0.000075,"output": 0.0003},
    "gemini-1.5-flash-8b":    {"input": 0.0000375,"output": 0.00015},
}

_DEFAULT_COST: dict[str, float] = {"input": 0.00125, "output": 0.005}

_ENV_KEY = "GEMINI_API_KEY"


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a single Gemini API call."""
    rates = _COST_PER_1K.get(model, _DEFAULT_COST)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000


class GeminiAdapter(BaseModel):
    """
    Adapter for Google Gemini models via the google-genai SDK.

    Args:
        model_name: Gemini model identifier (e.g. 'gemini-2.5-flash').
        temperature: Sampling temperature (0.0–2.0).
        max_tokens: Maximum output tokens. None uses the API default.
        api_key: Optional API key override. Defaults to GEMINI_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
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
                f"Gemini API key not found. Set the {_ENV_KEY!r} environment variable."
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "google"

    def run(self, prompt: str) -> ModelResponse:
        """
        Send a prompt to the Gemini generate_content API and return a ModelResponse.

        Args:
            prompt: Fully rendered prompt string.

        Returns:
            ModelResponse with text, token counts, cost, and latency.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If the Gemini API call fails.
        """
        self._validate_prompt(prompt)

        client = self._build_client()

        logger.info("Calling Gemini model='%s'.", self._model_name)

        generation_config = self._build_generation_config()

        try:
            with self._measure_latency() as timer:
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=generation_config,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Gemini API call failed for model '{self._model_name}': {exc}"
            ) from exc

        return self._parse_response(response, timer.elapsed)

    def _build_client(self) -> Any:
        """Instantiate the google-genai client. Separated for easy mocking in tests."""
        try:
            import google.genai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' library is required. "
                "Install it with: pip install google-genai"
            ) from exc
        return genai.Client(api_key=self._api_key)

    def _build_generation_config(self) -> Any:
        """Build the GenerateContentConfig object."""
        try:
            from google.genai import types  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' library is required. "
                "Install it with: pip install google-genai"
            ) from exc

        kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            kwargs["max_output_tokens"] = self.max_tokens

        return types.GenerateContentConfig(**kwargs)

    def _parse_response(self, response: Any, latency: float) -> ModelResponse:
        """Extract fields from a Gemini GenerateContentResponse object."""
        try:
            text = response.text if response.text is not None else ""
            usage = response.usage_metadata
            input_tokens = int(usage.prompt_token_count or 0)
            output_tokens = int(usage.candidates_token_count or 0)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Gemini response structure: {exc}"
            ) from exc

        cost = _calculate_cost(self._model_name, input_tokens, output_tokens)

        logger.info(
            "Gemini model='%s' tokens=(%d in, %d out) cost=$%.6f latency=%.3fs.",
            self._model_name, input_tokens, output_tokens, cost, latency,
        )

        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            latency=latency,
        )