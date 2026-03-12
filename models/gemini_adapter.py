"""
models/gemini_adapter.py

Google Gemini model adapter. Implements BaseModel using the google-generativeai SDK.
API key is read from the GEMINI_API_KEY environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from models.base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)

_COST_PER_1K: dict[str, dict[str, float]] = {
    "gemini-2.5-pro":        {"input": 0.00125,  "output": 0.010},
    "gemini-2.5-flash":      {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash":      {"input": 0.0001,   "output": 0.0004},
    "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro":        {"input": 0.00125,  "output": 0.005},
    "gemini-1.5-flash":      {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-flash-8b":   {"input": 0.0000375,"output": 0.00015},
}

_DEFAULT_COST: dict[str, float] = {"input": 0.00125, "output": 0.005}
_ENV_KEY = "GEMINI_API_KEY"


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_PER_1K.get(model, _DEFAULT_COST)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000


class GeminiAdapter(BaseModel):
    """
    Adapter for Google Gemini models via the google-generativeai SDK.

    Args:
        model_name: Gemini model identifier (e.g. 'gemini-2.5-flash').
        temperature: Default sampling temperature (overridden by generation_params).
        max_tokens: Default max output tokens (overridden by generation_params).
        top_p: Default top_p (overridden by generation_params).
        api_key: Optional API key override. Defaults to GEMINI_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int | None = 1024,
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
                f"Gemini API key not found. Set the {_ENV_KEY!r} environment variable."
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "google"

    def run(
        self,
        prompt: str | dict[str, str],
        generation_params: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """
        Send a prompt to the Gemini generate_content API and return a ModelResponse.

        Args:
            prompt: A plain string (user message) or normalized schema:
                    {"system": "...", "user": "..."}
            generation_params: Normalized params with keys temperature, top_p,
                               max_tokens. Falls back to adapter defaults.
        """
        contents, system_instruction = self._parse_prompt(prompt)
        params = self._resolve_params(generation_params)
        client = self._build_client(system_instruction)
        generation_config = self._build_generation_config(params)

        logger.info("Calling Gemini model='%s'.", self._model_name)

        try:
            with self._measure_latency() as timer:
                response = client.generate_content(
                    contents=contents,
                    generation_config=generation_config,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Gemini API call failed for model '{self._model_name}': {exc}"
            ) from exc

        return self._parse_response(response, timer.elapsed)

    def _parse_prompt(self, prompt: str | dict[str, str]) -> tuple[str, str]:
        """Return (user_contents, system_instruction) from string or normalized schema."""
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

    def _build_client(self, system_instruction: str = "") -> Any:
        """Instantiate the google-generativeai model. Separated for easy mocking."""
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-generativeai' library is required. "
                "Install it with: pip install google-generativeai"
            ) from exc
        genai.configure(api_key=self._api_key)
        kwargs: dict[str, Any] = {}
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        return genai.GenerativeModel(self._model_name, **kwargs)

    def _build_generation_config(self, params: dict[str, Any]) -> Any:
        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-generativeai' library is required."
            ) from exc
        kwargs: dict[str, Any] = {
            "temperature": params["temperature"],
            "top_p": params["top_p"],
        }
        if params.get("max_tokens"):
            kwargs["max_output_tokens"] = params["max_tokens"]
        return genai.GenerationConfig(**kwargs)

    def _parse_response(self, response: Any, latency: float) -> ModelResponse:
        try:
            text = response.text if response.text is not None else ""
            usage = response.usage_metadata
            input_tokens = int(usage.prompt_token_count or 0)
            output_tokens = int(usage.candidates_token_count or 0)
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response structure: {exc}") from exc

        cost = _calculate_cost(self._model_name, input_tokens, output_tokens)
        logger.info(
            "Gemini model='%s' tokens=(%d in, %d out) cost=$%.6f latency=%.3fs.",
            self._model_name, input_tokens, output_tokens, cost, latency,
        )
        return ModelResponse(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens,
            cost=cost, latency=latency,
        )