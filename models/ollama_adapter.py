"""
models/ollama_adapter.py

Ollama local model adapter.

Ollama exposes an OpenAI-compatible REST API at http://localhost:11434/v1,
so this adapter uses the openai SDK pointed at the local endpoint.

No API key is required. No logprobs support (Ollama doesn't expose them).
Cost is reported as $0.00 — local inference has no per-token billing.

Requirements:
  - Ollama installed and running: https://ollama.com
  - Model pulled: ollama pull llama3.1:8b
  - pip install openai  (already in requirements.txt)

Base URL can be overridden via OLLAMA_BASE_URL env var for non-default ports.
Default: http://localhost:11434/v1
"""

from __future__ import annotations

import logging
import os
from typing import Any

from models.base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_ENV_BASE_URL = "OLLAMA_BASE_URL"


class OllamaAdapter(BaseModel):
    """
    Adapter for locally-running Ollama models.

    Args:
        model_name:  Ollama model tag (e.g. 'llama3.1:8b', 'mistral:7b').
        temperature: Default sampling temperature (overridden by generation_params).
        max_tokens:  Default max output tokens (overridden by generation_params).
        top_p:       Default top_p (overridden by generation_params).
        base_url:    Ollama API base URL. Defaults to OLLAMA_BASE_URL env var
                     or http://localhost:11434/v1.
    """

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        temperature: float = 0.0,
        max_tokens: int | None = 512,
        top_p: float = 1.0,
        base_url: str | None = None,
    ) -> None:
        self._model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._base_url = base_url or os.environ.get(_ENV_BASE_URL, _DEFAULT_BASE_URL)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "ollama"

    def run(
        self,
        prompt: str | dict[str, str],
        generation_params: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """
        Send a prompt to the local Ollama instance and return a ModelResponse.

        Args:
            prompt: Plain string or normalized schema {"system": "...", "user": "..."}.
            generation_params: Normalized params (temperature, top_p, max_tokens).
        """
        messages = self._build_messages(prompt)
        params   = self._resolve_params(generation_params)
        client   = self._build_client()

        logger.info("Calling Ollama model='%s' at %s.", self._model_name, self._base_url)

        try:
            with self._measure_latency() as timer:
                response = client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    **({"max_tokens": params["max_tokens"]} if params.get("max_tokens") else {}),
                )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama API call failed for model '{self._model_name}': {exc}\n"
                f"Is Ollama running? Try: ollama serve"
            ) from exc

        return self._parse_response(response, timer.elapsed)

    def _build_messages(self, prompt: str | dict[str, str]) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            self._validate_prompt(prompt)
            return [{"role": "user", "content": prompt}]
        if not isinstance(prompt, dict) or "user" not in prompt:
            raise ValueError("Prompt dict must contain a 'user' key.")
        if not str(prompt["user"]).strip():
            raise ValueError("prompt['user'] must not be empty.")
        messages = []
        if prompt.get("system"):
            messages.append({"role": "system", "content": prompt["system"]})
        messages.append({"role": "user", "content": prompt["user"]})
        return messages

    def _resolve_params(self, generation_params: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "temperature": generation_params.get("temperature", self.temperature)
                           if generation_params else self.temperature,
            "top_p":       generation_params.get("top_p", self.top_p)
                           if generation_params else self.top_p,
            "max_tokens":  generation_params.get("max_tokens", self.max_tokens)
                           if generation_params else self.max_tokens,
        }

    def _build_client(self) -> Any:
        """Instantiate OpenAI client pointed at local Ollama endpoint."""
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' library is required. Install with: pip install openai"
            ) from exc
        # Ollama doesn't require an API key — pass a placeholder
        return OpenAI(base_url=self._base_url, api_key="ollama")

    def _parse_response(self, response: Any, latency: float) -> ModelResponse:
        try:
            text         = response.choices[0].message.content or ""
            usage        = response.usage
            input_tokens  = int(usage.prompt_tokens)     if usage else 0
            output_tokens = int(usage.completion_tokens) if usage else 0
        except (AttributeError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response structure: {exc}") from exc

        # Local inference has no billing — cost is always $0.00
        logger.info(
            "Ollama model='%s' tokens=(%d in, %d out) latency=%.3fs.",
            self._model_name, input_tokens, output_tokens, latency,
        )
        return ModelResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=0.0,
            latency=latency,
            confidence=None,
            confidence_source=None,
        )