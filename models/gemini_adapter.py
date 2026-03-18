"""
models/gemini_adapter.py

Google Gemini model adapter using the google-genai SDK (current).
Replaces the deprecated google-generativeai package.

Install: pip install google-genai
API key: GEMINI_API_KEY environment variable.

Confidence method: self_report
  Gemini does not expose token log-probabilities. When confidence is
  requested the prompt is extended to ask for a CONFIDENCE: <0-100> score.
  Treat as a soft routing signal only, not a calibrated probability.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from models.base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)

_COST_PER_1K: dict[str, dict[str, float]] = {
    # Gemini 2.5 family (current)
    "gemini-2.5-pro":        {"input": 0.00125,   "output": 0.010},
    "gemini-2.5-flash":      {"input": 0.0000375, "output": 0.00150},
    "gemini-2.5-flash-lite": {"input": 0.000010,  "output": 0.000040},
    # Gemini 2.0 legacy
    "gemini-2.0-flash":      {"input": 0.0001,    "output": 0.0004},
    "gemini-2.0-flash-lite": {"input": 0.000075,  "output": 0.0003},
}

_DEFAULT_COST: dict[str, float] = {"input": 0.00125, "output": 0.005}
_ENV_KEY = "GEMINI_API_KEY"

_CONFIDENCE_INSTRUCTION = (
    "\nAfter your answer, on a new line write only: "
    "CONFIDENCE: <integer 0-100>"
)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\d+)", re.IGNORECASE)

# finish_reason values that mean the response was blocked, not a real error
# finish_reason can be an int or a FinishReason enum depending on SDK version.
# Compare by name so both representations work.
_BLOCKED_FINISH_REASON_NAMES = {"SAFETY", "RECITATION", "OTHER", "PROHIBITED_CONTENT"}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_PER_1K.get(model, _DEFAULT_COST)
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000


def _parse_self_report(text: str) -> tuple[str, float | None]:
    match = _CONFIDENCE_RE.search(text)
    if not match:
        return text.strip(), None
    score = min(max(int(match.group(1)), 0), 100)
    label = _CONFIDENCE_RE.sub("", text).strip()
    return label, round(score / 100, 4)


def _extract_text_safe(response: Any) -> str | None:
    """
    Safely extract text from a Gemini response.

    Returns None if the response was blocked (finish_reason 2/3/4)
    rather than raising, so the caller can handle it gracefully.
    """
    try:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        # finish_reason may be an int OR a FinishReason enum — compare by name
        reason_name = (
            finish_reason.name if hasattr(finish_reason, "name")
            else str(finish_reason).upper()
        )
        if reason_name in _BLOCKED_FINISH_REASON_NAMES:
            logger.debug("Gemini response blocked, finish_reason=%s", finish_reason)
            return None
        # MAX_TOKENS: gemini-2.5-flash uses this as its normal stop reason.
        # Only warn if there are genuinely no parts (truly empty output).
        # Parts may be empty even without a safety block
        parts = getattr(candidate.content, "parts", [])
        if not parts:
            return ""
        return "".join(getattr(p, "text", "") for p in parts)
    except (AttributeError, IndexError):
        return None


class GeminiAdapter(BaseModel):
    """
    Adapter for Google Gemini models via the google-genai SDK.

    Args:
        model_name:        Gemini model identifier (e.g. 'gemini-2.5-flash').
        temperature:       Default sampling temperature.
        max_tokens:        Default max output tokens.
        top_p:             Default top_p.
        report_confidence: If True, append confidence instruction to prompt.
        api_key:           Optional override. Defaults to GEMINI_API_KEY env var.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int | None = 1024,
        top_p: float = 1.0,
        report_confidence: bool = False,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.report_confidence = report_confidence
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
        contents, system_instruction = self._parse_prompt(prompt)
        if self.report_confidence:
            contents = contents + _CONFIDENCE_INSTRUCTION

        params = self._resolve_params(generation_params)
        client = self._build_client()
        config = self._build_config(params, system_instruction)

        logger.info("Calling Gemini model='%s'.", self._model_name)

        # Retry on transient errors (503 high demand, 429 rate limit)
        # with exponential backoff. Fatal errors (400, 404) raise immediately.
        _RETRYABLE = ("503", "429", "unavailable", "resource_exhausted")
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with self._measure_latency() as timer:
                    response = client.models.generate_content(
                        model=self._model_name,
                        contents=contents,
                        config=config,
                    )
                break  # success
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if any(f in msg for f in _RETRYABLE):
                    wait = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "Gemini transient error (attempt %d/3), retrying in %ds: %s",
                        attempt + 1, wait, exc,
                    )
                    import time; time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Gemini API call failed for model '{self._model_name}': {exc}"
                    ) from exc
        else:
            raise RuntimeError(
                f"Gemini API call failed after 3 attempts for model '{self._model_name}': {last_exc}"
            ) from last_exc

        return self._parse_response(response, timer.elapsed)

    def _parse_prompt(self, prompt: str | dict[str, str]) -> tuple[str, str]:
        if isinstance(prompt, str):
            self._validate_prompt(prompt)
            return prompt, ""
        if not isinstance(prompt, dict) or "user" not in prompt:
            raise ValueError("Prompt dict must contain a 'user' key.")
        if not str(prompt["user"]).strip():
            raise ValueError("prompt['user'] must not be empty.")
        return prompt["user"], prompt.get("system", "")

    # Gemini tokenizes differently from OpenAI — "science_tech" alone is 4+ tokens.
    # Apply a hard floor regardless of what generation_config.yaml sets.
    # gemini-2.5-pro and gemini-2.5-flash are thinking models: max_tokens covers
    # the internal reasoning chain + visible output combined. Pro's reasoning
    # chain is significantly longer than flash's (~2000-4000 tokens for complex
    # tasks), so we use a model-aware floor. Flash needs ~512, pro needs ~4096.
    _MIN_TOKENS_PRO   = 4096   # pro reasoning budget + output
    _MIN_TOKENS_FLASH = 512    # flash reasoning budget + output
    _CONF_EXTRA       = 100    # extra headroom for CONFIDENCE line

    # Models considered "pro-tier" thinking models needing the larger floor
    _PRO_MODELS = {"gemini-2.5-pro", "gemini-2.5-pro-preview"}

    def _resolve_params(self, generation_params: dict[str, Any] | None) -> dict[str, Any]:
        base_max = (
            generation_params.get("max_tokens", self.max_tokens)
            if generation_params else self.max_tokens
        )
        if base_max is not None:
            is_pro = any(p in self._model_name for p in self._PRO_MODELS)
            base_floor = self._MIN_TOKENS_PRO if is_pro else self._MIN_TOKENS_FLASH
            if self.report_confidence:
                base_floor += self._CONF_EXTRA
            base_max = max(base_max, base_floor)
        return {
            "temperature": generation_params.get("temperature", self.temperature)
                           if generation_params else self.temperature,
            "top_p":       generation_params.get("top_p", self.top_p)
                           if generation_params else self.top_p,
            "max_tokens":  base_max,
        }

    def _build_client(self) -> Any:
        """Instantiate the google-genai client. Separated for easy mocking."""
        try:
            import google.genai as genai  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' library is required. "
                "Install with: pip install google-genai"
            ) from exc
        return genai.Client(api_key=self._api_key)

    def _build_config(self, params: dict[str, Any], system_instruction: str = "") -> Any:
        """Build GenerateContentConfig. Separated for easy mocking."""
        try:
            from google.genai import types  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("The 'google-genai' library is required.") from exc

        kwargs: dict[str, Any] = {
            "temperature": params["temperature"],
            "top_p":       params["top_p"],
        }
        if params.get("max_tokens"):
            kwargs["max_output_tokens"] = params["max_tokens"]
        if system_instruction:
            kwargs["system_instruction"] = system_instruction

        return types.GenerateContentConfig(**kwargs)

    def _parse_response(self, response: Any, latency: float) -> ModelResponse:
        # Token counts — usage_metadata present even on blocked responses
        try:
            usage = response.usage_metadata
            input_tokens = int(getattr(usage, "prompt_token_count", None) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", None) or 0)
        except (AttributeError, TypeError):
            input_tokens, output_tokens = 0, 0

        raw_text = _extract_text_safe(response)

        if raw_text is None:
            # Blocked response — return empty string, no confidence
            # Caller sees an empty prediction which will score as incorrect
            # but does NOT raise, so the run continues.
            logger.debug("Gemini blocked response for model='%s'.", self._model_name)
            return ModelResponse(
                text="", input_tokens=input_tokens, output_tokens=output_tokens,
                cost=_calculate_cost(self._model_name, input_tokens, output_tokens),
                latency=latency, confidence=None, confidence_source=None,
            )

        if self.report_confidence:
            text, confidence = _parse_self_report(raw_text)
            confidence_source = "self_report" if confidence is not None else None
        else:
            text, confidence, confidence_source = raw_text, None, None

        cost = _calculate_cost(self._model_name, input_tokens, output_tokens)
        logger.info(
            "Gemini model='%s' tokens=(%d in, %d out) cost=$%.6f latency=%.3fs confidence=%s.",
            self._model_name, input_tokens, output_tokens, cost, latency, confidence,
        )
        return ModelResponse(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens,
            cost=cost, latency=latency, confidence=confidence,
            confidence_source=confidence_source,
        )