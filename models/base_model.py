"""
models/base_model.py

Abstract base class and shared response schema for all model adapters.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """
    Normalized response returned by every model adapter.

    Attributes:
        text:          The model's generated text output.
        input_tokens:  Number of tokens in the prompt sent to the model.
        output_tokens: Number of tokens in the model's response.
        cost:          Estimated cost in USD for this request.
        latency:       Wall-clock time in seconds from request to response.
        confidence:    Probability that the output is correct, in [0.0, 1.0].
                       None means confidence is not available for this provider/call.
                       Source is recorded in confidence_source.
        confidence_source: How confidence was estimated:
                       "logprobs"    — derived from token log-probabilities (OpenAI).
                                       Most statistically meaningful.
                       "self_report" — model was asked to rate its own confidence.
                                       Weakly calibrated; treat as a soft signal only.
                       None          — no confidence available.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency: float
    confidence: float | None = field(default=None)
    confidence_source: str | None = field(default=None)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        if not isinstance(self.input_tokens, int) or self.input_tokens < 0:
            raise ValueError(f"input_tokens must be a non-negative int, got {self.input_tokens!r}")
        if not isinstance(self.output_tokens, int) or self.output_tokens < 0:
            raise ValueError(f"output_tokens must be a non-negative int, got {self.output_tokens!r}")
        if not isinstance(self.cost, (int, float)) or self.cost < 0:
            raise ValueError(f"cost must be a non-negative number, got {self.cost!r}")
        if not isinstance(self.latency, (int, float)) or self.latency < 0:
            raise ValueError(f"latency must be a non-negative number, got {self.latency!r}")
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise TypeError(f"confidence must be a float or None, got {type(self.confidence).__name__}")
            if not (0.0 <= self.confidence <= 1.0):
                raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return the response as a plain dictionary for serialization."""
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": float(self.cost),
            "latency": float(self.latency),
            "confidence": self.confidence,
            "confidence_source": self.confidence_source,
        }

    @property
    def total_tokens(self) -> int:
        """Convenience property: sum of input and output tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def low_confidence(self, threshold: float = 0.7) -> bool | None:
        """
        True if confidence is below threshold, False if above, None if unavailable.
        Useful for routing: if response.low_confidence: trigger_review()
        """
        if self.confidence is None:
            return None
        return self.confidence < threshold


class BaseModel(ABC):
    """
    Abstract base class for all LLM provider adapters.

    Subclasses must implement:
        - run(prompt, generation_params) -> ModelResponse
    """

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    @property
    def provider(self) -> str:
        return "unknown"

    @abstractmethod
    def run(
        self,
        prompt: str | dict[str, str],
        generation_params: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Send a prompt to the model and return a normalized ModelResponse."""

    def _validate_prompt(self, prompt: str) -> None:
        if not isinstance(prompt, str):
            raise ValueError(f"prompt must be a str, got {type(prompt).__name__}")
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

    def _measure_latency(self) -> _LatencyTimer:
        return _LatencyTimer()


class _LatencyTimer:
    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> _LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start