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
    """

    text: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency: float

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

    def to_dict(self) -> dict[str, Any]:
        """Return the response as a plain dictionary for serialization."""
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": float(self.cost),
            "latency": float(self.latency),
        }

    @property
    def total_tokens(self) -> int:
        """Convenience property: sum of input and output tokens."""
        return self.input_tokens + self.output_tokens


class BaseModel(ABC):
    """
    Abstract base class for all LLM provider adapters.

    Subclasses must implement:
        - run(prompt: str) -> ModelResponse

    Subclasses may optionally override:
        - model_name (property)
        - provider (property)
    """

    @property
    def model_name(self) -> str:
        """Identifier for the specific model being called."""
        return self.__class__.__name__

    @property
    def provider(self) -> str:
        """Name of the LLM provider (e.g. 'openai', 'anthropic', 'google')."""
        return "unknown"

    @abstractmethod
    def run(self, prompt: str) -> ModelResponse:
        """
        Send a prompt to the model and return a normalized ModelResponse.

        Args:
            prompt: The fully rendered prompt string to send to the model.

        Returns:
            ModelResponse containing text, token counts, cost, and latency.

        Raises:
            ValueError: If the prompt is empty.
            RuntimeError: If the API call fails.
        """

    def _validate_prompt(self, prompt: str) -> None:
        """
        Validate that the prompt is a non-empty string.
        Subclasses should call this at the start of run().

        Raises:
            ValueError: If the prompt is empty or not a string.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"prompt must be a str, got {type(prompt).__name__}")
        if not prompt.strip():
            raise ValueError("prompt must not be empty.")

    def _measure_latency(self) -> _LatencyTimer:
        """Return a context manager that measures elapsed wall-clock time."""
        return _LatencyTimer()


class _LatencyTimer:
    """Simple context manager for measuring elapsed time in seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> _LatencyTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start