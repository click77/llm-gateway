from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Literal


@dataclass
class ModelConfig:
    provider_name: str
    model_id: str
    cost_per_input_token: float   # cost in USD per input token
    cost_per_output_token: float  # cost in USD per output token
    average_latency_ms: float     # expected latency in milliseconds
    quality_tier: Literal["high", "medium", "low"]


@dataclass
class Response:
    output_text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    model_id: str


class BaseProvider(ABC):

    @abstractmethod
    def send_request(self, prompt: str, model_config: ModelConfig) -> Response:
        """
        Send a prompt to the provider and return a standardized Response.
        Every provider subclass must implement this method.
        """
        ...