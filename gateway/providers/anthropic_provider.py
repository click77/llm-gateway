import time
import os
import anthropic
from .base import BaseProvider, ModelConfig, Response


class AnthropicProvider(BaseProvider):

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def send_request(self, prompt: str, model_config: ModelConfig) -> Response:
        start = time.monotonic()

        message = self.client.messages.create(
            model=model_config.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        latency_ms = (time.monotonic() - start) * 1000
        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        output_text = message.content[0].text

        cost = (
            input_tokens * model_config.cost_per_input_token +
            output_tokens * model_config.cost_per_output_token
        )

        return Response(
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            model_id=model_config.model_id
        )