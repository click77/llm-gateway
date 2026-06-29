import time
import os
from openai import OpenAI
from .base import BaseProvider, ModelConfig, Response


class OpenAIProvider(BaseProvider):

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def send_request(self, prompt: str, model_config: ModelConfig) -> Response:
        start = time.monotonic()

        completion = self.client.chat.completions.create(
            model=model_config.model_id,
            messages=[{"role": "user", "content": prompt}]
        )

        latency_ms = (time.monotonic() - start) * 1000
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        output_text = completion.choices[0].message.content

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