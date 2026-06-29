import time
import requests
from .base import BaseProvider, ModelConfig, Response


class OllamaProvider(BaseProvider):

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def send_request(self, prompt: str, model_config: ModelConfig) -> Response:
        start = time.monotonic()

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model_config.model_id,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        data = response.json()

        latency_ms = (time.monotonic() - start) * 1000
        output_text = data.get("response", "")

        # Ollama does not always return token counts — estimate if missing
        input_tokens = data.get("prompt_eval_count", len(prompt.split()))
        output_tokens = data.get("eval_count", len(output_text.split()))

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