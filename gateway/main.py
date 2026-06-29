import os
import time
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from gateway.config.loader import load_teams, MODEL_REGISTRY
from gateway.providers.openai_provider import OpenAIProvider
from gateway.providers.anthropic_provider import AnthropicProvider
from gateway.providers.ollama_provider import OllamaProvider
from gateway.providers.base import ModelConfig

load_dotenv()

app = FastAPI(title="LLM Gateway")

# Load team configs and provider instances at startup
TEAMS = load_teams()
PROVIDERS = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "ollama": OllamaProvider(),
}


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False


@app.post("/v1/completions")
def completions(
    request: CompletionRequest,
    authorization: Optional[str] = Header(None)
):
    # --- Auth ---
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    api_key = authorization.removeprefix("Bearer ").strip()
    team = TEAMS.get(api_key)
    if not team:
        raise HTTPException(status_code=401, detail="Unknown API key")

    # --- Model access check ---
    if request.model not in team.allowed_models:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{request.model}' is not allowed for your team"
        )

    # --- Resolve model config ---
    model_config = MODEL_REGISTRY.get(request.model)
    if not model_config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

    # --- Request enrichment ---
    enriched_prompt = request.prompt
    if team.system_prompt_prefix:
        enriched_prompt = f"{team.system_prompt_prefix}\n\n{enriched_prompt}"
    if team.compliance_suffix:
        enriched_prompt = f"{enriched_prompt}\n\n{team.compliance_suffix}"

    # --- Route to provider ---
    provider = PROVIDERS.get(model_config.provider_name)
    if not provider:
        raise HTTPException(status_code=500, detail="Provider not available")

    response = provider.send_request(enriched_prompt, model_config)

    return {
        "output": response.output_text,
        "model_served": response.model_id,
        "provider": model_config.provider_name,
        "usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": round(response.cost_usd, 8),
            "latency_ms": round(response.latency_ms, 2),
        }
    }


@app.get("/v1/health")
def health():
    return {"status": "ok"}


@app.get("/v1/admin/teams")
def list_teams():
    return [
        {
            "team_id": t.team_id,
            "allowed_models": t.allowed_models,
            "requests_per_minute": t.requests_per_minute,
            "monthly_budget_usd": t.monthly_budget_usd,
        }
        for t in TEAMS.values()
    ]