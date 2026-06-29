import yaml
import os
from dataclasses import dataclass
from typing import Dict
from dotenv import load_dotenv
from gateway.providers.base import ModelConfig

load_dotenv()


@dataclass
class TeamConfig:
    team_id: str
    api_key: str
    allowed_models: list
    requests_per_minute: int
    tokens_per_minute: int
    monthly_budget_usd: float
    system_prompt_prefix: str = ""
    compliance_suffix: str = ""
    content_filter_enabled: bool = False


def load_teams(path: str = "configs/teams.yaml") -> Dict[str, TeamConfig]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    teams = {}
    for team in data["teams"]:
        teams[team["api_key"]] = TeamConfig(**team)
    return teams


MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider_name="openai",
        model_id="gpt-4o",
        cost_per_input_token=0.000005,
        cost_per_output_token=0.000015,
        average_latency_ms=800,
        quality_tier="high"
    ),
    "gpt-4o-mini": ModelConfig(
        provider_name="openai",
        model_id="gpt-4o-mini",
        cost_per_input_token=0.00000015,
        cost_per_output_token=0.0000006,
        average_latency_ms=400,
        quality_tier="medium"
    ),
    "claude-sonnet-4-6": ModelConfig(
        provider_name="anthropic",
        model_id="claude-sonnet-4-6",
        cost_per_input_token=0.000003,
        cost_per_output_token=0.000015,
        average_latency_ms=700,
        quality_tier="high"
    ),
    "claude-haiku-4-5": ModelConfig(
        provider_name="anthropic",
        model_id="claude-haiku-4-5",
        cost_per_input_token=0.00000025,
        cost_per_output_token=0.00000125,
        average_latency_ms=300,
        quality_tier="medium"
    ),
    "llama3": ModelConfig(
        provider_name="ollama",
        model_id="llama3",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
        average_latency_ms=1200,
        quality_tier="low"
    ),
}