"""Single place for runtime configuration (env vars, optionally from .env)."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# USD per 1M tokens: (input, cached input, output). Override/extend with MODEL_PRICES='{"model": [in, cached, out]}'.
DEFAULT_PRICES = {
    "gpt-5.5": (5.0, 0.5, 30.0),  # prompts under 272K tokens
    "gpt-5": (1.25, 0.125, 10.0),
    "gpt-5.1": (1.25, 0.125, 10.0),
    "gpt-5-mini": (0.25, 0.025, 2.0),
    "gpt-5-nano": (0.05, 0.005, 0.4),
}


def _prices() -> dict[str, tuple[float, float, float]]:
    prices = dict(DEFAULT_PRICES)
    prices.update({k: tuple(v) for k, v in json.loads(_env("MODEL_PRICES") or "{}").items()})
    return prices


@dataclass
class Settings:
    openai_api_key: str = _env("OPENAI_API_KEY")
    openai_base_url: str | None = _env("OPENAI_BASE_URL") or None
    model_extract: str = _env("MODEL_EXTRACT", "gpt-5-mini")
    model_match: str = _env("MODEL_MATCH", "gpt-5.5")
    model_report: str = _env("MODEL_REPORT", "gpt-5-mini")
    reasoning_extract: str = _env("REASONING_EXTRACT", "low")
    reasoning_match: str = _env("REASONING_MATCH", "medium")
    demo_mode: bool = _env("DEMO_MODE", "0") == "1"
    quote_min_score: int = int(_env("QUOTE_MIN_SCORE", "85"))
    data_dir: Path = ROOT / "data"
    jobs_dir: Path = ROOT / "data" / "jobs"
    demo_result_path: Path = ROOT / "data" / "demo_result.json"
    frontend_dir: Path = ROOT / "frontend"
    prices: dict = field(default_factory=_prices)


settings = Settings()
