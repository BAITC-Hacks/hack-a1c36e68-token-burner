"""Single place for runtime configuration (env vars, optionally from .env)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = _env("OPENAI_API_KEY")
    openai_base_url: str | None = _env("OPENAI_BASE_URL") or None
    model_extract: str = _env("MODEL_EXTRACT", "gpt-5-mini")
    model_match: str = _env("MODEL_MATCH", "gpt-5")
    model_report: str = _env("MODEL_REPORT", "gpt-5-mini")
    reasoning_extract: str = _env("REASONING_EXTRACT", "low")
    reasoning_match: str = _env("REASONING_MATCH", "medium")
    demo_mode: bool = _env("DEMO_MODE", "0") == "1"
    quote_min_score: int = int(_env("QUOTE_MIN_SCORE", "85"))
    data_dir: Path = ROOT / "data"
    jobs_dir: Path = ROOT / "data" / "jobs"
    demo_result_path: Path = ROOT / "data" / "demo_result.json"
    frontend_dir: Path = ROOT / "frontend"


settings = Settings()
