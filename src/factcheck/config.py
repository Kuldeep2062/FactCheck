import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

load_dotenv(ROOT / ".env")

DATASET_VERSION = "long-brand-answers-50-v4"
FACT_VERSION = "speaker-catalogue-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"
AGENT_VERSIONS = ("v1", "v2")
LABELS = ("supported", "contradicted", "unknown")


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: os.getenv("FACTCHECK_PROVIDER", "groq"))
    model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
                       if os.getenv("FACTCHECK_PROVIDER", "groq") == "groq"
                       else os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    temperature: float = 0.0
    # Gemini: thinking level; Groq gpt-oss: reasoning_effort. Same knob, provider-specific name.
    thinking_level: str = field(default_factory=lambda: os.getenv("REASONING_LEVEL", "low"))
    max_concurrent_requests: int = 3
    request_timeout_s: float = 15.0
    max_attempts_per_turn: int = 2
    retry_delay_s: float = 2.0
    retry_delay_cap_s: float = 10.0
    max_model_turns: int = 3
    # Free-tier Gemini keys have a per-minute request quota; spacing request starts
    # avoids burning the single allowed retry on predictable 429s.
    requests_per_minute: float = field(default_factory=lambda: float(os.getenv("PROVIDER_RPM", "25")))
    # Groq free tier enforces 8000 tokens/minute; a lower budget absorbs estimation error.
    tokens_per_minute: float = field(default_factory=lambda: float(os.getenv("PROVIDER_TPM", "7500")))

    def as_dict(self) -> dict:
        return asdict(self)
