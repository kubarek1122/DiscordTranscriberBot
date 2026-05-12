from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class WhisperConfig(BaseModel):
    model: str = "large-v3"
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: str = "float16"
    language: str = "pl"


class ClaudeConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096


class OpenAIConfig(BaseModel):
    model: str = "gpt-4o"
    max_tokens: int = 4096


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:14b-instruct"


class SummarizerConfig(BaseModel):
    backend: Literal["claude", "openai", "ollama"] = "claude"
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


class RecordingConfig(BaseModel):
    output_dir: Path = Path("./recordings")
    keep_audio: bool = False
    chunk_seconds: int = 30
    idle_timeout_s: int = 300
    heartbeat_interval_s: int = 10
    heartbeat_stale_after_s: int = 60


class ReliabilityConfig(BaseModel):
    summarizer_retries: int = 3
    post_retries: int = 5


class RecorderConfig(BaseModel):
    socket_path: Path = Path("/tmp/skryba/recorder.sock")
    join_timeout_s: float = 15.0
    leave_timeout_s: float = 15.0


class Secrets(BaseModel):
    discord_token: str
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


class AppConfig(BaseModel):
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    secrets: Secrets


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    load_dotenv()
    raw: dict = {}
    cfg_path = Path(path)
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN is not set (check your .env)")

    secrets = Secrets(
        discord_token=token,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )

    return AppConfig(secrets=secrets, **raw)
