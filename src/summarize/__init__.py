from __future__ import annotations

from config import AppConfig
from src.summarize.base import Summarizer


def get_summarizer(cfg: AppConfig) -> Summarizer:
    backend = cfg.summarizer.backend
    if backend == "claude":
        from src.summarize.claude import ClaudeSummarizer
        if not cfg.secrets.anthropic_api_key:
            raise RuntimeError("summarizer.backend=claude but ANTHROPIC_API_KEY is not set")
        return ClaudeSummarizer(
            api_key=cfg.secrets.anthropic_api_key,
            model=cfg.summarizer.claude.model,
            max_tokens=cfg.summarizer.claude.max_tokens,
        )
    if backend == "openai":
        from src.summarize.openai_backend import OpenAISummarizer
        if not cfg.secrets.openai_api_key:
            raise RuntimeError("summarizer.backend=openai but OPENAI_API_KEY is not set")
        return OpenAISummarizer(
            api_key=cfg.secrets.openai_api_key,
            model=cfg.summarizer.openai.model,
            max_tokens=cfg.summarizer.openai.max_tokens,
        )
    if backend == "ollama":
        from src.summarize.ollama_backend import OllamaSummarizer
        return OllamaSummarizer(
            host=cfg.summarizer.ollama.host,
            model=cfg.summarizer.ollama.model,
        )
    raise ValueError(f"Unknown summarizer backend: {backend}")


__all__ = ["Summarizer", "get_summarizer"]
