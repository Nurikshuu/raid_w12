"""Centralized config, loaded from environment / .env (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    top_k: int = int(os.getenv("TOP_K", "5"))

    index_dir: Path = ROOT / os.getenv("INDEX_DIR", "data/index")
    chunks_path: Path = ROOT / os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
    cache_dir: Path = ROOT / os.getenv("CACHE_DIR", "data/cache")

    retrieval_cache_size: int = int(os.getenv("RETRIEVAL_CACHE_SIZE", "256"))
    response_cache_ttl_seconds: float = float(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "3600"))


settings = Settings()
