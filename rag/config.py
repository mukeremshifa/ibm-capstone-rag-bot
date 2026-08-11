"""Runtime configuration for the RAG pipeline.

Every tunable lives here so the retrieval behaviour can be changed (and
recorded alongside eval runs) without touching pipeline code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    """Pipeline knobs, overridable via environment variables."""

    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            chat_model=os.getenv("CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            retrieval_k=_env_int("RETRIEVAL_K", cls.retrieval_k),
        )


def require_api_key() -> str:
    """Return the Gemini API key, failing loudly if it is missing.

    langchain-google-genai accepts either name; we check both so a missing
    key surfaces as a clear message instead of an auth error mid-query.
    """
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key found. Copy .env.example to .env and set "
            "GEMINI_API_KEY, or export it in your shell. "
            "Get a key at https://aistudio.google.com/apikey"
        )
    return key
