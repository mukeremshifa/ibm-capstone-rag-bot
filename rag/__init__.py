"""A conversational RAG pipeline over a single PDF."""

from .config import Settings, require_api_key
from .ingest import IngestResult, ingest_pdf
from .pipeline import RagPipeline, StreamChunk

__all__ = [
    "IngestResult",
    "RagPipeline",
    "Settings",
    "StreamChunk",
    "ingest_pdf",
    "require_api_key",
]
