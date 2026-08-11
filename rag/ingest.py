"""Turn a PDF into a queryable vector store.

Load -> split -> embed -> index. The embedding model is loaded once per
process and reused; the vector store is per-document and in-memory, so two
users of the same app instance never see each other's documents.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings


@lru_cache(maxsize=2)
def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Load the sentence-transformer once; it costs a few seconds each time.

    Embeddings run locally rather than through the Gemini API: the corpus is
    a single document, so the network round-trips would dominate indexing
    time, and it keeps document text off a third-party service.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


@dataclass
class IngestResult:
    vector_store: Chroma
    source_name: str
    page_count: int
    chunk_count: int


def load_pdf(path: str | Path, display_name: str | None = None) -> list[Document]:
    """Read a PDF into one Document per page, keeping only citation metadata.

    PyMuPDF attaches a dozen fields (producer, creation date, file path, ...)
    that would be embedded into every prompt. We keep the source name and a
    1-indexed page label, which is what a reader needs to check an answer.
    """
    path = Path(path)
    name = display_name or path.name
    pages = PyMuPDFLoader(str(path)).load()

    for page in pages:
        page_index = page.metadata.get("page", 0)
        page.metadata = {"source": name, "page_label": str(page_index + 1)}
    return pages


def split_pages(pages: list[Document], settings: Settings) -> list[Document]:
    """Split pages into overlapping chunks, dropping any that are blank.

    Scanned or image-only pages yield empty text; indexing them wastes
    retrieval slots on chunks that can never answer anything.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    return [c for c in chunks if c.page_content.strip()]


def build_vector_store(chunks: list[Document], settings: Settings) -> Chroma:
    """Index chunks into a fresh in-memory Chroma collection.

    A unique collection name per call means re-uploading a document replaces
    the knowledge base instead of merging into a stale one.
    """
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(settings.embedding_model),
        collection_name=f"doc-{uuid.uuid4().hex[:12]}",
    )


def ingest_pdf(
    path: str | Path, settings: Settings, display_name: str | None = None
) -> IngestResult:
    """Run the full load -> split -> index pass over one PDF."""
    pages = load_pdf(path, display_name)
    chunks = split_pages(pages, settings)
    if not chunks:
        raise ValueError(
            "No extractable text found. This PDF is probably a scan; "
            "it would need OCR before it can be indexed."
        )
    return IngestResult(
        vector_store=build_vector_store(chunks, settings),
        source_name=display_name or Path(path).name,
        page_count=len(pages),
        chunk_count=len(chunks),
    )
