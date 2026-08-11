"""Gradio front end for the document QA bot.

The UI keeps no module-level state: the indexed document and the pipeline
built over it live in per-session gr.State, so concurrent users of one
running instance never share a knowledge base.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import gradio as gr
from langchain_core.documents import Document

from rag import RagPipeline, Settings, ingest_pdf

SETTINGS = Settings.from_env()
SAMPLE_PDF = Path(__file__).parent / "state-of-the-union.pdf"

SAMPLE_QUESTIONS = [
    "What does the document say about inflation and the cost of living?",
    "Summarise the position on Ukraine.",
    "What is proposed for manufacturing jobs?",
    "Who is nominated to the Supreme Court, and what is said about them?",
]

NO_SOURCES = "_Ask a question and the passages behind the answer appear here._"


def format_sources(docs: list[Document]) -> str:
    """Render retrieved chunks as readable, page-numbered evidence."""
    if not docs:
        return NO_SOURCES

    blocks = []
    for rank, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page_label", "?")
        source = doc.metadata.get("source", "document")
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 500:
            snippet = snippet[:500].rstrip() + "…"
        blocks.append(f"**{rank}. {source} — page {page}**\n\n> {snippet}")
    return "\n\n---\n\n".join(blocks)


def build_index(file_path: str | None):
    """Index an uploaded PDF and hand back a pipeline bound to it."""
    if not file_path:
        return None, "⚠️ Choose a PDF first.", gr.update(), gr.update()

    # Copy out of Gradio's temp area before handing the path to PyMuPDF:
    # on Windows the upload can still be held open, which fails the read.
    suffix = Path(file_path).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        working_copy = Path(tmp.name)
    try:
        shutil.copy(file_path, working_copy)
        result = ingest_pdf(working_copy, SETTINGS, display_name=Path(file_path).name)
    except Exception as exc:  # surfaced in the UI rather than the console
        return None, f"❌ Could not index the document: {exc}", gr.update(), gr.update()
    finally:
        working_copy.unlink(missing_ok=True)

    pipeline = RagPipeline(result.vector_store, SETTINGS, result.source_name)
    status = (
        f"✅ **{result.source_name}** indexed — {result.page_count} pages split into "
        f"{result.chunk_count} chunks, embedded with `{SETTINGS.embedding_model}`. "
        f"Each answer retrieves the top {SETTINGS.retrieval_k} chunks."
    )
    return pipeline, status, [], NO_SOURCES


def load_sample():
    """Index the bundled sample so the demo is usable without an upload."""
    if not SAMPLE_PDF.exists():
        return None, f"❌ Sample file not found at {SAMPLE_PDF.name}.", gr.update(), gr.update()
    return build_index(str(SAMPLE_PDF))


def add_user_message(question: str, history: list[dict]):
    """Append the question and clear the box before the answer streams in."""
    question = question.strip()
    if not question:
        return "", history
    return "", history + [{"role": "user", "content": question}]


def stream_reply(history: list[dict], pipeline: RagPipeline | None):
    """Stream the assistant turn, updating the sources panel alongside it."""
    if not history or history[-1]["role"] != "user":
        return

    question = history[-1]["content"]
    prior = history[:-1]

    if pipeline is None:
        history.append(
            {
                "role": "assistant",
                "content": "⚠️ Load a document first — upload a PDF or click "
                "**Try the sample document**.",
            }
        )
        yield history, NO_SOURCES
        return

    history = history + [{"role": "assistant", "content": ""}]
    try:
        for chunk in pipeline.stream(question, prior):
            history[-1]["content"] = chunk.answer
            yield history, format_sources(chunk.sources)
    except Exception as exc:
        history[-1]["content"] = f"❌ The query failed: {exc}"
        yield history, NO_SOURCES


with gr.Blocks(theme=gr.themes.Soft(), title="Document QA Bot") as demo:
    pipeline_state = gr.State(None)

    gr.Markdown(
        "# 📄 Document QA Bot\n"
        "Ask questions about a PDF and get answers grounded in it — with the "
        "retrieved passages shown next to every reply, so you can check the "
        "answer against the source."
    )

    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="PDF", file_types=[".pdf"], type="filepath", height=110
            )
            with gr.Row():
                index_btn = gr.Button("Index document", variant="primary", scale=2)
                sample_btn = gr.Button("Try the sample document", scale=2)
            status = gr.Markdown("_No document loaded yet._")

            gr.Markdown("### Retrieved passages")
            sources_panel = gr.Markdown(NO_SOURCES)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=460,
                resizable=True,
                placeholder="Load a document, then ask away. Follow-up "
                "questions keep their context.",
            )
            question = gr.Textbox(
                label="Question",
                placeholder="e.g. What does the document say about inflation?",
                autofocus=True,
                submit_btn=True,
            )
            gr.Examples(SAMPLE_QUESTIONS, inputs=question, label="Example questions")
            clear_btn = gr.Button("Clear conversation", size="sm")

    index_outputs = [pipeline_state, status, chatbot, sources_panel]
    index_btn.click(build_index, [file_input], index_outputs)
    sample_btn.click(load_sample, None, index_outputs)

    question.submit(
        add_user_message, [question, chatbot], [question, chatbot]
    ).then(stream_reply, [chatbot, pipeline_state], [chatbot, sources_panel])

    clear_btn.click(lambda: ([], NO_SOURCES), None, [chatbot, sources_panel])


if __name__ == "__main__":
    demo.launch()
