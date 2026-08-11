"""Conversational retrieval over an indexed document.

Two things separate this from a single-shot RAG call:

* Follow-up questions ("what about the second one?") are rewritten into
  standalone queries before retrieval, so the retriever sees a question that
  makes sense without the conversation around it.
* Answers stream token by token, and the chunks they were grounded in are
  returned alongside so a reader can verify the claim.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Settings, require_api_key

CONTEXTUALIZE_SYSTEM = (
    "Given a chat history and the latest user question — which may reference "
    "earlier turns — rewrite it as a standalone question understandable "
    "without the chat history. Do NOT answer it. Return the question "
    "unchanged if it is already standalone."
)

ANSWER_SYSTEM = (
    "You are a careful research assistant answering questions about a single "
    "document.\n\n"
    "Rules:\n"
    "1. Answer using only the context below. It is the whole of what you know "
    "about this document.\n"
    "2. If the context does not contain the answer, say: 'I cannot find the "
    "answer in this document.' Never fill the gap from general knowledge.\n"
    "3. Cite the page you drew each claim from, inline, like [p. 12].\n"
    "4. If the context only partly answers the question, say what it supports "
    "and what it does not.\n\n"
    "Context:\n{context}"
)

# Each retrieved chunk enters the prompt labelled with its page, which is what
# lets the model produce the inline [p. N] citations asked for above.
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[{source}, p. {page_label}]\n{page_content}"
)


@dataclass
class StreamChunk:
    """One step of a streaming answer: text so far, plus its grounding."""

    answer: str
    sources: list[Document] = field(default_factory=list)


def to_messages(history: Sequence[Mapping[str, str]]) -> list[BaseMessage]:
    """Convert role/content turns into LangChain messages, ignoring others."""
    roles = {"user": HumanMessage, "assistant": AIMessage}
    return [
        roles[turn["role"]](content=turn["content"])
        for turn in history
        if turn.get("role") in roles and turn.get("content")
    ]


class RagPipeline:
    """A retrieval chain bound to one indexed document."""

    def __init__(self, vector_store: Chroma, settings: Settings, source_name: str = ""):
        self.settings = settings
        self.source_name = source_name

        llm = ChatGoogleGenerativeAI(
            model=settings.chat_model,
            temperature=settings.temperature,
            google_api_key=require_api_key(),
        )
        retriever = vector_store.as_retriever(
            search_kwargs={"k": settings.retrieval_k}
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            ChatPromptTemplate.from_messages(
                [
                    ("system", CONTEXTUALIZE_SYSTEM),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            ),
        )
        answer_chain = create_stuff_documents_chain(
            llm,
            ChatPromptTemplate.from_messages(
                [
                    ("system", ANSWER_SYSTEM),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            ),
            document_prompt=DOCUMENT_PROMPT,
        )
        self.chain = create_retrieval_chain(history_aware_retriever, answer_chain)

    def stream(
        self, question: str, history: Sequence[Mapping[str, str]] = ()
    ) -> Iterator[StreamChunk]:
        """Yield the answer as it is generated, with retrieved sources.

        The chain emits its retrieved context before the first answer token,
        so sources are attached from the start of the stream.
        """
        state = StreamChunk(answer="")
        payload = {"input": question, "chat_history": to_messages(history)}

        for step in self.chain.stream(payload):
            if "context" in step:
                state.sources = list(step["context"])
            if "answer" in step:
                state.answer += step["answer"]
                yield state

        if not state.answer:
            # A safety filter or an empty completion leaves nothing to show.
            yield StreamChunk(
                answer="The model returned an empty response. Try rephrasing "
                "the question.",
                sources=state.sources,
            )

    def answer(self, question: str, history: Sequence[Mapping[str, str]] = ()) -> StreamChunk:
        """Collect the full answer in one call, for scripted or batch use."""
        result = StreamChunk(answer="")
        for result in self.stream(question, history):
            pass
        return result
