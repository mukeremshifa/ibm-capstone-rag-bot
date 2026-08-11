# 🤖 Enterprise PDF-Insight Bot: Production-Grade RAG Architecture

![QA_bot](QA_bot.png)

A powerful, context-aware Retrieval-Augmented Generation (RAG) system built to ingest large-scale documentation, index semantic meaning into a vector space, and handle complex real-time user queries with extreme accuracy. Powered by LangChain, Google Gemini, and Chroma DB.

---

## 🚀 Key Features

- **Advanced Document Parsing:** Employs high-speed binary parsing via PyMuPDF to extract text stream structures seamlessly.
- **Deterministic Responses:** Temperature-locked LLM orchestration ensuring zero hallucinations and strict context adherence.
- **Localized Vector Embeddings:** Integrated with light-weight semantic tokenizers to abstract complex mathematical vectors locally.
- **Reactive UX Framework:** Built on top of an intuitive, multi-column Gradio interface engineered for rapid testing and real-time interaction workflows.

---

## 🛠️ System Architecture

The pipeline follows a highly efficient, production-grade 6-stage RAG lifecycle:

1. **Document Loading:** Native extraction of file binaries into clean string data structures.
2. **Text Chunking:** Recursive partitioning using target chunk overlaps to preserve semantic context across chunk edges.
3. **Vector Ingestion:** Mapping text structures to abstract high-dimensional geometric coordinates.
4. **Persistence Layer:** Storage of vectorized matrix profiles inside an in-memory database instance (Chroma DB).
5. **Retriever Sub-system:** Real-time semantic K-Nearest Neighbors (KNN) query calculations against user input string vectors.
6. **Contextual Synthesizer:** Injecting localized source information right into the LLM system prompt template for perfect inference.

---

## 📦 Tech Stack & Packages

- **Framework:** LangChain (`langchain-classic`, `langchain-community`, `langchain-core`)
- **Inference Engine:** Google GenAI (`gemini-2.5-flash`)
- **Vector Store & Indexing:** Chroma DB / HuggingFace Transformers
- **UI/UX Layer:** Gradio Web Server
- **Environment:** Python managed natively via the hyper-fast `uv` project dependency management tool.

---

## 🛠️ Step-by-Step Installation

1. **Clone the repository:**

```bash
   git clone [https://github.com/YOUR_USERNAME/pdf-insight-rag-bot.git](https://github.com/YOUR_USERNAME/pdf-insight-rag-bot.git)
   cd pdf-insight-rag-bot
```
