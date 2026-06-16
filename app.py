import os
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- REMOVED OPENAI IMPORTS & ADDED GEMINI IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()


# ==========================================
# 1. INITIALIZATION & SETUP
# ==========================================
# Make sure your API key from Google AI Studio is exported in your environment:
# export GOOGLE_API_KEY="your_api_key_here"

DB_DIR = "./chroma_db"

# Initialize Gemini components
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
retrieval_chain = None

# ==========================================
# 2. THE RAG PIPELINE FUNCTION
# ==========================================
import shutil  # <-- Add this import at the top of your app.py file

def process_document(file_path):
    global retrieval_chain
    try:
        # 1. Create a stable, permanent local path in your current folder
        local_filename = "uploaded_document.pdf"
        
        # 2. Copy the file out of Gradio's locked temp directory into your project folder
        shutil.copy(file_path, local_filename)
        
        # Step 1: Use the local, unlocked file copy for PyMuPDFLoader
        loader = PyMuPDFLoader(local_filename)
        documents = loader.load()
        
        # Step 2: Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Step 3 & 4: Embeddings & Vector Database Store (Chroma)
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        
        # Step 5: Retriever Interface
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Step 6: Create the QA Chain (LLM + Retriever)
        system_prompt = (
            "You are an expert assistant specialized in analyzing documents.\n"
            "Answer the user's question using strictly the provided context below. "
            "If you do not know the answer or if it's not in the context, say "
            "'I cannot find the answer in the loaded document.' Do not make things up.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Optional: clean up the local file after processing is done
        if os.path.exists(local_filename):
            os.remove(local_filename)
            
        return f"✅ Successfully processed {len(chunks)} text chunks from the document! The bot is ready for your questions."
    
    except Exception as e:
        return f"❌ Error processing document: {str(e)}"

def answer_question(user_question):
    global retrieval_chain
    if retrieval_chain is None:
        return "⚠️ Please upload and process a document first before asking questions."
    
    try:
        response = retrieval_chain.invoke({"input": user_question})
        return response["answer"]
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

# ==========================================
# 3. GRADIO FRONT-END INTERFACE
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Document-Based Gemini RAG QA Bot")
    gr.Markdown("Upload a document (PDF), let the system chunk and embed it via Gemini, then ask real-time questions!")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Load Knowledge Base")
            file_input = gr.File(label="Upload Document (PDF)", file_types=[".pdf"])
            process_btn = gr.Button("Build Vector Database", variant="primary")
            status_output = gr.Textbox(label="System Status", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### Step 2: Chat with your Document")
            question_input = gr.Textbox(label="Ask a question about the document:", placeholder="e.g., What is the summary of section 3?")
            ask_btn = gr.Button("Submit Query", variant="secondary")
            answer_output = gr.Textbox(label="Bot Response", interactive=False, lines=6)

    process_btn.click(
        fn=process_document, 
        inputs=[file_input], 
        outputs=[status_output]
    )
    
    ask_btn.click(
        fn=answer_question, 
        inputs=[question_input], 
        outputs=[answer_output]
    )

if __name__ == "__main__":
    demo.launch()