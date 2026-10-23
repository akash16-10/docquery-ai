"""
DocQuery AI — Intelligent Document Q&A System
A RAG-powered application that lets users upload PDFs and ask questions,
receiving AI-generated answers grounded in document content with source citations.
"""

import os
import uuid
import shutil
import asyncio
import threading
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
FAISS_DIR  = BASE_DIR / "faiss_stores"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)

# Load .env — always resolves relative to this file, not CWD
_env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=_env_path, override=True)
logger.info(f".env path: {_env_path} | GROQ key present: {bool(os.getenv('GROQ_API_KEY'))}")

# ─── Global State ─────────────────────────────────────────────────────────────

# In-memory session storage
sessions: Dict[str, Dict[str, Any]] = {}

# Background processing jobs: job_id → {status, cancel_event, filename, ...}
processing_jobs: Dict[str, Dict[str, Any]] = {}

# Embedding model — lazy-loaded, thread-safe singleton
_embeddings: Optional[HuggingFaceEmbeddings] = None
_embeddings_lock = threading.Lock()


# ─── App Lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown lifecycle."""
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down, cleaning up resources...")
    for session_id in list(sessions.keys()):
        try:
            session = sessions.pop(session_id)
            fp = Path(session["file_path"])
            if fp.exists():
                fp.unlink()
            sp = FAISS_DIR / session_id
            if sp.exists():
                shutil.rmtree(sp)
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {e}")


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocQuery AI",
    description="RAG-powered Document Q&A System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    question:   str = Field(..., min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or only whitespace")
        return v.strip()


class QueryResponse(BaseModel):
    answer:     str
    sources:    list[dict]
    session_id: str


class UploadJobResponse(BaseModel):
    job_id:   str
    status:   str   # always "processing" on first response
    filename: str


class JobStatusResponse(BaseModel):
    job_id:        str
    status:        str   # processing | complete | cancelled | error
    filename:      Optional[str] = None
    total_pages:   Optional[int] = None
    total_chunks:  Optional[int] = None
    message:       Optional[str] = None
    error:         Optional[str] = None


class SessionInfo(BaseModel):
    session_id:          str
    filename:            str
    total_pages:         int
    total_chunks:        int
    chat_history_length: int


class SetKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=10, description="Groq API key starting with gsk_")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-load the embedding model (thread-safe singleton)."""
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                logger.info("Loading HuggingFace embedding model...")
                _embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                )
                logger.info("Embedding model loaded successfully")
    return _embeddings


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF with page numbers for source tracking."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page_num in range(len(doc)):
            try:
                text = doc[page_num].get_text("text").strip()
                if text:
                    pages.append({"page_number": page_num + 1, "content": text})
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num + 1}: {e}")
        doc.close()
        return pages
    except Exception as e:
        logger.error(f"Failed to open PDF: {e}")
        raise ValueError("Could not open or read the PDF file. It may be corrupted or password-protected.")


def chunk_documents(pages: list[dict]) -> list:
    """Split extracted pages into smaller chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        chunks.extend(splitter.create_documents(
            texts=[page["content"]],
            metadatas=[{"page_number": page["page_number"]}],
        ))
    return chunks


def create_vector_store(chunks: list, session_id: str) -> FAISS:
    """Create and persist a FAISS vector store from document chunks."""
    embeddings = get_embeddings()

    if len(chunks) > 1000:
        logger.info("Large document — processing embeddings in batches...")
        batch_size = 500
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        vector_store = None
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.merge_from(FAISS.from_documents(batch, embeddings))
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    store_path = FAISS_DIR / session_id
    vector_store.save_local(str(store_path))
    logger.info("Vector store saved to disk")
    return vector_store


def build_qa_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """Build the conversational retrieval QA chain using Groq LLM."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set. Please enter your API key in the app. "
            "Get a free key at https://console.groq.com/keys"
        )

    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
    try:
        llm = ChatGroq(
            model=groq_model,
            temperature=0.3,
            max_tokens=1024,
            groq_api_key=api_key,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {e}")
        raise ValueError("Failed to connect to Groq API. Check your API key and internet connection.")

    qa_prompt = PromptTemplate(
        template="""You are DocQuery AI, an intelligent document assistant. Answer the user's question
based ONLY on the provided context from the uploaded document. Follow these rules strictly:

1. If the answer is found in the context, provide a clear, detailed answer.
2. Always mention which part of the document your answer comes from.
3. If the context does not contain enough information to answer, say:
   "I couldn't find this information in the uploaded document. Try rephrasing your question or upload a document that covers this topic."
4. Never make up information that isn't in the context.
5. Be concise but thorough.

Context from the document:
{context}

Chat History:
{chat_history}

User Question: {question}

Answer:""",
        input_variables=["context", "chat_history", "question"],
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=6,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ─── Background Processing ────────────────────────────────────────────────────

def _process_document_sync(
    session_id: str,
    file_path: str,
    cancel_event: threading.Event,
) -> Optional[tuple]:
    """CPU-heavy processing run in a thread pool. Returns (chain, pages, chunks) or None if cancelled."""

    logger.info(f"[{session_id}] Step 1: Extracting text...")
    pages = extract_text_from_pdf(file_path)
    if cancel_event.is_set():
        return None
    if not pages:
        raise ValueError("Could not extract text. The PDF may be scanned/image-based or empty.")
    logger.info(f"[{session_id}] Extracted {len(pages)} pages")

    logger.info(f"[{session_id}] Step 2: Chunking...")
    chunks = chunk_documents(pages)
    if cancel_event.is_set():
        return None
    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")
    logger.info(f"[{session_id}] Created {len(chunks)} chunks")

    logger.info(f"[{session_id}] Step 3: Generating embeddings for {len(chunks)} chunks...")
    vector_store = create_vector_store(chunks, session_id)
    if cancel_event.is_set():
        return None

    logger.info(f"[{session_id}] Step 4: Building QA chain...")
    qa_chain = build_qa_chain(vector_store)

    return qa_chain, len(pages), len(chunks)


async def _run_processing_job(
    session_id: str,
    file_path: Path,
    filename: str,
    cancel_event: threading.Event,
) -> None:
    """Async wrapper: runs sync processing in a thread and updates processing_jobs."""
    try:
        result = await asyncio.to_thread(
            _process_document_sync, session_id, str(file_path), cancel_event
        )

        if result is None:
            # Cancelled — clean up files
            processing_jobs[session_id]["status"] = "cancelled"
            file_path.unlink(missing_ok=True)
            sp = FAISS_DIR / session_id
            if sp.exists():
                shutil.rmtree(sp)
            logger.info(f"[{session_id}] Job cancelled and cleaned up")
            return

        qa_chain, total_pages, total_chunks = result
        sessions[session_id] = {
            "chain":        qa_chain,
            "filename":     filename,
            "total_pages":  total_pages,
            "total_chunks": total_chunks,
            "file_path":    str(file_path),
        }
        processing_jobs[session_id].update({
            "status":       "complete",
            "total_pages":  total_pages,
            "total_chunks": total_chunks,
            "message":      f"Successfully processed '{filename}' — {total_pages} pages, {total_chunks} chunks indexed.",
        })
        logger.info(f"[{session_id}] Job complete")

    except Exception as e:
        logger.error(f"[{session_id}] Job failed: {e}", exc_info=True)
        processing_jobs[session_id].update({"status": "error", "error": str(e)})
        file_path.unlink(missing_ok=True)


# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health_check():
    """Health check — also reports whether the Groq key is configured."""
    api_key = os.getenv("GROQ_API_KEY", "")
    groq_configured = bool(api_key) and api_key.strip() != "" and api_key != "your_groq_api_key_here"
    return {
        "status":              "healthy",
        "groq_api_configured": groq_configured,
        "active_sessions":     len(sessions),
        "embeddings_loaded":   _embeddings is not None,
    }


@app.post("/api/set-key")
async def set_api_key(request: SetKeyRequest):
    """Save the Groq API key to .env and inject it into the running process immediately."""
    key = request.api_key.strip()
    if not key.startswith("gsk_"):
        raise HTTPException(status_code=400, detail="Invalid key format. Groq API keys start with 'gsk_'.")

    # Persist to .env
    (BASE_DIR / ".env").write_text(
        f"# Get your free API key from https://console.groq.com/keys\nGROQ_API_KEY={key}\n",
        encoding="utf-8",
    )
    # Apply immediately — no restart needed
    os.environ["GROQ_API_KEY"] = key
    logger.info("GROQ_API_KEY updated via /api/set-key")
    return {"status": "ok", "message": "API key saved and active."}


@app.post("/api/upload", response_model=UploadJobResponse)
async def upload_document(file: UploadFile = File(...)):
    """Validate and save PDF, start background processing, return job_id immediately."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty.")
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File too large. Maximum size is 20MB.")

    if len(contents) > 5 * 1024 * 1024:
        logger.warning(f"Large file uploaded: {len(contents) / 1024 / 1024:.1f}MB")

    session_id  = str(uuid.uuid4())[:8]
    safe_name   = "".join(c for c in file.filename if c.isalnum() or c in "._- ")[:100]
    file_path   = UPLOAD_DIR / f"{session_id}_{safe_name}"

    try:
        file_path.write_bytes(contents)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save uploaded file.")

    cancel_event = threading.Event()
    processing_jobs[session_id] = {
        "status":       "processing",
        "cancel_event": cancel_event,
        "filename":     file.filename,
        "total_pages":  None,
        "total_chunks": None,
        "message":      None,
        "error":        None,
    }
    asyncio.create_task(_run_processing_job(session_id, file_path, file.filename, cancel_event))
    logger.info(f"Upload job {session_id} started for '{file.filename}'")

    return UploadJobResponse(job_id=session_id, status="processing", filename=file.filename)


@app.get("/api/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the status of a background processing job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    job = processing_jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        filename=job.get("filename"),
        total_pages=job.get("total_pages"),
        total_chunks=job.get("total_chunks"),
        message=job.get("message"),
        error=job.get("error"),
    )


@app.delete("/api/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Request cancellation of an in-progress upload job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    job = processing_jobs[job_id]
    if job["status"] == "processing":
        job["cancel_event"].set()
        job["status"] = "cancelling"
        logger.info(f"Cancellation requested for job {job_id}")
    return {"message": "Cancellation requested.", "job_id": job_id, "status": job["status"]}


@app.post("/api/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Ask a question about the uploaded document."""
    if request.session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found. Please upload a document first.",
        )

    chain = sessions[request.session_id]["chain"]

    try:
        result = await asyncio.to_thread(chain.invoke, {"question": request.question})

        sources = []
        seen_pages: set = set()
        for doc in result.get("source_documents", []):
            page_num = doc.metadata.get("page_number", "N/A")
            if page_num not in seen_pages:
                seen_pages.add(page_num)
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                sources.append({"page_number": page_num, "content_preview": preview})

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            session_id=request.session_id,
        )

    except Exception as e:
        logger.error(f"Error generating answer: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}",
        )


@app.get("/api/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get information about an active session."""
    if session_id not in sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    session = sessions[session_id]
    try:
        history_len = len(session["chain"].memory.chat_memory.messages) // 2
    except (AttributeError, TypeError):
        history_len = 0

    return SessionInfo(
        session_id=session_id,
        filename=session["filename"],
        total_pages=session["total_pages"],
        total_chunks=session["total_chunks"],
        chat_history_length=history_len,
    )


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a session and delete its associated files."""
    if session_id not in sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    session = sessions.pop(session_id)

    try:
        fp = Path(session["file_path"])
        if fp.exists():
            fp.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete uploaded file: {e}")

    try:
        sp = FAISS_DIR / session_id
        if sp.exists():
            shutil.rmtree(sp)
    except Exception as e:
        logger.warning(f"Failed to delete FAISS store: {e}")

    return {"message": "Session cleared successfully.", "session_id": session_id}


# ─── Run Server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=300,
    )
