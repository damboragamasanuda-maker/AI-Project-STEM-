from pathlib import Path
import tempfile
import traceback
import os

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .models import QuestionRequest, QAResponse
from .services.indexing_service import index_pdf_file

# ✅ settings (you already have this in src/app/core/config.py)
from .core.config import get_settings

# ✅ OpenAI + Pinecone
from openai import OpenAI
from pinecone import Pinecone

BASE_DIR = Path(__file__).resolve().parents[2]  # project root if src/app/api.py


app = FastAPI(
    title="IKMS Spark",
    description=(
        "Colorful AI PDF Q&A. Upload a PDF, index it into Pinecone, then ask questions using OpenAI."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# ✅ Serve frontend assets
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    # Useful for Railway debugging
    return {
        "status": "ok",
        "port_env": os.getenv("PORT"),
        "tmp_dir": tempfile.gettempdir(),
        "base_dir": str(BASE_DIR),
        "static_dir": str(BASE_DIR / "static"),
        "templates_dir": str(BASE_DIR / "templates"),
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Let FastAPI HTTPExceptions pass through
    if isinstance(exc, HTTPException):
        raise exc

    # Log the real error in Railway logs
    print("UNHANDLED_ERROR:", repr(exc))
    traceback.print_exc()

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# -------------------------
# RAG helpers
# -------------------------
def _get_openai_client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.openai_api_key)


def _get_pinecone_index():
    s = get_settings()
    pc = Pinecone(api_key=s.pinecone_api_key)
    return pc.Index(s.pinecone_index_name)


def _embed_text(text: str) -> list[float]:
    s = get_settings()
    client = _get_openai_client()
    emb = client.embeddings.create(
        model=s.openai_embedding_model_name,
        input=text,
    )
    return emb.data[0].embedding


def _retrieve_context(question: str, top_k: int = 4) -> list[dict]:
    """
    Retrieve top_k relevant chunks from Pinecone.

    Assumes indexing stored chunk text in:
      metadata["text"]   (preferred)
      metadata["chunk"]  (fallback)
    """
    idx = _get_pinecone_index()
    q_vec = _embed_text(question)

    res = idx.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True,
    )

    # pinecone client may return object-like or dict-like response
    matches = None
    if isinstance(res, dict):
        matches = res.get("matches", [])
    else:
        matches = getattr(res, "matches", []) or []

    contexts: list[dict] = []
    for m in matches:
        if isinstance(m, dict):
            score = m.get("score")
            md = m.get("metadata", {}) or {}
        else:
            score = getattr(m, "score", None)
            md = getattr(m, "metadata", {}) or {}

        text = md.get("text") or md.get("chunk") or md.get("content") or ""
        if text:
            contexts.append(
                {
                    "score": score,
                    "text": text,
                    "metadata": md,
                }
            )

    return contexts


# -------------------------
# QA endpoint (REAL)
# -------------------------
@app.post("/qa", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: QuestionRequest) -> QAResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`question` must be a non-empty string.",
        )

    s = get_settings()

    # 1) Retrieve
    contexts = _retrieve_context(question, top_k=s.retrieval_k)

    if not contexts:
        return QAResponse(
            answer=(
                "I couldn't find anything relevant in the index. "
                "Please upload/index a PDF first (and make sure indexing produced chunks)."
            ),
            context="No matches found in Pinecone.",
            citations={},
        )

    joined_context = "\n\n---\n\n".join([c["text"] for c in contexts])

    # 2) Generate answer using OpenAI
    client = _get_openai_client()

    system_msg = (
        "You are a helpful assistant. Answer ONLY using the provided CONTEXT. "
        "If the answer is not in the context, say: 'I don't know based on the document.'"
    )

    resp = client.chat.completions.create(
        model=s.openai_model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"CONTEXT:\n{joined_context}\n\nQUESTION:\n{question}"},
        ],
        temperature=0.2,
    )

    answer = (resp.choices[0].message.content or "").strip()

    citations = {
        "chunks": [
            {
                "score": c["score"],
                "metadata": c["metadata"],
            }
            for c in contexts
        ]
    }

    return QAResponse(
        answer=answer,
        context="Answer generated from top Pinecone matches.",
        citations=citations,
    )


# -------------------------
# Index PDF endpoint
# -------------------------
@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    try:
        # ✅ Railway-safe writable temp directory
        upload_dir = Path(tempfile.gettempdir()) / "ikms_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(file.filename or "upload.pdf").name
        file_path = upload_dir / safe_name

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_path.write_bytes(contents)

        chunks_indexed = index_pdf_file(file_path)

        return {
            "filename": safe_name,
            "chunks_indexed": chunks_indexed,
            "message": "PDF indexed successfully.",
        }

    except HTTPException:
        raise

    except Exception as e:
        print("INDEX_PDF_ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}",
        )