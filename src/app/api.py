from pathlib import Path
import os
import tempfile
import traceback

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .services.qa_service import answer_question

from .models import QuestionRequest, QAResponse
from .services.indexing_service import index_pdf_file

BASE_DIR = Path(__file__).resolve().parents[2]  # project root if src/app/api.py

# FastAPI backend for the application
app = FastAPI(
    title="IKMS Spark",
    description="Upload a PDF, index it, then ask questions (RAG).",
    version="0.1.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# Serve frontend assets
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
    # Helpful for Railway debugging
    return {
        "status": "ok",
        "PORT": os.getenv("PORT"),
        "tmp_dir": tempfile.gettempdir(),
        "BASE_DIR": str(BASE_DIR),
        "static_dir_exists": (BASE_DIR / "static").exists(),
        "templates_dir_exists": (BASE_DIR / "templates").exists(),
        "has_OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "has_PINECONE_API_KEY": bool(os.getenv("PINECONE_API_KEY")),
        "has_PINECONE_INDEX_NAME": bool(os.getenv("PINECONE_INDEX_NAME")),
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, HTTPException):
        raise exc

    print("UNHANDLED_ERROR:", repr(exc))
    traceback.print_exc()

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# Endpoint for uploading and indexing PDFs
@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    try:
        # Railway-safe writable temp directory
        # Saves uploaded PDF to a temporary directory before processing
        upload_dir = Path(tempfile.gettempdir()) / "ikms_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_name = Path(file.filename or "upload.pdf").name
        file_path = upload_dir / safe_name

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_path.write_bytes(contents)

        # Index the PDF into the vector database
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
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

# Endpoint for asking questions
@app.post("/qa", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: QuestionRequest) -> QAResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`question` must be a non-empty string.",
        )

    try:
        result = answer_question(question) # Runs the multi-agent RAG pipeline

# Returns answer, retrieved context, and citations
        return QAResponse(
            answer=str(result.get("answer", "")) or "No answer generated.",
            context=str(result.get("context", "")) or "",
            citations=result.get("citations", {}) or {},
        )

    except Exception as e:
        # show real error in Railway logs
        print("QA_ERROR:", repr(e))
        traceback.print_exc()

        # return user-friendly error without crashing the app
        return QAResponse(
            answer=f"❌ Error: QA failed: {str(e)}",
            context="",
            citations={},
        )