from pathlib import Path
import tempfile
import traceback

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .models import QuestionRequest, QAResponse
from .services.indexing_service import index_pdf_file

BASE_DIR = Path(__file__).resolve().parents[2]  # project root if src/app/api.py


app = FastAPI(
    title="IKMS Spark",
    description=(
        "Demo API for asking questions about a vector databases paper. "
        "The `/qa` endpoint currently returns placeholder responses and "
        "will be wired to a multi-agent RAG pipeline in later user stories."
    ),
    version="0.1.0",
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


@app.post("/qa", response_model=QAResponse, status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: QuestionRequest) -> QAResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`question` must be a non-empty string.",
        )

    return QAResponse(
        answer=(
            "This is a placeholder response. "
            "The multi-agent RAG pipeline will be integrated in a future iteration."
        ),
        context="Document ingestion and API exposure validated successfully.",
        citations={},
    )


@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    try:
        # ✅ Use Railway-safe writable temp directory
        upload_dir = Path(tempfile.gettempdir()) / "ikms_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # ✅ sanitize filename (avoid path tricks / empty names)
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
        # ✅ show real error in Railway logs + return readable message
        print("INDEX_PDF_ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}",
        )