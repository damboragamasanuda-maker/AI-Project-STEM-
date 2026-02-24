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
from .services.qa_service import answer_question

BASE_DIR = Path(__file__).resolve().parents[2]  # project root if src/app/api.py


app = FastAPI(
    title="IKMS Spark",
    description="Colorful AI PDF Q&A • Upload → Index → Ask → Get Answers",
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
    if isinstance(exc, HTTPException):
        raise exc

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

    try:
        result = answer_question(question)  # <-- your multi-agent RAG flow

        # Make it robust even if the pipeline returns slightly different keys
        answer = result.get("answer") or result.get("final_answer") or str(result)
        context = result.get("context") or result.get("sources") or ""
        citations = result.get("citations") or {}

        return QAResponse(answer=answer, context=context, citations=citations)

    except Exception as e:
        print("QA_ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")


@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict:
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    try:
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