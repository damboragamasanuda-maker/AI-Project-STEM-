User Guide – IKMS Multi-Agent RAG API (Feature 4)
Prerequisites
Before starting, ensure the following are installed on your system:
Python 3.10 or higher
pip (Python package manager)
Git
A modern web browser (Chrome, Edge, Firefox)
Note: This project is a backend API demonstration built using FastAPI for the IKMS Feature 4 requirement.
Step 1: Clone the Project
Clone the GitHub repository and navigate into the project folder:
git clone https://github.com/damboragamasanuda-maker/AI-Project-STEM-.git
cd class-12-clean
Step 2: Create and Activate a Virtual Environment
A virtual environment is required to isolate project dependencies.
macOS / Linux
python -m venv .venv
source .venv/bin/activate
Windows
python -m venv .venv
.venv\Scripts\activate
After activation, your terminal should show:
(.venv)
Step 3: Install Dependencies
Install all required Python packages using:
pip install -r requirements.txt
Wait until installation completes successfully.
Step 4: Project Structure Overview
The project is organised as follows:
class-12-clean/
│
├── api/
│   └── index.py            # FastAPI entry point for deployment
│
├── src/
│   └── app/
│       ├── api.py          # Main FastAPI application
│       ├── models.py       # Request and response schemas
│       ├── services/       # Indexing services
│       └── core/           # Core utilities
│
├── data/                   # Uploaded PDF files
├── .venv/                  # Virtual environment
├── requirements.txt
├── README.md
└── USER_GUIDE.md
Step 5: Run the Application Locally
Start the FastAPI development server using Uvicorn:
uvicorn src.app.api:app --reload --port 8000
If successful, you will see:
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
The backend API is now running at:
http://localhost:8000
Step 6: Access the API Interface (Website)
Open a browser and navigate to:
http://localhost:8000/docs
This opens the Swagger UI, which serves as the interactive web interface for the system.
Step 7: Using the Application
1. Upload and Index a PDF
Endpoint:
POST /index-pdf
Steps:
Click on /index-pdf
Click Try it out
Upload a PDF file
Execute the request
The document will be stored and indexed successfully.
2. Ask a Question
Endpoint:
POST /qa
Steps:
Click on /qa
Click Try it out
Enter a question in JSON format:
{
  "question": "What is this document about?"
}
Execute the request
Note:
This endpoint currently returns a placeholder response, as required for IKMS Feature 4. The multi-agent RAG logic will be implemented in future iterations.
Step 8: Stop the Server
To stop the application, press:
CTRL + C
in the terminal.
Troubleshooting
Error: ModuleNotFoundError
Ensure the virtual environment is activated
Ensure the command is run from the project root directory
Correct command:
uvicorn src.app.api:app --reload
Error: Port Already in Use
Run the server on a different port:
uvicorn src.app.api:app --reload --port 8080
Error: Swagger Page Not Loading
Confirm the server is running
Open http://localhost:8000/docs in a browser
Check terminal logs for errors
Quick Start Summary
git clone <repository-url>
cd class-12-clean
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.app.api:app --reload
Open in browser:
http://localhost:8000/docs
Conclusion
This system demonstrates the IKMS Feature 4 – Multi-Agent RAG API architecture using FastAPI.
It validates document ingestion, structured API design, and future extensibility for intelligent retrieval workflows.

Deployment attempted on Railway & Vercel.
Build failed due to FastAPI entrypoint constraints.
Project is fully runnable locally (see UserGuide.md).
