How the IKMS API System Works

(Feature 4 – Multi-Agent RAG API Demo)
Table of Contents
Tech Stack
The Big Picture
Part 1: Uploading a Document
Part 2: Asking a Question
Part 3: Planned Multi-Agent Design
Part 4: API Execution Flow

Summary
Tech Stack
The IKMS API System is built using modern backend technologies focused on clarity, scalability, and future AI integration.
Backend Technologies
Python (≥3.10) – Core programming language
FastAPI – High-performance API framework
Uvicorn – ASGI server for running the API
Pydantic – Request and response validation
PyPDF – PDF document parsing
Pathlib – File handling and directory management
Why These Technologies?
FastAPI provides fast performance and automatic API documentation
Pydantic ensures clean and validated data flow
Service-based structure allows easy expansion to AI pipelines
Python enables future LLM and vector database integration
The Big Picture
System Flow
User Uploads PDF
      ↓
PDF is Stored & Indexed
      ↓
User Asks Question
      ↓
API Validates Request
      ↓
Placeholder Answer Returned
This implementation demonstrates API readiness and document handling, which is the focus of IKMS Feature 4.
Part 1: Uploading a Document
What Happens Step-by-Step
Step 1: User Uploads a PDF
The user uploads a PDF through the API endpoint:
POST /index-pdf
Step 2: System Saves the File
The uploaded file is validated
Only PDF files are accepted
The file is stored in the local directory:
data/uploads/
Step 3: Document Indexing
The PDF file is passed to the indexing service, which:
Reads the document
Prepares it for future vector-based indexing
Returns metadata such as number of chunks indexed
At Feature 4 stage, indexing logic validates the workflow rather than performing full semantic search.
Part 2: Asking a Question
Example Question
"What is this document about?"
Step-by-Step Flow
Step 1: Question Sent to API
The user sends a request to:
POST /qa
With JSON input:
{
  "question": "What is this document about?"
}
Step 2: Input Validation
The system checks if the question is empty
Invalid requests are rejected with HTTP 400 errors
Step 3: Placeholder Response Returned
Since Feature 4 focuses on API structure, the system returns:
A placeholder answer
A context message
An empty citations object
This confirms the question-answering pipeline is operational.
Part 3: Planned Multi-Agent Design
Although not fully implemented yet, the system is designed to support a Multi-Agent RAG Architecture.
Planned Agents
Agent 1: Retrieval Agent
Finds relevant document sections
Agent 2: Answer Generation Agent
Generates an answer based on retrieved content
Agent 3: Verification Agent
Ensures answers are grounded in document content
Agent 4: Memory Agent
Maintains conversation context
Feature 4 focuses on API readiness, not full AI logic.
Part 4: API Execution Flow
API Entry Point
api/index.py
This file exposes the FastAPI application for execution and deployment.
Main Application Logic
src/app/api.py
Responsibilities:
Defines API routes
Handles validation
Manages file uploads
Returns structured responses
Service Layer
src/app/services/
Purpose:
Keeps business logic separate from API routes
Enables easy expansion of AI-based services later
Summary
This implementation demonstrates:
A clean FastAPI architecture
Document upload and indexing workflow
Structured question-answering API
Clear separation of concerns
Readiness for Multi-Agent RAG integration
The project successfully fulfills IKMS Feature 4 by validating the system design, API exposure, and future extensibility.

References
Author: Sanuda Damboragama
Project: IKMS Multi-Agent RAG API
Feature: Feature 4
Date: 2026-01-22