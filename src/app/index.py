from fastapi import FastAPI
from src.app.api import app as internal_app

app = internal_app
