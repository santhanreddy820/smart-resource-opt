# src/api_app.py
# Lightweight wrapper so uvicorn can import src.api_app:app
from .api import app  # import the FastAPI 'app' from src/api.py
