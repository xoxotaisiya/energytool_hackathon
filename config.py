"""Настройки Energytool AI Consultant. Секреты задаются через переменные окружения."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # ollama | openai
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")  # OpenAI-compatible URL, например https://host/v1
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
KB_PATH = os.getenv("KB_PATH", "")
RETRIEVER_MODE = os.getenv("RETRIEVER_MODE", "bm25")
SESSION_DIR = os.getenv("SESSION_DIR", "sessions")
