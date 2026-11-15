"""Настройки приложения. Секреты читаются только из окружения/.env."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def env(*names, default=""):
    for name in names:
        value = os.getenv(name, "").strip()
        if value and value != name:
            return value
    return default


def number(names, default, lower, upper, kind=float):
    value = kind(env(*names, default=str(default)))
    if not lower <= value <= upper:
        raise ValueError(f"{names[0]} должен быть в диапазоне {lower}..{upper}")
    return value


LLM_BASE_URL = env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/")
LLM_API_KEY = env("LLM_API_KEY", "OPENAI_API_KEY")
LLM_PROVIDER = env("LLM_PROVIDER", default="openai" if LLM_BASE_URL else "ollama").lower()
if LLM_PROVIDER not in {"ollama", "openai"}:
    raise ValueError("LLM_PROVIDER должен быть ollama или openai")
OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", default="http://localhost:11434").rstrip("/")
OLLAMA_MODEL = env("OLLAMA_MODEL", default="qwen3:4b-instruct-2507-q4_K_M")
OLLAMA_THINK = env("OLLAMA_THINK", default="false").lower() == "true"
OLLAMA_NUM_CTX = number(("OLLAMA_NUM_CTX",), 8192, 2048, 131072, int)
LLM_MODEL = env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL", default="qwen/qwen3.8-27b")
LLM_TEMPERATURE = number(("TEMPERATURE", "LLM_TEMPERATURE"), 0.1, 0, 2)
MAX_TOKENS = number(("MAX_TOKENS",), 2048, 128, 8192, int)
LLM_TIMEOUT = number(("LLM_TIMEOUT",), 45, 1, 300)
LLM_RETRIES = number(("LLM_RETRIES",), 1, 0, 3, int)
CONTEXT_MAX_CHARS = number(("CONTEXT_MAX_CHARS",), 18000, 1000, 100000, int)
KB_PATH = env("KB_PATH")
RETRIEVER_MODE = env("RETRIEVER_MODE", default="auto").lower()
SESSION_DIR = env("SESSION_DIR", default=str(ROOT / "sessions"))
DEBUG = env("ENERGYTOOL_DEBUG", default="0").lower() in {"1", "true", "yes", "on"}
GROUNDED_ANSWERS = env("GROUNDED_ANSWERS", default="1").lower() in {"1", "true", "yes", "on"}
SOURCE_FIRST_RESPONSES = env("SOURCE_FIRST_RESPONSES", default="1").lower() in {"1", "true", "yes", "on"}
