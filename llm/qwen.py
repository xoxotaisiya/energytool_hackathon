"""Клиент настоящей языковой модели через Ollama или OpenAI-compatible API."""

import json

import httpx

import config


class LLMError(RuntimeError):
    """Ошибка подключения к языковой модели или обработки её ответа."""


class LLMClient:
    """Отправляет запросы настроенной языковой модели."""

    def __init__(self):
        self.client = httpx.Client(timeout=config.LLM_TIMEOUT)

    def check_connection(self):
        """Проверяет доступность сервиса и наличие выбранной модели."""
        try:
            if config.LLM_PROVIDER == "ollama":
                response = self.client.get(config.OLLAMA_BASE_URL + "/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                names = {item.get("name", "") for item in models}
                short_names = {name.split(":")[0] for name in names}
                if config.OLLAMA_MODEL not in names and config.OLLAMA_MODEL.split(":")[0] not in short_names:
                    raise LLMError(
                        f"Модель {config.OLLAMA_MODEL} не найдена в Ollama. "
                        f"Установите её командой: ollama pull {config.OLLAMA_MODEL}"
                    )
                return True
            if config.LLM_PROVIDER == "openai":
                if not config.LLM_BASE_URL:
                    raise LLMError("Для LLM_PROVIDER=openai задайте LLM_BASE_URL.")
                return True
            raise LLMError("Неизвестный LLM_PROVIDER: " + config.LLM_PROVIDER)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Не удалось подключиться к языковой модели: {exc}. "
                "Убедитесь, что Ollama запущена."
            ) from exc

    def chat(self, messages, json_mode=False):
        """Отправляет сообщения модели и возвращает её текстовый ответ."""
        try:
            if config.LLM_PROVIDER == "ollama":
                payload = {
                    "model": config.OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": config.LLM_TEMPERATURE},
                }
                if json_mode:
                    payload["format"] = "json"
                response = self.client.post(
                    config.OLLAMA_BASE_URL + "/api/chat", json=payload
                )
                response.raise_for_status()
                content = response.json().get("message", {}).get("content", "")
            elif config.LLM_PROVIDER == "openai":
                if not config.LLM_BASE_URL:
                    raise LLMError("Для LLM_PROVIDER=openai задайте LLM_BASE_URL.")
                headers = (
                    {"Authorization": "Bearer " + config.LLM_API_KEY}
                    if config.LLM_API_KEY else {}
                )
                payload = {
                    "model": config.LLM_MODEL,
                    "messages": messages,
                    "temperature": config.LLM_TEMPERATURE,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                response = self.client.post(
                    config.LLM_BASE_URL + "/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
            else:
                raise LLMError("Неизвестный LLM_PROVIDER: " + config.LLM_PROVIDER)

            if not isinstance(content, str) or not content.strip():
                raise LLMError("Модель вернула пустой ответ.")
            return content.strip()
        except LLMError:
            raise
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(
                f"Ошибка обращения к LLM: {type(exc).__name__}: {exc}"
            ) from exc

    def chat_json(self, messages):
        """Получает JSON-ответ модели и проверяет его синтаксис."""
        raw = self.chat(messages, json_mode=True)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                raise LLMError("Модель вернула ответ не в формате JSON.")
            try:
                result = json.loads(raw[start:end + 1])
            except json.JSONDecodeError as exc:
                raise LLMError("Модель вернула некорректный JSON.") from exc
        if not isinstance(result, dict):
            raise LLMError("JSON-ответ модели должен быть объектом.")
        return result

    def close(self):
        """Закрывает HTTP-соединения."""
        self.client.close()
