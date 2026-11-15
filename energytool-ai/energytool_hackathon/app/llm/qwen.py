"""HTTP-клиент Ollama/OpenAI-compatible с ограниченными повторами."""
import json
import time
from contextvars import ContextVar
import httpx
import config


class LLMError(RuntimeError):
    def __init__(self, message, code="llm_error"):
        super().__init__(message)
        self.code = code


class LLMClient:
    def __init__(self, client=None):
        self.client = client or httpx.Client(timeout=config.LLM_TIMEOUT)
        self._metrics = ContextVar("energytool_llm_metrics", default=None)

    def reset_metrics(self):
        self._metrics.set([])

    def metrics(self):
        return list(self._metrics.get() or [])

    def _request(self, method, url, **kwargs):
        for attempt in range(config.LLM_RETRIES + 1):
            try:
                response = self.client.request(method, url, **kwargs)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < config.LLM_RETRIES:
                        time.sleep(0.25 * (attempt + 1))
                        continue
                response.raise_for_status()
                return response
            except httpx.TransportError:
                if attempt < config.LLM_RETRIES:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise LLMError("Сервис модели недоступен или превышено время ожидания.", "transport") from None
            except httpx.HTTPStatusError as exc:
                raise LLMError(f"Сервис модели вернул HTTP {exc.response.status_code}.", "http_status") from None

    def _headers(self):
        return {"Authorization": "Bearer " + config.LLM_API_KEY} if config.LLM_API_KEY else {}

    @staticmethod
    def _json(response):
        try:
            data = response.json()
        except ValueError:
            raise LLMError("Сервис модели вернул некорректный JSON.", "response_format") from None
        if not isinstance(data, dict):
            raise LLMError("Ответ API должен быть JSON-объектом.", "response_format")
        return data

    def check_connection(self):
        if config.LLM_PROVIDER == "ollama":
            data = self._json(self._request("GET", config.OLLAMA_BASE_URL + "/api/tags"))
            models = data.get("models")
            if not isinstance(models, list):
                raise LLMError("В ответе Ollama отсутствует список моделей.", "response_format")
            names = {m.get("name") for m in models if isinstance(m, dict)}
            wanted = config.OLLAMA_MODEL
            aliases = {wanted} if ":" in wanted else {wanted, wanted + ":latest"}
            if not names.intersection(aliases):
                raise LLMError(f"Модель {wanted} не установлена в Ollama.", "model_missing")
            info = self._json(self._request("POST", config.OLLAMA_BASE_URL + "/api/show", json={"model": wanted}))
            controls = info.get("thinking", {})
            supported = controls.get("values") if isinstance(controls, dict) else None
            if isinstance(supported, list) and supported and config.OLLAMA_THINK not in supported:
                raise LLMError("OLLAMA_THINK несовместим с установленной версией модели. Проверьте thinking.values в /api/show.", "thinking_mode")
        else:
            if not config.LLM_BASE_URL:
                raise LLMError("Не задан адрес OpenAI-compatible сервиса.", "configuration")
            self.chat([{"role": "user", "content": "Ответь одним словом: готов."}])
        return True

    def chat(self, messages, json_mode=False, schema=None):
        started = time.perf_counter()
        if config.LLM_PROVIDER == "ollama":
            payload = {"model": config.OLLAMA_MODEL, "messages": messages, "stream": False,
                       "think": config.OLLAMA_THINK,
                       "options": {"temperature": config.LLM_TEMPERATURE, "num_predict": config.MAX_TOKENS,
                                   "num_ctx": config.OLLAMA_NUM_CTX}}
            if json_mode:
                payload["format"] = schema if schema is not None else "json"
            data = self._json(self._request("POST", config.OLLAMA_BASE_URL + "/api/chat", json=payload))
            message = data.get("message")
            finish = data.get("done_reason")
        else:
            if not config.LLM_BASE_URL:
                raise LLMError("Не задан адрес OpenAI-compatible сервиса.", "configuration")
            payload = {"model": config.LLM_MODEL, "messages": messages,
                       "temperature": config.LLM_TEMPERATURE, "max_tokens": config.MAX_TOKENS}
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            data = self._json(self._request("POST", config.LLM_BASE_URL + "/chat/completions",
                                           headers=self._headers(), json=payload))
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise LLMError("В ответе API отсутствует choices[0].", "response_format")
            message = choices[0].get("message")
            finish = choices[0].get("finish_reason")
        metric = {"elapsed_ms": round((time.perf_counter() - started) * 1000), "finish_reason": finish}
        if config.LLM_PROVIDER == "ollama":
            for key in ("load_duration", "prompt_eval_duration", "eval_duration", "total_duration"):
                if isinstance(data.get(key), (int, float)):
                    metric[key + "_ms"] = round(data[key] / 1_000_000, 2)
            for key in ("prompt_eval_count", "eval_count"):
                if isinstance(data.get(key), int):
                    metric[key] = data[key]
        elif isinstance(data.get("usage"), dict):
            metric["usage"] = {k: v for k, v in data["usage"].items() if k in {"prompt_tokens", "completion_tokens", "total_tokens"} and isinstance(v, int)}
        self._metrics.set(self.metrics() + [metric])
        if finish in {"length", "max_tokens"}:
            raise LLMError("Модель исчерпала лимит генерации. Увеличьте MAX_TOKENS.", "token_limit")
        if not isinstance(message, dict):
            raise LLMError("В ответе API отсутствует объект message.", "response_format")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Модель вернула пустой content.", "empty_content")
        return content.strip()

    def chat_json(self, messages, schema=None):
        raw = self.chat(messages, json_mode=True, schema=schema)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            if raw.startswith("```") and raw.endswith("```"):
                try:
                    result = json.loads("\n".join(raw.splitlines()[1:-1]))
                except json.JSONDecodeError:
                    raise LLMError("Модель вернула некорректный JSON.", "json_format") from None
            else:
                raise LLMError("Модель вернула ответ не в формате JSON.", "json_format") from None
        if not isinstance(result, dict):
            raise LLMError("JSON-ответ модели должен быть объектом.", "json_format")
        return result

    def chat_structured(self, messages, schema):
        # Ollama поддерживает ограниченную генерацию JSON. У прокси организаторов
        # достаточно обычного json_object + промпта; алгоритм всегда валидирует
        # тот же контракт сам и не требует поддержки JSON Schema от прокси.
        return self.chat_json(messages, schema=schema)

    def close(self):
        self.client.close()
