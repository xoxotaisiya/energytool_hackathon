"""Простое локальное хранение истории сессий в JSON."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import config


class ConversationState:
    def __init__(self, session_id=None):
        """Создает локальную сессию диалога."""
        self.session_id = session_id or str(uuid.uuid4())
        self.dir = Path(config.SESSION_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.session_id}.json"
        self.data = {"session_id": self.session_id, "created_at": self._now(),
                     "updated_at": self._now(), "messages": [], "lead": None}

    @staticmethod
    def _now():
        """Возвращает текущее время в формате ISO."""
        return datetime.now(timezone.utc).isoformat()

    def load(self):
        """Загружает сохраненную историю сессии."""
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        return self

    def add(self, role, content, meta=None):
        """Добавляет сообщение в историю и сохраняет ее."""
        item = {"role": role, "content": content, "at": self._now()}
        if meta:
            item["meta"] = meta
        self.data["messages"].append(item)
        self.data["updated_at"] = self._now()
        self.save()

    def save(self):
        """Записывает историю сессии в JSON."""
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def history(self):
        """Возвращает сообщения в формате для LLM."""
        return [{"role": m["role"], "content": m["content"]} for m in self.data["messages"]]

    def reset(self):
        """Создает новую сессию, не удаляя старую."""
        # Новый диалог — новая сессия; старый JSON остаётся для аудита.
        return ConversationState()
