"""Persistent JSON storage for chat sessions and generated leads."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config


class ConversationState:
    """A single conversation that is persisted after every state change."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.dir = Path(config.SESSION_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.session_id}.json"
        now = self._now()
        self.data = {
            "session_id": self.session_id,
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "lead": None,
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def load(self) -> "ConversationState":
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            self.data.setdefault("messages", [])
            self.data.setdefault("lead", None)
        return self

    def add(self, role: str, content: str, meta: dict | None = None) -> None:
        item = {"role": role, "content": content, "at": self._now()}
        if meta:
            item["meta"] = meta
        self.data["messages"].append(item)
        self._touch_and_save()

    def set_lead(self, lead: dict | None) -> None:
        self.data["lead"] = lead
        self._touch_and_save()

    def save(self) -> None:
        """Write atomically so an interrupted process does not corrupt history."""
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)

    def _touch_and_save(self) -> None:
        self.data["updated_at"] = self._now()
        self.save()

    def history(self) -> list[dict]:
        return [
            {"role": message["role"], "content": message["content"]}
            for message in self.data["messages"]
        ]

    def reset(self) -> "ConversationState":
        """Start a clean session while preserving the previous JSON file."""
        new_state = ConversationState()
        new_state.save()
        return new_state

    @classmethod
    def list_sessions(cls, limit: int = 100) -> list[dict]:
        """Return newest session summaries without exposing filesystem paths."""
        directory = Path(config.SESSION_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        sessions = []
        for path in directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            messages = data.get("messages", [])
            first_user = next(
                (m.get("content", "") for m in messages if m.get("role") == "user"),
                "Новый диалог",
            )
            sessions.append(
                {
                    "session_id": data.get("session_id", path.stem),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "title": first_user[:80],
                    "message_count": len(messages),
                    "lead_status": (data.get("lead") or {}).get("status"),
                }
            )
        sessions.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return sessions[: max(1, min(limit, 500))]
