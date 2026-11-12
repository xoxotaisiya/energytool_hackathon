#!/usr/bin/env python3
"""Собирает и проверяет плоский ZIP-сабмит для автоматической оценки."""

from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
ARCHIVE = DIST / "energytool_submission.zip"
FILES = {
    ROOT / "solution.py": "solution.py",
    ROOT / "requirements-submission.txt": "requirements.txt",
    ROOT / "rag" / "build_chunks.py": "build_chunks.py",
    ROOT / "rag" / "retriever.py": "retriever.py",
    ROOT / "rag" / "retrieval.py": "retrieval.py",
    ROOT / "rag" / "embedder.py": "embedder.py",
}


def build_archive(destination: Path = ARCHIVE) -> Path:
    """Создаёт архив без внешней папки, данных и секретов."""
    missing = [str(path) for path in FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Не найдены файлы сабмита: " + ", ".join(missing))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, arcname in FILES.items():
            archive.write(source, arcname)
    temporary.replace(destination)
    validate_archive(destination)
    return destination


def validate_archive(path: Path) -> None:
    """Проверяет структуру, размер и обязательные файлы ZIP."""
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("ZIP превышает лимит 128 МБ")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any("/" in name or "\\" in name for name in names):
            raise ValueError("Все файлы ZIP должны находиться в корне")
        if not {"solution.py", "requirements.txt"}.issubset(names):
            raise ValueError("В ZIP отсутствует solution.py или requirements.txt")
        if len(names) != len(set(names)):
            raise ValueError("В ZIP есть повторяющиеся имена файлов")


def main() -> None:
    archive = build_archive()
    size_kb = archive.stat().st_size / 1024
    print(f"Создан {archive} ({size_kb:.1f} КиБ)")


if __name__ == "__main__":
    main()
