# Установка и конфигурация

Python 3.11, Windows PowerShell, Ollama с HTTP-сервером 127.0.0.1:11434.
Установка зависимостей и первая загрузка моделей требуют сети. Сервер
Ollama и модели могут быть установлены обычным способом вне репозитория.
Скрипт также поддерживает `.runtime/ollama/ollama.exe` внутри репозитория
или путь из переменной ENERGYTOOL_OLLAMA_EXE. Прежний project не требуется.

Если используете переносную Ollama в произвольной папке, перед запуском
задайте в текущем PowerShell ENERGYTOOL_OLLAMA_EXE (полный путь к ollama.exe)
и при необходимости OLLAMA_MODELS (папку уже загруженных моделей). Эти
машинные пути не нужно добавлять в исходники. При уже работающем сервере
достаточно его адреса; Ollama остаётся отдельной внешней зависимостью.

Команды создания окружения и первого запуска приведены в ../README.md.
Для уже запущенного сервера CLI Ollama не обязателен, кроме режима -Pull.
Проверка сервера: `Invoke-RestMethod http://127.0.0.1:11434/api/tags`.
Модели: `qwen3:4b-instruct-2507-q4_K_M`, `bge-m3`. Старый thinking-only alias
qwen3:4b нельзя без проверки подставлять вместо указанной Instruct-версии.
GPU не обязателен; на CPU время ответа будет отличаться.

## Основные параметры .env в energytool_hackathon/app

| Параметр | Назначение |
| --- | --- |
| LLM_PROVIDER=ollama | Локальная генерация |
| OLLAMA_BASE_URL | Адрес сервера Ollama |
| OLLAMA_MODEL, OLLAMA_THINK=false | Модель и режим Instruct |
| RETRIEVER_MODE=hybrid-ollama | BM25 + локальные embeddings |
| OLLAMA_EMB_MODEL=bge-m3 | Эмбеддер |
| TOP_K=8, CANDIDATE_K=30 | Объём итоговой выдачи и пула кандидатов |
| BM25_WEIGHT, DENSE_WEIGHT, RRF_K | Объединение результатов |
| RERANKER_ENABLED=0 | Реранкер не включён в проверенный основной профиль |
| GROUNDED_ANSWERS=1 | Отбор цитат и проверка ответа |
| SOURCE_FIRST_RESPONSES=1 | Ответы по структуре источников для поддерживаемых задач |
| SESSION_DIR | Отдельное постоянное хранилище сессий; по умолчанию app/sessions |
| LLM_TIMEOUT=180 | Таймаут локальной модели |

При ошибке embeddings поиск откатывается на BM25. Ошибки LLM показываются
как error; их нельзя считать отсутствием ответа в базе.

Для API задайте LLM_PROVIDER=openai, LLM_BASE_URL, LLM_API_KEY и имя модели
из окружения платформы (LLM_MODEL/GEN_MODEL/OPENAI_MODEL). Не публикуйте .env.
Для Linux ядро запускается Python из app; скрипты PowerShell — удобство Windows.

## Платформенный сабмит

Из app: `python -m tools.build_submission`. Результат — dist/platform.zip,
21 корневой файл. Это отдельный артефакт для автоматической проверки, а не
GitHub-репозиторий. Архив строится без .env, базы, моделей, индексов и тестов;
база и questions.csv поступают через DATA_DIR при запуске платформой.
Локальная сборка не подтверждает качество и время платформенной Qwen3.8-27B.

## Загрузка в GitHub

В папке уже инициализирован локальный Git-репозиторий main, без коммитов
и удалённого адреса. После просмотра файлов выполните из его корня:

```powershell
git add .
git diff --cached --stat
git diff --cached
git commit -m "Добавить ядро RAG-консультанта EnergyTool"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

OWNER/REPOSITORY замените адресом своего репозитория. Эти команды публикации
пока не выполнялись. Для GitHub Desktop добавьте существующий локальный
репозиторий. При ручной загрузке файлов через сайт .gitignore не фильтрует
выбранные файлы: не отправляйте окружение, .env, кеши, модели и сессии.
