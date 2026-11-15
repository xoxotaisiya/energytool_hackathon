# Реализация пунктов 1–3

## 1. База знаний

В knowledge-base находятся 19 Markdown-документов. 00-reference содержит
правила области компетенции, ограничений достоверности и подготовки расчётов;
10-energy — устройство рынка, составляющие счёта, тарифы и FAQ;
20-products — портфель и пять направлений: облачный учёт (askue), солнечные
станции (solar), выход на оптовый рынок (orem), коммерческие потери (losses)
и распознавание показаний по фото (meter-reading).

Документы сохраняются без изменения. rag/build_chunks.py выделяет разделы,
заголовки, пункты и таблицы с метаданными: id, file, product, path,
section_type и guard. В проверенной конфигурации получается 273 фрагмента.
Тип фрагмента важен: инструкция, ограничение, пример и свойство продукта
не должны одинаково использоваться как клиентский ответ.

## 2. Поиск

rag/query_normalization.py определяет текущую тему и нормализует запрос;
rag/retriever.py реализует BM25. rag/embedder.py и rag/retrieval.py добавляют
BGE-M3. rag/fusion.py объединяет ранги через взвешенный RRF, сохраняются
источники и устраняются повторы ID. При ошибке embeddings остаётся BM25.
Структурное расширение добавляет нужные ограничения и разделы исходных данных.
Реранкер доступен как опция, в основном проверенном профиле отключён.

## 3. Консультант

Поток: chat → история и явное действие → поиск → применимые цитаты →
проверка условий/составление ответа → сохранение ответа и диагностики.
consultant/consultant.py управляет обработкой, consultant/dialogue.py
извлекает только сообщённые клиентом сведения с цитатой и номером хода.
Отрицательные, неизвестные и выполненные условия различаются; позднее
исправление заменяет прежний факт. Это ограниченный извлекатель, не
универсальное понимание всех возможных отрицаний и ссылок на прошлое.

consultant/answer_policy.py задаёт цель и обязательные смыслы ответа;
evidence.py отделяет подтверждения от инструкций и проверяет пересказ;
relations.py связывает числа с единицей, объектом и диапазоном.
Присоединённая мощность не подменяется мощностью станции или энергией в кВт·ч.
grounded.py либо составляет ответ по структуре найденных источников, либо
использует Qwen для выбора цитат и пересказа с проверками и ограниченным
повтором. Истинная цитата сама по себе ещё не гарантирует релевантный ответ.

При неопределённом продукте/цене бот уточняет задачу; при недостатке
подтверждений не придумывает сведения. Прямая просьба о специалисте
обрабатывается приоритетно, отмена снимает ожидание контакта.

## Разделение ответственности

integration/interface.py — стабильная точка входа для будущего веб-чата.
memory/conversation_state.py хранит JSON-сессии. llm/qwen.py — адаптеры Ollama
и API. config.py читает окружение. main.py — консольная демонстрация.
Не переносите RAG в frontend и не создавайте вторую независимую историю:
backend должен передавать в ядро ту же session_id для одного диалога.

## Перечень исходных документов базы

| Файл | Тема |
| --- | --- |
| [00-product-map.md](../energytool_hackathon/app/knowledge-base/00-reference/00-product-map.md) | Карта продуктов МТС EnergyTool |
| [01-numbers-and-indicators.md](../energytool_hackathon/app/knowledge-base/00-reference/01-numbers-and-indicators.md) | Числовые показатели и оговорки |
| [02-product-scope-and-limits.md](../energytool_hackathon/app/knowledge-base/00-reference/02-product-scope-and-limits.md) | Область применения продуктов и ограничения |
| [03-calculation-and-requests.md](../energytool_hackathon/app/knowledge-base/00-reference/03-calculation-and-requests.md) | Расчёт по объекту и обращение к специалисту |
| [10-market.md](../energytool_hackathon/app/knowledge-base/10-energy/10-market.md) | Как устроен рынок электроэнергии и за что вы платите |
| [11-price-categories.md](../energytool_hackathon/app/knowledge-base/10-energy/11-price-categories.md) | Ценовые категории: какая у вас и какая выгоднее |
| [12-power-thresholds.md](../energytool_hackathon/app/knowledge-base/10-energy/12-power-thresholds.md) | Пороги мощности и уровни напряжения: что меняется и когда |
| [13-metering.md](../energytool_hackathon/app/knowledge-base/10-energy/13-metering.md) | Приборы учёта и системы сбора данных |
| [14-tariffs.md](../energytool_hackathon/app/knowledge-base/10-energy/14-tariffs.md) | Почему растёт счёт за электроэнергию |
| [15-microgeneration.md](../energytool_hackathon/app/knowledge-base/10-energy/15-microgeneration.md) | Собственная генерация и отдача энергии в сеть |
| [16-regulation.md](../energytool_hackathon/app/knowledge-base/10-energy/16-regulation.md) | Нормативная база: на что мы ссылаемся |
| [17-faq.md](../energytool_hackathon/app/knowledge-base/10-energy/17-faq.md) | Частые вопросы по энергетике |
| [20-portfolio.md](../energytool_hackathon/app/knowledge-base/20-products/20-portfolio.md) | Обзор продуктов: что подходит под вашу задачу |
| [21-orem.md](../energytool_hackathon/app/knowledge-base/20-products/21-orem.md) | Выход на оптовый рынок электроэнергии под ключ |
| [22-askue.md](../energytool_hackathon/app/knowledge-base/20-products/22-askue.md) | Облачная платформа учёта электроэнергии |
| [23-solar.md](../energytool_hackathon/app/knowledge-base/20-products/23-solar.md) | Солнечные электростанции под ключ |
| [24-losses.md](../energytool_hackathon/app/knowledge-base/20-products/24-losses.md) | Поиск коммерческих потерь электроэнергии |
| [25-meter-reading.md](../energytool_hackathon/app/knowledge-base/20-products/25-meter-reading.md) | Распознавание показаний приборов учёта по фотографии |
| [26-general.md](../energytool_hackathon/app/knowledge-base/20-products/26-general.md) | Общие вопросы |
