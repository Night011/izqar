# Материалы и чек-лист подачи (хакатон, 20 ноября 2026)

Документ фиксирует, что именно проверено на живом стенде и что прикладывается к заявке.
Все проверки выполнены на `main`-совместимой ветке `feat/hackathon-release`, 20.11.2026,
локально, без CI (GitHub Actions заблокированы billing lock; успех CI не заявляется).

## 1. Состав заявки

| Материал | Путь | Статус |
|---|---|---|
| README проекта | [README.md](../README.md) | готов, проверены все цифры |
| Презентация PPTX (Google Slides) | [docs/presentation/IZQAR-hackathon.pptx](presentation/IZQAR-hackathon.pptx) | 8 слайдов, проверена рендером |
| Презентация PDF | [docs/presentation/IZQAR-hackathon.pdf](presentation/IZQAR-hackathon.pdf) | 8 страниц, сверена с PPTX |
| Текст слайдов и речь | [PRESENTATION.md](PRESENTATION.md) | готово |
| Сценарий видео 2–4 минуты | [DEMO_VIDEO_SCRIPT.md](DEMO_VIDEO_SCRIPT.md) | готово; запись не выполнялась |
| Прямые ссылки для формы | [SUBMISSION_LINKS.md](SUBMISSION_LINKS.md) | проверены анонимно |
| Сценарий показа на 5 минут | [JURY_DEMO.md](JURY_DEMO.md) | проверен на живом стенде |
| Скриншоты приложения | [screenshots/](screenshots/) (5 PNG) | сняты с работающего приложения |
| Архитектура | [ARCHITECTURE.md](ARCHITECTURE.md) | актуальна |
| PaySim: политики порога | [PAYSIM_THRESHOLD_POLICIES.md](PAYSIM_THRESHOLD_POLICIES.md) | измерено на validation |
| Данные и модели (аудит) | [MULTIDATA_AUDIT.md](MULTIDATA_AUDIT.md) | актуален |
| Статус разработки | [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md) | актуален |

## 2. Как запустить (проверено)

```powershell
git clone https://github.com/Night011/izqar.git
cd izqar
powershell -ExecutionPolicy Bypass -File .\scripts\start_local.ps1   # .env + ключ + API
cd frontend; npm install; npm run dev                                  # http://127.0.0.1:5173
python ..\scripts\seed_demo.py                                        # демо-данные
```

В другом терминале (для скрипта демо-данных):

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

ML-артефакты в Git не входят. Их нужно положить в `backend/models/` из проверенного
экспорта (см. [NEXT_STEPS.md](NEXT_STEPS.md)). Без артефакта endpoint отвечает `503`,
интерфейс показывает «не установлена» — фиктивных прогнозов нет.

## 3. Что проверено на живом стенде (20.11.2026)

- **Backend:** `162 passed` (pytest, из `backend/`; четыре теста добавлены вместе с PaySim threshold policies).
- **Frontend:** `npm run build` успешен; Vite dev-сервер отдаёт приложение на `:5173`.
- **Демо-сценарии (API):** `normal` 1 операция, `new_recipient` 1 (high), `velocity` 4,
  `fan_in` 6, `chain` 4, `cycle` 3 — все срабатывания объяснены Rule Engine.
- **Граф:** 23 узла / 19 рёбер на демо-истории.
- **ML-предикты:** PaySim 99.93 (v5), IEEE-CIS 68.81 (v2), BAF 45.6 (v1),
  IBM AML 1.24 (v1), Unified IEEE 73.18 (эксперимент) — реальные артефакты, версии совпадают.
- **Расследование:** комментарий аналитика и ML-assessment сохранены и видны в журнале.
- **Модели в статусе:** 5 из 5 загружены, ошибок артефактов нет.

Команда проверки (воспроизводимо):

```powershell
cd backend; .\.venv\Scripts\python.exe -m pytest -q
cd ..\frontend; npm run build
# затем scripts/seed_demo.py против запущенного API
```

## 4. Точные показатели для заявки

Берутся только из `backend/models/*.metrics.json`:

| Модель | PR-AUC | F1 | Recall | Precision | FP/10k | Test |
|---|---:|---:|---:|---:|---:|---:|
| PaySim v5 | 0.967377 | 0.852542 | 0.752618 | 0.983062 | 0.544 | 955 744 |
| IEEE-CIS v2 | 0.444182 | 0.464247 | 0.414856 | 0.526988 | 129.599 | 88 581 |
| BAF Base v1 | 0.199692 | 0.241141 | 0.207283 | 0.288218 | 75.483 | 96 843 |
| IBM AML v1 | 0.101293 | 0.115342 | 0.667521 | 0.063125 | 202.925 | 762 103 |

Unified MTL — только validation, статус «эксперимент», в сравнительные таблицы не включать.

## 5. Чего в заявке нет и почему

- **Промоушен IEEE v3 / BAF-ансамбля:** research-результаты выше production, но это
  изменение прод-контракта; в заявке они упоминаются как roadmap, не как факт.
- **IBM AML v4:** параллельная незавершённая работа другого агента; не включена.
- **Потоковые счётчики BAF (7 velocity-признаков):** требуют стриминговой инфраструктуры,
  которой нет в MVP; только в планах.
- **CI:** не запускался (billing lock). Заявлять зелёный CI нельзя.

## 6. Тайминг релизного дня

| Время (KZT) | Работа |
|---|---|
| до 18:00 | README, презентация, скриншоты, демо-данные, PaySim-сравнение, проверки |
| 18:00–21:00 | только критические исправления и упаковка |
| 21:00–23:00 | финальное прохождение демо по сценарию, отправка |
| 23:59 | дедлайн |
