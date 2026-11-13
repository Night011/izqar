# IZQAR — Intelligent Fraud Prevention Platform

**Рабочий исследовательский MVP для хакатона.** Система демонстрирует оценку риска синтетических переводов, анализ структуры связей между счетами и ручное расследование инцидентов. Два отдельных скрипта обучают модели PaySim и IEEE-CIS Fraud Detection.

> **Не банковское ПО.** Демо-скор 0–100 рассчитан **правилами**, не является вероятностью мошенничества. IEEE-CIS не содержит полноценного графа банковских переводов. Нельзя выдавать точность на IEEE-CIS за эффективность в Казахстане или на настоящих клиентах.

## Что уже реализовано

- **FastAPI** с версионированными endpoints, схемами Pydantic, проверкой API-ключа.
- **SQLite** для быстрого локального запуска, **PostgreSQL** в Docker Compose.
- **Rule Engine** с объяснениями оценки риска и признаками, вычисляемыми только по истории до транзакции.
- **5 сценариев симулятора**: обычный перевод, новый получатель, серия переводов, много отправителей на один счёт, цепочка переводов.
- **Графовый анализ NetworkX** (уникальные счета/рёбра, признаки высокого fan-in/fan-out); граф сам по себе не доказывает мошенничество.
- **Журнал операций**, статусы «подтверждено аналитиком» / «ложная тревога».
- **Два интерфейса**: автономный веб-интерфейс без Node.js и расширенная React/TypeScript панель с Cytoscape.
- **Два независимых ML-пайплайна**: PaySim и IEEE-CIS с отдельными схемами, артефактами и endpoint'ами; XGBoost, baseline, хронологическое разделение и порог только по validation.
- **ML Analytics в React** показывает метрики только из установленных артефактов, отдельный инференс двух моделей и состояние их загрузки. Журнал демо-операций по-прежнему показывает оценку Rule Engine.
- **Pytest** для API, симуляций, авторизации и графа; GitHub Actions для тестов и сборки фронтенда.

## Самый быстрый запуск в Windows, без Node.js

Требуется **Python 3.11+** и доступ к PyPI для первоначальной установки зависимостей.

1. Распакуй архив в отдельную папку, открой папку в VS Code.
2. Открой PowerShell в корне проекта.
3. Выполни:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\start_local.ps1
   ```

Скрипт создаёт `.env` со **случайным ключом**, создаёт `.venv`, устанавливает минимальные зависимости и открывает `http://127.0.0.1:8000/demo/`.

**Где брать ключ?** Открой `.env` в корне проекта, скопируй значение `IZQAR_API_KEY` и вставь его на экране входа. Ключ хранится в `sessionStorage` только на время сессии браузера. Не публикуй скриншоты ключа и не коммить `.env`.

Если команда `py -3` отсутствует, установи Python 3 и повтори запуск либо пользуйся ручным способом ниже.

## Ручной локальный запуск, Linux / Windows

В корне проекта создай `.env` на основе `.env.example`. **Замени** `IZQAR_API_KEY` на секрет длиной от 24 символов (генерация: `python -c "import secrets; print(secrets.token_urlsafe(32))"`).

Затем в терминале:

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Открой `http://127.0.0.1:8000/demo/`. API-документация: `http://127.0.0.1:8000/docs`.

`/health` доступен без ключа, все маршруты с данными `/api/v1/*` требуют заголовок `X-API-Key`.

## React + TypeScript (расширенный интерфейс)

Понадобится Node.js 22 и интернет для `npm install`. Backend должен работать.

```bash
cd frontend
npm install
npm run dev
```

Открой `http://127.0.0.1:5173`. Веб-приложение использует Vite proxy для `/api` и не требует дополнительной настройки CORS в локальном режиме.

## Обучение модели IEEE-CIS Fraud Detection

1. Открой <https://www.kaggle.com/competitions/ieee-fraud-detection/data>, войди в аккаунт, прими правила конкурса и скачай оба файла: `train_transaction.csv` и `train_identity.csv`.
2. Сохрани исходные CSV в `backend/data/ieee/` (директория игнорируется Git).
3. Установи ML-зависимости и запусти обучение из `backend/`:

```bash
python -m pip install -r requirements-ml.txt
python scripts/train_ieee.py --data-dir data/ieee --output models/ieee_final_v2.joblib --estimators 180 --mode final
```

Для экспериментов 100k/200k/300k сначала прочитай [протокол оценки](docs/IEEE_EVALUATION.md). `--max-train-rows` меняет только обучающий сегмент; validation и финальный test остаются фиксированными. Режим `experiment` не рассчитывает метрики финального test и сохраняет только validation-отчёт.

Ограниченный эксперимент с дополнительными C/V-признаками, без доступа к финальному test:

```powershell
.\.venv\Scripts\python.exe scripts\train_ieee.py --data-dir data\ieee --mode experiment --feature-set extended_cv --estimators 180 --output models\experiments\ieee_extended_cv_NEW.joblib
```

Режим `final` для `extended_cv` намеренно заблокирован: прежний финальный test уже просматривался. Протокол и результаты аудита: [docs/MULTIDATA_AUDIT.md](docs/MULTIDATA_AUDIT.md).

Выход: `models/ieee_final_v2.joblib` и `models/ieee_final_v2.metrics.json` с настоящими метриками отложенной части **IEEE-CIS**. Отчёт включает baseline и XGBoost, Precision, Recall, F1, PR-AUC, ROC-AUC, матрицу ошибок, FPR/FNR и ложные тревоги на 10 000 операций. Артефакт модели содержит сериализованный объект Python (`joblib`); загружай только собственные/доверенные файлы. Для использования нового файла в API установи `IZQAR_MODEL_PATH` на его путь и перезапусти API. Скрипт не перезаписывает прежние результаты.

## Обучение модели PaySim

1. Скачай исходный CSV PaySim после принятия условий на [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) и помести его в `backend/data/paysim/`. Исходник не меняется и не коммитится.
2. Запусти из `backend/`:

```bash
python -m pip install -r requirements-ml.txt
python scripts/train_paysim.py --data data/paysim/PS_20174392719_1491204439457_log.csv --output models/paysim_model.joblib --estimators 180 --full-data
```

Для PaySim без `--full-data` действует ограничение 500 000 строк; IEEE-CIS теперь всегда читает весь размеченный CSV, чтобы закрепить финальный test, и ограничивает только train с `--max-train-rows`. Финальными считаются только метрики реального запуска с документированным размером данных. Модель PaySim использует исключительно доступные до операции `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`; исключены `isFraud`, `isFlaggedFraud`, post-transaction balances, идентификаторы счетов и `step`. `oldbalance*` предполагаются остатками до операции; эту семантику необходимо проверять при переносе на иной источник данных.

PaySim и IEEE-CIS не объединяются и не сравниваются как одна популяция. Их оценки доступны только через отдельные API: `/api/v1/paysim/predict` и `/api/v1/ieee/predict`.

### Обучение в Kaggle Notebooks без локальных CSV

Загрузи [PaySim notebook](notebooks/paysim_kaggle.ipynb) и [IEEE-CIS notebook](notebooks/ieee_cis_kaggle.ipynb) как два отдельных Kaggle Notebooks. В первом используй **Add Input → ealaxi/paysim1**. Для второго сначала самостоятельно прими правила соревнования, затем **Add Input → ieee-fraud-detection**. Нужны только два размеченных train-файла; конкурсные test-файлы для оценки качества не используются. Включи Internet в настройках ноутбука для клонирования публичной feature-ветки и установки зависимостей. Если IEEE-файлы не появились в `/kaggle/input`, проверь принятие правил и повторно подключи Input.

В PaySim запусти все ячейки. В IEEE сначала запусти ячейки экспериментов и изучи validation-отчёты; финальная ячейка по умолчанию отключена (`RUN_FINAL = False`). После фиксации параметров и стратегии порога выставь `RUN_FINAL = True` и запусти её один раз. В `/kaggle/working` появятся `izqar_paysim_artifacts.zip` и `izqar_ieee_artifacts.zip`; каждый содержит `.joblib` со схемой, preprocessing и порогом, а также `.metrics.json` с фактическими validation/test метриками после финального запуска. Скачай ZIP из Output и распакуй проверенные артефакты в `backend/models/` на локальном IZQAR. Устанавливай совместимые версии Python и ML-библиотек при загрузке; `joblib` допускается только из доверенного собственного запуска. Перезапусти API после установки моделей. CSV остаются в Kaggle и не копируются в репозиторий. Для PaySim путь по умолчанию — `backend/models/paysim_model.joblib`; IEEE-CIS берёт путь из `IZQAR_MODEL_PATH` или `backend/models/ieee_model.joblib`.

**Совместимость существующих результатов.** Первоначальный полный PaySim-артефакт был обучен с scikit-learn 1.6.1, тогда как локальный IEEE-CIS артефакт и окружение API используют 1.9.1. `joblib` не гарантирует загрузку между версиями; локальный API сообщает о несовместимости и возвращает 503 для PaySim. Исходный `paysim_model.joblib` и его отчёт сохраняются. Для нового совместимого файла используй [PaySim v2 notebook](notebooks/paysim_compatible_kaggle.ipynb): он повторяет **те же зафиксированные параметры** с scikit-learn 1.9.1 и пишет отдельные `paysim_model_v2.joblib`, `.metrics.json` и `izqar_paysim_artifacts_v2.zip`. Установи `IZQAR_PAYSIM_MODEL_PATH` на локальный путь к `paysim_model_v2.joblib` в `.env` и перезапусти API. Повторная оценка на прежнем test — проверка воспроизводимости/совместимости, а не новый независимый эксперимент; для подбора улучшений используй только validation.

Для импорта PaySim ноутбука через веб-интерфейс Kaggle: **Code → New Notebook → Import Notebook → GitHub** и выбери `Night011/izqar`, ветку `feat/ml-pipelines-and-startup`, файл `notebooks/paysim_kaggle.ipynb`. Затем **Add Input → ealaxi/paysim1**. Если импорт локального файла через расширение Chrome заблокирован, включи доступ к file URLs в настройках расширения либо используй импорт из GitHub. Потребуется включить Internet в настройках Kaggle Notebook для `git clone` и установки пакетов. Не принимай правила конкурса IEEE-CIS от имени другого пользователя.

API `/api/v1/ieee/predict` принимает JSON `{"features": {...}}` со всеми полями обученной схемы. Список `features` записан в `models/ieee_model.metrics.json`; для отсутствующих значений передай явный `null`. Неизвестные и пропущенные ключи возвращают 422.

API **отдельный** от синтетического симулятора, потому что поля/популяции двух наборов данных несовместимы. Если модель не обучена, endpoint возвращает **503**, а не придумывает прогноз.

### Как оцениваем ML без самообмана

- Сортировка `TransactionDT`: 70% **train**, 15% **validation**, 15% **test**.
- Препроцессинг (импутация и OneHotEncoder) обучается **только** на train.
- Порог по максимуму F1 выбирается **только** на validation.
- PR-AUC, precision/recall/F1 и confusion matrix считаются на test **один раз**.
- Разная доля мошеннических операций в разных источниках и временных периодах делает предсказанные скоры неоткалиброванными.
- Не используем `isFraud` в признаках и не подменяем тестовые метрики примерными числами.

Улучшения на будущее: калибровка на отдельном временном периоде, анализ data drift, устойчивость к новым устройствам и клиентам, групповые разбиения, оценка стоимости ложных блокировок и monitoring.

## Docker Compose: PostgreSQL + API + React

Создай `.env` и укажи **два разных случайных секрета** `IZQAR_API_KEY` и `POSTGRES_PASSWORD`. Далее:

```bash
mkdir -p backend/models
docker compose up --build
```

Открой `http://127.0.0.1:5173`. Внешние порты Docker связаны только с `127.0.0.1`. Если хочешь использовать обученную модель в Docker, сначала создай `backend/models/ieee_model.joblib` локальным скриптом. После обучения перезапусти API.

**Важно:** не публикуй текущую версию через Cloudflare Tunnel или на общедоступном сервере: для публичного банка нужны многоуровневая аутентификация, HTTPS, RBAC, аудит, rate limiting, полноценные миграции БД и верификация алгоритмов. Один demo API-key не заменяет банковскую безопасность.

## API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Healthcheck (без ключа) |
| GET | `/demo/` | Автономная HTML/JS панель (без данных без ключа) |
| POST | `/api/v1/transfers` | Оценить и сохранить синтетический перевод |
| GET | `/api/v1/transactions` | Журнал переводов |
| PATCH | `/api/v1/transactions/{id}/review` | Ручное решение аналитика |
| GET | `/api/v1/metrics` | Статистика демонстрационной базы |
| GET | `/api/v1/graph` | Граф счетов и индикаторы fan-in/out |
| POST | `/api/v1/demo/simulate` | Запуск одного из пяти сценариев |
| GET | `/api/v1/model/status` | Состояние обеих независимых ML-моделей, версии, схемы и реальные метрики установленных артефактов |
| POST | `/api/v1/ieee/predict` | ML-скор записи IEEE-CIS (после обучения) |
| POST | `/api/v1/paysim/predict` | ML-скор совместимой записи PaySim (после обучения) |

### Тестирование

```bash
cd backend
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Структура репозитория

```text
backend/
  app/                  FastAPI, риск-движок, граф, автономная панель
  scripts/train_ieee.py Обучение XGBoost на IEEE-CIS
  tests/                Pytest
frontend/               React + TypeScript + Cytoscape
scripts/start_local.ps1 Быстрый локальный запуск Windows
docker-compose.yml      PostgreSQL + API + фронтенд
docs/                   Архитектура, задачи для Codex
.github/workflows/      CI для backend и frontend
```

## Ограничения v0.1.0

- Текущая модель правил использует фиксированные эвристики, а не обученные веса.
- Graph Engine показывает fan-in и fan-out, но **не** объявляет найденные счета мошенническими.
- IEEE-CIS используется как отдельный классификатор онлайн-платежей; на нём нельзя обучить граф реальных переводов между счетами.
- В начальном MVP нет полноценной ролевой модели, реальной интеграции с банком, миграций БД, журналирования действий аналитика и защищённого деплоя.
- Исторические демонстрационные счета и сценарии не отражают реальную банковскую активность.
- PaySim остаётся синтетическим набором данных; метрики на нём нельзя переносить на реальный банк. ML-скор не откалиброван как вероятность.
- Разработка возможна без Kaggle CSV; исходные датасеты и большие модели **не включены** в Git. Пока PaySim артефакт не установлен, соответствующая карточка в ML Analytics показывает его отсутствие.

Порядок показа жюри и проверяемые сценарии: [docs/JURY_DEMO.md](docs/JURY_DEMO.md).

## Лицензии и данные

Исходники прототипа доступны под MIT (см. `LICENSE`). IEEE-CIS принадлежит владельцам датасета и подчиняется условиям Kaggle; **не загружай CSV в GitHub**. В приложении нет реальных банковских или персональных данных.
