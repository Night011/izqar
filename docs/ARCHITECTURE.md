# Архитектура IZQAR v0.1

## 1. Два независимых потока данных

### Демонстрационный поток

```
HTML / React UI
    └─POST /api/v1/demo/simulate или /transfers
          └─RiskEngine.assess(известная история + текущий перевод)
              └─SQLAlchemy → SQLite (localhost) / PostgreSQL (Docker)
              └─Объяснимые причины + risk_score, decision
    └─GET /api/v1/graph → NetworkX → Cytoscape / SVG
    └─PATCH /api/v1/transactions/{id}/review
```

Каждое действие сохраняет уже **синтетическую** операцию. Анализ признаков берёт только предыдущие строки БД. Высокая связность на графе — признак для изучения, а не обвинение.

### ML-потоки (строго раздельные)

```
train_transaction.csv (+ train_identity.csv)
  └─Хронологическая сортировка / 70:15:15
     └─Pipeline: Imputer, OneHotEncoder, XGBClassifier
        └─Порог по validation (F1)
           └─Оценка на test → JSON метрики
              └─models/ieee_model.joblib
                 └─POST /api/v1/ieee/predict

PaySim CSV
  └─Хронологическая сортировка `step` / 70:15:15
     └─только pre-transaction `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`
        └─исключение target, post-event balances, account IDs и `isFlaggedFraud`
           └─models/paysim_model_v2_portable.metrics.json
              + .preprocessor.joblib + .xgboost.json
              └─POST /api/v1/paysim/predict
```

Значения скоров симулятора **не** подаются в IEEE-модель и наоборот.

## 2. API и безопасность MVP

- Все `/api/v1/*` эндпоинты требуют заголовок `X-API-Key`, `secrets.compare_digest`.
- CORS ограничен указанными localhost origins; публикация в интернет запрещена до отдельного аудита.
- Транзакции только с фиктивными счетами, максимальная сумма 100 млн KZT, идентификаторы разрешают только безопасные латинские символы.
- SQLAlchemy parameter binding защищает от типовых SQL-инъекций.
- Для реальной эксплуатации необходимы индивидуальные учётные записи аналитиков, права, полноценный audit trail, throttling, CSRF (если появятся cookie-сессии), защищённый секрет-менеджмент и Alembic.

## 3. Корректность ML

- IEEE-CIS содержит онлайн-платёжные признаки и не содержит переводов `account A → account B`, достаточных для достоверного графового расследования.
- `TransactionDT` используется только для **разбиения**, не включён в ML-признаки. Identity merge проверяется `one_to_one` по `TransactionID`.
- PaySim `isFraud` используется исключительно как y. `isFlaggedFraud`, новые балансы, ID счетов и будущая история физически отсутствуют в feature contract.
- Обе модели сохраняют baseline и XGBoost метрики раздельно. Скор называется uncalibrated ML score, пока отдельная калибровка не доказана.
- PaySim v1/v2 архивы `joblib` сохранены, но их XGBoost pickle не является переносимым между Kaggle Linux и Windows. API принимает также формат v3: отчёт-манифест, обученный preprocessing и модель XGBoost JSON. Источник установки должен быть доверенным; загрузка preprocessing через `joblib` выполняет Python-десериализацию.
- Любой вывод о качестве модели должен ссылаться на `models/ieee_model.metrics.json`, **не** на примеры из симулятора.
- Отдельный графовый датасет (например, IBM AML, с соблюдением лицензии и воспроизводимым ETL) — следующий этап.

## 4. Как не сломать разработку

- Не меняй эндпоинты или JSON-контракты без обновления TypeScript типов, HTML-демо, Pytest и README.
- Старайся делать небольшие коммиты Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`.
- Не пиши в README метрики, которых не было в реально запущенных экспериментах.
