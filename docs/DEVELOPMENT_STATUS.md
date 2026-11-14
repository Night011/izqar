# IZQAR — статус разработки

Обновлено: 14 ноября 2026, Asia/Qyzylorda. Рабочая ветка: `feat/ml-pipelines-and-startup`; `main` не изменялась.

## Модели

| Компонент | Статус | Подтверждённый результат | Ограничение |
|---|---|---|---|
| PaySim v2 | Установлена, API проверен | test: F1 0.780426, PR-AUC 0.833058, 314 FP / 1 243 FN | Синтетические переводы; test ранее просматривался; скор не калиброван |
| IEEE-CIS v2 | Установлена, API проверен | test: F1 0.464247, PR-AUC 0.444182, 1 148 FP / 1 804 FN | Онлайн-платежи, не граф переводов; скор не калиброван |
| BAF Base v1 | Установлена, API проверен | test: F1 0.241141, PR-AUC 0.199692, 731 FP / 1 132 FN | Синтетические заявки; низкий Recall 0.207283 |
| IBM AML HI-Small v1 | Установлена, API проверен | test: F1 0.115342, PR-AUC 0.101293, 15 465 FP / 519 FN | На test 202.925 ложных тревог/10k против validation-бюджета 100; не менять порог по этому test |
| Unified Intelligence v1 | Установлена из проверенного Kaggle ZIP; API проверен для всех четырёх heads | PaySim PR-AUC 0.037139 / F1 0.050000; IEEE PR-AUC 0.188936 / F1 0.265172; BAF PR-AUC 0.137183 / F1 0.223346; IBM AML PR-AUC 0.013440 / F1 0.037538 | Это validation-only, не final test и не основание заменить специализированные модели; scores не калиброваны |

### Unified Intelligence v1 — Kaggle validation-only, 14 ноября 2026

Источник: четыре подключённых Kaggle inputs, commit `6f91892ac1062425f2a34a822dd617ee38f3b8cc`, CPU (`torch 2.11.0+cpu`, CUDA недоступна). Конфигурация: до 150 000 train-строк и до 75 000 validation-строк на задачу, 12 эпох, batch 1024. Финальные 15% периодов не загружались тренером.

| Задача | PR-AUC | ROC-AUC | Precision | Recall | F1 | FP / FN | Ложных тревог / 10k |
|---|---:|---:|---:|---:|---:|---:|---:|
| PaySim | 0.037139 | 0.937116 | 1.000000 | 0.025641 | 0.050000 | 0 / 38 | 0.000 |
| IEEE-CIS | 0.188936 | 0.765135 | 0.292580 | 0.242459 | 0.265172 | 1 516 / 1 959 | 202.133 |
| BAF Base | 0.137183 | 0.863841 | 0.203859 | 0.246951 | 0.223346 | 949 / 741 | 126.533 |
| IBM AML | 0.013440 | 0.875654 | 0.020129 | 0.277778 | 0.037538 | 1 217 / 65 | 162.267 |

Gradient audit для общей backbone был ненулевым: PaySim `0.21800983`, IEEE-CIS `0.01058507`, BAF `0.04945304`, IBM AML `0.13348407`. Это подтверждает, что все четыре task head посылали градиент в shared layers, но не подтверждает эксплуатационное превосходство модели.

## Платформа

- React: мониторинг, Model Registry, ML Analytics, интерактивный NetworkX/Cytoscape граф и рабочее место расследований. Граф ограниченно показывает fan-in/fan-out, быстрые цепочки и направленные трёхузловые циклы как сигналы, а не доказательства.
- IBM AML graph-анализатор может быть запущен отдельно на Kaggle и сохранён как ограниченный JSON-отчёт. `GET /api/v1/graph/ibm-aml-report` и Network Explorer показывают из него только агрегированные счётчики; названия синтетических счетов и цепочки не отправляются в браузер. Пока реальный отчёт не импортирован, UI явно показывает этот статус.
- Backend: четыре специализированных endpoint-инференса; очередь расследований, статусы, комментарии и append-only `investigation_events` для синтетических переводов. Endpoint ML assessment запускает выбранную модель server-side и хранит только provenance, итоговый score и имена полей — не пользовательские feature values; score не меняет Rule Engine или решение аналитика. Investigation UI позволяет явно выбрать доступную модель, вставить точную schema и прикрепить пересчитанную оценку к уже выбранному делу.
- Unified endpoint: `POST /api/v1/unified/{paysim|ieee_cis|baf_base|ibm_aml}/predict` установлен и проверен через FastAPI `TestClient`: все четыре heads вернули `200` с точной schema из manifest. `/api/v1/model/status` показывает экспериментальный статус, validation-only метрики и schema без меток.
- Специализированные XGBoost endpoint-инференсы возвращают до пяти локальных TreeSHAP-вкладов в log-odds без feature values. Это объяснение поведения модели, а не причинное доказательство; для Unified local attribution пока не валидирован и честно недоступен.
- Специализированные API и Unified загружены одновременно: PaySim, IEEE-CIS, BAF Base, IBM AML и Unified имеют `artifact_errors: null`; `ready: true`.
- Последний локальный integration smoke с фактическим `.env` и установленными артефактами: `/api/v1/model/status` вернул `200`, все четыре specialist endpoint и все четыре Unified head endpoint вернули `200` и `score_kind=uncalibrated_ml_score`. Это проверка контракта инференса, не новая оценка качества.
- Frontend production build разделён по вкладкам: начальный JS уменьшен с 731.84 kB до 256.45 kB; Cytoscape/Network Explorer и рабочие пространства расследований/ML загружаются только при открытии нужной вкладки.
- Unified trainer поддерживает `--mode final`: это отдельная однократная оценка уже зафиксированного validation-экспорта. Она не обучает веса и не перезаписывает serving-манифест, а создаёт `unified.final_test.json` с SHA-256 компонентов и неизменёнными validation-порогами. Пока этот режим не запускался на реальных данных.
- Если однократный final-отчёт появится, API сверяет его тип, версию, четыре задачи, validation-пороги и SHA-256 manifest/weights/preprocessing перед показом метрик. Некорректный отчёт исключается из UI, но не мешает безопасному Unified-инференсу по установленным весам.
- ML Analytics содержит Model Comparison: она сопоставляет сохранённые метрики специализированной и Unified-модели по каждому источнику, но явно не делает вывод о превосходстве, пока не доказан один и тот же заранее зафиксированный holdout.
- Для экспорта Unified подготовлен `scripts/install_unified_artifact.py`: он принял ZIP из `C:\Users\a\Downloads\izqar_unified_artifacts.zip`, сверил все четыре зафиксированные SHA-256 и установил компоненты без ручного извлечения. Chrome сохранял ZIP в `C:\Users\a\Downloads`, а не в `F:\download`.
- Kaggle preprocessing создан в scikit-learn 1.6.1, тогда как четыре специализированных portable-артефакта требуют локальный 1.9.1. Вместо downgrade всего runtime Unified loader использует узкий adapter для доверенного legacy `ColumnTransformer` и `SimpleImputer`; status явно сообщает `sklearn_1_6_to_1_9_adapter`. Модель не переобучалась.
- Локальная проверка: `66 passed` backend-тестов (одно предупреждение Starlette о будущей deprecation) и production React build прошли. В том числе проверены Unified adapter protocol, NumPy shared-backbone inference, schema safety, loader compatibility, IBM graph signals, локальные TreeSHAP-вклады, validation-only operating profiles и calibration diagnostics, experiment-only chunk-safe IBM history candidate, explicit Unified task weights и выдача Unified serving-схемы через status API.
- Model Registry различает целостность файлов (`installed` / `report_only`) и жизненный цикл (`active` / `candidate` / `archived`). Отчёт из `models/experiments` или с `experiment.mode=experiment` является кандидатом и никогда не получает API-трафик автоматически, даже если рядом лежат компоненты модели.
- PaySim trainer дополняет будущие validation-only candidate отчёты срезами Recall/Precision по `TRANSFER` и `CASH_OUT` и двумя явно синтетическими проверками устойчивости preprocessing (детерминированные пропуски балансов и неизвестный `type`). Новых метрик на прежнем final test не получалось и active PaySim-артефакт не менялся.
- `/api/v1/model/status` и ML Analytics передают и показывают эти PaySim validation diagnostics только когда они реально присутствуют в артефакте; текущий старый active-артефакт не получает фиктивных значений.
- API ограничивает тела всех `/api/v1/*` запросов 64 KiB по фактическим ASGI chunk-байтам, а не только по `Content-Length`; превышение возвращает `413` до JSON parsing/ML-инференса. CORS остаётся внешним слоем для browser-клиентов.
- PaySim candidate `paysim_positive_weight_1.25` реально выполнен в Kaggle как validation-only experiment (`paysim-a00818b2f58f`): на 973 173 validation операциях Precision `0.639493`, Recall `0.619298`, F1 `0.629234`, PR-AUC `0.616578`, FP `199`, FN `217`, 2.045 FP/10k. На том же окне baseline имеет Recall `0.666667`, F1 `0.630182`, PR-AUC `0.619489`; candidate не улучшает приоритет Recall/PR-AUC и остаётся report-only кандидатом без final test и без serving-модели.
- Первый IBM causal-history Kaggle candidate остановился до обучения, потому что HI-Small CSV физически не был хронологичен. Для `causal_history_v1` добавлена disk-backed SQLite external sort по parsed timestamp; она не меняет source CSV, ограничивает RAM и сохраняет batch-семантику одинакового времени. Повторный Kaggle validation-only run v2 запущен 14 ноября 2026 на commit `34656c00d64b2b3a0e997cd6f817fbba9fed933b`; на момент обновления его результат ещё не получен и active-артефакт не менялся.
- Добавлен Kaggle notebook `paysim_recall_candidate_kaggle.ipynb` для единственного validation-only candidate (`positive_weight_multiplier=1.25`). Он фиксирует code revision, отказывается от stale checkout и экспортирует только report ZIP без serving-модели. Он ещё не выполнялся, поэтому результатов кандидата нет.
- Последняя проверка реального локального сервера: `scripts/start_local.ps1 -ConfigureOnly` подтвердил существующую конфигурацию без печати или изменения API-ключа. Временный Uvicorn smoke test с отдельной SQLite БД вернул `200` для `/health`, `401` для защищённого status без ключа и `200` для status с временным тестовым ключом; пять моделей теперь готовы для локального инференса.

## Последние проверенные коммиты

- `00421b7 fix: preserve existing demo secrets during startup`
- `e262abb feat: add portable unified network inference kernel`
- `b3d598b feat: add persistent investigation workspace and audit events`
- `aaa51d4 fix(ml): unpack BAF temporal split in unified trainer`
- `6f91892 fix(ml): resolve unified auto device on CPU`

## Не закрыто

1. Не запускать Unified final test для подбора параметров. Если решение о конфигурации зафиксировано, выполнить ровно одну изолированную final-оценку в Kaggle по документированной команде ниже.
2. Продолжить validation-only исследования улучшения IBM AML, BAF, IEEE и PaySim по [ML_OPTIMIZATION_PROTOCOL.md](ML_OPTIMIZATION_PROTOCOL.md); прежние финальные test не переиспользовать для выбора. IBM `causal_history_v1` уже подключён к experiment-only тренеру с chunk-safe causality, но реальных новых candidate runs пока нет. Единые validation reports содержат Precision@K/Recall@K, Monitoring/Analyst Review/High Confidence profiles и experiment provenance.
3. Связать UI ML Analytics с созданием отдельной ML assessment для существующего расследования только после явного выбора аналитиком; не подменять Rule Engine или решение человека.
4. Запустить новый ограниченный IBM graph-анализатор на Kaggle HI-Small и сохранить агрегированный отчёт; не выдавать графовый сигнал за подтверждённое отмывание.
