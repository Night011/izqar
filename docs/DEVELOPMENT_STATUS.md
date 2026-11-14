# IZQAR — статус разработки

Обновлено: 14 ноября 2026, Asia/Qyzylorda. Рабочая ветка: `feat/ml-pipelines-and-startup`; `main` не изменялась.

## Модели

| Компонент | Статус | Подтверждённый результат | Ограничение |
|---|---|---|---|
| PaySim v2 | Установлена, API проверен | test: F1 0.780426, PR-AUC 0.833058, 314 FP / 1 243 FN | Синтетические переводы; test ранее просматривался; скор не калиброван |
| IEEE-CIS v2 | Установлена, API проверен | test: F1 0.464247, PR-AUC 0.444182, 1 148 FP / 1 804 FN | Онлайн-платежи, не граф переводов; скор не калиброван |
| BAF Base v1 | Установлена, API проверен | test: F1 0.241141, PR-AUC 0.199692, 731 FP / 1 132 FN | Синтетические заявки; низкий Recall 0.207283 |
| IBM AML HI-Small v1 | Установлена, API проверен | test: F1 0.115342, PR-AUC 0.101293, 15 465 FP / 519 FN | На test 202.925 ложных тревог/10k против validation-бюджета 100; не менять порог по этому test |
| Unified Intelligence v1 | Kaggle CPU validation-only run завершён на четырёх входах; artifact ещё не экспортирован локально | PaySim PR-AUC 0.037139 / F1 0.050000; IEEE PR-AUC 0.188936 / F1 0.265172; BAF PR-AUC 0.137183 / F1 0.223346; IBM AML PR-AUC 0.013440 / F1 0.037538 | Это не final test и не основание заменить специализированные модели; scores не калиброваны |

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
- Backend: четыре специализированных endpoint-инференса; очередь расследований, статусы, комментарии и append-only `investigation_events` для синтетических переводов.
- Unified endpoint: `POST /api/v1/unified/{paysim|ieee_cis|baf_base|ibm_aml}/predict` подготовлен, но честно возвращает 503 до установки совместно обученного артефакта. После установки `/api/v1/model/status` выдаёт точную схему каждой головы (без меток), а ML Analytics позволяет выбрать Unified-задачу и сформировать валидный запрос.
- Специализированные API локально проверены через FastAPI `TestClient`: PaySim, IEEE-CIS, BAF Base и IBM AML ответили `200` с установленными версиями артефактов. Unified не выдавался за доступный: он отсутствует до доверенного локального экспорта.
- Frontend production build разделён по вкладкам: начальный JS уменьшен с 731.84 kB до 256.45 kB; Cytoscape/Network Explorer и рабочие пространства расследований/ML загружаются только при открытии нужной вкладки.
- Unified trainer поддерживает `--mode final`: это отдельная однократная оценка уже зафиксированного validation-экспорта. Она не обучает веса и не перезаписывает serving-манифест, а создаёт `unified.final_test.json` с SHA-256 компонентов и неизменёнными validation-порогами. Пока этот режим не запускался на реальных данных.
- Если однократный final-отчёт появится, API сверяет его тип, версию, четыре задачи, validation-пороги и SHA-256 manifest/weights/preprocessing перед показом метрик. Некорректный отчёт исключается из UI, но не мешает безопасному Unified-инференсу по установленным весам.
- ML Analytics содержит Model Comparison: она сопоставляет сохранённые метрики специализированной и Unified-модели по каждому источнику, но явно не делает вывод о превосходстве, пока не доказан один и тот же заранее зафиксированный holdout.
- Локальная проверка: 40 backend-тестов прошли; React build прошёл. После исправлений Kaggle CPU-path 11 профильных Unified-тестов прошли. В том числе проверены Unified adapter protocol, NumPy shared-backbone inference, 503 при отсутствии артефакта, загрузка тестового доверенного экспорта и выдача Unified serving-схемы через status API.
- Последняя проверка реальной локальной конфигурации: `/api/v1/model/status` и `/api/v1/model/registry` вернули 200; PaySim, IEEE-CIS, BAF Base и IBM AML загрузились, в registry 17 активных и исторических отчётов. После добавления лимита тела API полный тестовый набор вырос до 38 успешных тестов.

## Последние проверенные коммиты

- `00421b7 fix: preserve existing demo secrets during startup`
- `e262abb feat: add portable unified network inference kernel`
- `b3d598b feat: add persistent investigation workspace and audit events`
- `aaa51d4 fix(ml): unpack BAF temporal split in unified trainer`
- `6f91892 fix(ml): resolve unified auto device on CPU`

## Не закрыто

1. Экспортировать из завершённой Kaggle-сессии `unified_weights.npz`, `unified_preprocessors.joblib` и `unified.metrics.json` (и сохранить `unified_checkpoint.pt` для аудита). Не использовать final test для выбора параметров.
2. Установить только доверенный экспорт, проверить Unified API и добавить его в Model Registry и React, показав именно validation-only метрики и экспериментальный статус.
3. Продолжить validation-only исследования улучшения IBM AML, BAF, IEEE и PaySim; прежние финальные test не переиспользовать для выбора.
4. Ввести ML-оценки как самостоятельные сохраняемые инциденты без подмены их Rule Engine переводами.
5. Расширить временный граф IBM AML на данных Kaggle, не выдавая графовый сигнал за подтверждённое отмывание.
