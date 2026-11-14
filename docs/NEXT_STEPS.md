# IZQAR — продолжение работы

## Unified Intelligence: текущее состояние

1. Notebook уже выполнен в Kaggle на commit `6f91892ac1062425f2a34a822dd617ee38f3b8cc`; не перезапускайте его ради подбора параметров.
2. Через **Add Input** подключите: `ealaxi/paysim1`, `ieee-fraud-detection`, `sgpjesus/bank-account-fraud-dataset-neurips-2022`, `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`. Условия IEEE-CIS принимает только владелец аккаунта.
3. Включите Internet только для клонирования feature-ветки и зависимостей. Исходные CSV не скачивать в локальный репозиторий.
4. `izqar_unified_artifacts.zip` уже скачан в `C:\Users\a\Downloads` и установлен штатным установщиком. Не заменяйте файлы в `backend/models/` вручную.
5. Сохраните вывод ячеек, commit SHA, размеры выборок, validation метрики, `gradient_audit_shared_backbone_l2`, `training.seconds` и checkpoint. До сравнения не называйте модель лучше специализированных.

## Повторяемая проверка установленного Unified

Если артефакты были удалены, безопасная повторная установка принимает только проверенный ZIP и откажется от подмены или перезаписи:

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
.\.venv\Scripts\python.exe scripts\install_unified_artifact.py --archive "C:\Users\a\Downloads\izqar_unified_artifacts.zip"
```

Не извлекайте ZIP вручную в `models/`: это обходит проверку происхождения. Скрипт сохранит `unified_weights.npz`, `unified_preprocessors.joblib`, `unified.metrics.json` и `unified_checkpoint.pt`.
2. Не меняйте `.env`, если используется путь по умолчанию. Для другого манифеста укажите `IZQAR_UNIFIED_MODEL_PATH=<путь к unified.metrics.json>`.
3. Запустите:

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
.\.venv\Scripts\python.exe -m pytest -q
```

4. Проверьте `GET /api/v1/model/status`: поле `unified_mtl.task_schemas.<task>` должно совпасть с `task_metadata.<task>.feature_schema.all` из манифеста. Затем выполните отдельный `POST /api/v1/unified/<task>/predict` с этой точной схемой. Панель ML Analytics умеет сформировать такой шаблон после установки артефакта.

## Однократная final-оценка Unified

Не запускайте этот этап для выбора модели или настроек. После зафиксированного решения по validation используйте существующий Kaggle output-каталог с доверенными `unified.metrics.json`, `unified_weights.npz` и `unified_preprocessors.joblib`:

```powershell
python backend/scripts/train_unified.py --paysim <PaySim.csv> --ieee-dir <ieee-dir> --baf <Base.csv> --ibm-aml <HI-Small_Trans.csv> --output-dir <existing-unified-output> --mode final
```

Команда не выполняет обучение: она применяет неизменённые веса, preprocessing и пороги, выбранные на validation, к закрытым test-окнам и создаёт `unified.final_test.json`. Повторный запуск в том же output-каталоге намеренно завершается ошибкой; не обходите эту защиту.

После установки артефактов `GET /api/v1/model/status` показывает `unified_mtl.final_evaluation_available: true` только если SHA-256 отчёта совпадают с установленными компонентами. В противном случае final-метрики не отображаются, а Unified-инференс остаётся доступен по проверенным весам.

## IBM AML graph-анализ на Kaggle

Новый потоковый анализатор не читает `Is Laundering`, не передаёт весь граф браузеру и требует хронологически упорядоченный `HI-Small_Trans.csv`. Он строит только ограниченный отчёт fan-in/fan-out, быстрых цепочек и трёхузловых циклов; результаты являются leads для расследования, не доказательством отмывания:

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
.\.venv\Scripts\python.exe scripts\analyze_ibm_graph.py --data <путь-к-HI-Small_Trans.csv> --output models\experiments\ibm_graph_report.json --max-rows 500000
```

Не публикуйте отчёт с реальными идентификаторами счетов. При работе с Kaggle сохраняйте JSON в Output и переносите только проверенный агрегированный отчёт.

После проверки перенесите JSON в `backend/models/experiments/ibm_graph_report.json` (этот каталог исключён из Git) либо задайте `IZQAR_IBM_AML_GRAPH_REPORT_PATH`. Перезапустите API: вкладка **Граф связей** получит только число строк, рёбер и структурных сигналов. Идентификаторы и сами цепочки API намеренно не возвращает.

## Новые validation-only ML experiments

До новых data artifacts используйте [протокол оптимизации](ML_OPTIMIZATION_PROTOCOL.md). В нём фиксируются три operating profile, Precision@K/Recall@K и паспорт каждого запуска. Новый `train_paysim.py --mode experiment` теперь, как IEEE/BAF/IBM, не считает и не записывает уже просмотренный final test.

Каждый отчёт из `models/experiments/` показывается в Model Registry как **кандидат** и не может заменить API-модель сам по себе. После независимой проверки и явного решения о promotion сохраните финальный артефакт вне `models/experiments/`, проверьте provenance/SHA-256 и только затем явно настройте путь активной модели. Не копируйте candidate-файлы поверх serving-артефактов вручную.

Для PaySim candidate сравнивайте `validation_type_slices.TRANSFER/CASH_OUT` и `validation_robustness` вместе с общими F1/PR-AUC/FP. Нулевой или отрицательный эффект в одном synthetic stress scenario не объясняет будущий дрейф и не допускает настройки на старом test.

Для первого controlled recall candidate подготовлен [ноутбук](../notebooks/paysim_recall_candidate_kaggle.ipynb). Он фиксирует `positive_weight_multiplier=1.25`, запускает только `--mode experiment` на полном PaySim и экспортирует **report-only** ZIP. Ноутбук проверяет точный SHA Git commit до обучения; перед запуском feature-ветка с этим commit должна быть явно опубликована или текущий код должен быть загружен в Kaggle эквивалентным проверяемым способом. Не обходите эту проверку и не устанавливайте report-only ZIP как модель API.

Все specialist Kaggle notebooks теперь фиксируют проверенную ревизию до установки зависимостей. Перед запуском их также необходимо сделать доступными Kaggle через разрешённую публикацию feature-ветки или загрузку текущего кода; отказ checkout — корректный стоп, а не повод запускать другой commit.

### IBM AML causal-history candidate

Повторный validation-only запуск уже создан как Kaggle kernel `hoogle57/izqar-ibm-aml-causal-history-candidate-v1` на commit `34656c00d64b2b3a0e997cd6f817fbba9fed933b`. Он выполняет `causal_history_v1` с disk-backed SQLite сортировкой истории по parsed timestamp и не экспортирует `joblib` либо final test. После завершения скачивайте только `izqar_ibm_causal_history_candidate_report.zip`, затем импортируйте его штатно:

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
.\.venv\Scripts\python.exe scripts\import_experiment_report.py `
  --archive <путь-к-izqar_ibm_causal_history_candidate_report.zip> `
  --output-dir models\experiments
```

Импортёр проверяет лимит размера, плоскую структуру ZIP, pinned SHA ревизии и отсутствие final test; он не десериализует модели. Сравнивайте только validation с текущим IBM baseline и не меняйте active-артефакт без отдельного решения.

## Локальные проверки и запуск

```powershell
cd C:\Users\a\Documents\IZQAR-MVP
powershell -ExecutionPolicy Bypass -File .\scripts\start_local.ps1 -ConfigureOnly
cd backend
.\.venv\Scripts\python.exe -m pytest -q
cd ..\frontend
npm run build
```

## Текущий handoff

Kaggle вкладка содержит успешный validation-only output и четыре файла в `/kaggle/working/izqar_unified_mtl_v1`. Доверенный export установлен локально; все четыре Unified API heads проверены. `.vscode/` остаётся пользовательским неотслеживаемым каталогом.
