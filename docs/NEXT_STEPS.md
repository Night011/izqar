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

Новый потоковый анализатор не читает `Is Laundering`, не передаёт весь граф браузеру и внешне упорядочивает фактически неотсортированный `HI-Small_Trans.csv` по времени во временной SQLite БД. Он строит только ограниченный отчёт fan-in/fan-out, быстрых цепочек и трёхузловых циклов; результаты являются leads для расследования, не доказательством отмывания:

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
.\.venv\Scripts\python.exe scripts\analyze_ibm_graph.py --data <путь-к-HI-Small_Trans.csv> --output models\experiments\ibm_graph_report.json --max-rows 500000
```

Не публикуйте отчёт с реальными идентификаторами счетов. При работе с Kaggle сохраняйте JSON в Output и переносите только проверенный агрегированный отчёт.

После проверки перенесите JSON в `backend/models/experiments/ibm_graph_report.json` (этот каталог исключён из Git) либо задайте `IZQAR_IBM_AML_GRAPH_REPORT_PATH`. Перезапустите API: вкладка **Граф связей** получит только число строк, рёбер и структурных сигналов. Идентификаторы и сами цепочки API намеренно не возвращает.

Первый проверенный Kaggle run уже импортирован штатным `scripts/import_ibm_graph_report.py`: 500 000 операций, 361 541 ребро, 50 fan-in, 50 fan-out, 50 rapid chains и 0 cycles. SHA-256 отчёта: `b710be76fbc87636ab469afcf217b308484471ced12e42e1564f05f2948bd65b`; pinned commit: `9e63654cb64c19174a5a863e82789b2ac60dcee6`; `label_used=false`. Не заменяйте его вручную: importer проверяет лимит размера, exact revision provenance, отсутствие target key и запрет перезаписи.

## Новые validation-only ML experiments

До новых data artifacts используйте [протокол оптимизации](ML_OPTIMIZATION_PROTOCOL.md). В нём фиксируются три operating profile, Precision@K/Recall@K и паспорт каждого запуска. Новый `train_paysim.py --mode experiment` теперь, как IEEE/BAF/IBM, не считает и не записывает уже просмотренный final test.

Каждый отчёт из `models/experiments/` показывается в Model Registry как **кандидат** и не может заменить API-модель сам по себе. После независимой проверки и явного решения о promotion сохраните финальный артефакт вне `models/experiments/`, проверьте provenance/SHA-256 и только затем явно настройте путь активной модели. Не копируйте candidate-файлы поверх serving-артефактов вручную.

Для PaySim candidate сравнивайте `validation_type_slices.TRANSFER/CASH_OUT` и `validation_robustness` вместе с общими F1/PR-AUC/FP. Нулевой или отрицательный эффект в одном synthetic stress scenario не объясняет будущий дрейф и не допускает настройки на старом test.

Для первого controlled recall candidate подготовлен [ноутбук](../notebooks/paysim_recall_candidate_kaggle.ipynb). Он фиксирует `positive_weight_multiplier=1.25`, запускает только `--mode experiment` на полном PaySim и экспортирует **report-only** ZIP. Ноутбук проверяет точный SHA Git commit до обучения; перед запуском feature-ветка с этим commit должна быть явно опубликована или текущий код должен быть загружен в Kaggle эквивалентным проверяемым способом. Не обходите эту проверку и не устанавливайте report-only ZIP как модель API.

Все specialist Kaggle notebooks теперь фиксируют проверенную ревизию до установки зависимостей. Перед запуском их также необходимо сделать доступными Kaggle через разрешённую публикацию feature-ветки или загрузку текущего кода; отказ checkout — корректный стоп, а не повод запускать другой commit.

### IBM AML causal-history candidate

Повторный validation-only запуск завершён как Kaggle kernel `hoogle57/izqar-ibm-aml-causal-history-candidate-v1` на commit `34656c00d64b2b3a0e997cd6f817fbba9fed933b`. Он выполнил `causal_history_v1` с disk-backed SQLite сортировкой истории по parsed timestamp и не экспортировал `joblib` либо final test. ZIP SHA-256: `c7f9dd444883ea7e8dcfd2fbf90b656d88a5b2aac8e92350884ea7f3141ebaed`; он уже безопасно импортирован как report-only candidate. Для повторяемого импорта используйте:

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
.\.venv\Scripts\python.exe scripts\import_experiment_report.py `
  --archive <путь-к-izqar_ibm_causal_history_candidate_report.zip> `
  --output-dir models\experiments
```

Импортёр проверяет лимит размера, плоскую структуру ZIP, pinned SHA ревизии и отсутствие final test; он не десериализует модели. На одинаковом validation окне candidate дал PR-AUC `0.099919` против `0.031192`, Recall `0.592105` против `0.361842` и F1 `0.102424` против `0.064050` при 99.489 против 99.161 FP/10k. Это достаточное основание для дальнейшего controlled validation research, но не для использования уже просмотренного final test, automatic promotion или замены active-артефакта.

### IBM AML causal-history full early-train candidate

Полный ранний train-period run завершён как Kaggle kernel `hoogle57/izqar-ibm-causal-fulltrain-candidate-v1` на commit `0d4c222d4d97d677f514e841ae40968e67046156`. Он использовал `causal_history_v1`, 3 554 649 ранних train операций, следующие 761 593 validation операций и не запускал последние 762 103 final-test операций. ZIP `izqar_ibm_causal_fulltrain_candidate_report.zip` прошёл CRC/ZIP integrity и имеет SHA-256 `6906b4196e38ed905cdf84a0c6ac114edd3fe15a51bbd245eaec43616c7d6549`; он уже импортирован report-only через тот же importer.

На validation при лимите 100 FP/10k: PR-AUC `0.120583`, ROC-AUC `0.978720`, Precision `0.063157`, Recall `0.664474`, F1 `0.115349`, FP `7 491`, FN `255`, 98.360 FP/10k и Precision@1000 `0.179`. Score не является калиброванной вероятностью. Это лучший IBM validation candidate на текущем фиксированном окне, но он не является основанием для automatic promotion: прежде чем делать ровно одну final evaluation, надо явно зафиксировать конфигурацию и не менять её по старому final test.

### IEEE-CIS extended_cv weight 1.25 candidate

Kaggle kernel `hoogle57/izqar-ieee-extended-cv-weight-1-25-candidate-v1` завершил one-factor validation-only run на commit `34fda4d25c1ee388815ce21e771999f5d0468dad`. Его единственное отличие от verified `extended_cv` candidate — `positive_weight_multiplier=1.25`; final test не рассчитывался и модель не экспортировалась. ZIP `izqar_ieee_extended_weight125_candidate_report.zip` прошёл integrity check, SHA-256 `58038239a736c347ee21a97aed80c93bfa1fefa2268646daa432b660541962d1`, и уже импортирован в `models/experiments` как candidate.

На fixed validation window (88 581 операций, budget 100 FP/10k): PR-AUC `0.464304`, Precision `0.577458`, Recall `0.395792`, F1 `0.469670`, FP `881`, FN `1 838`, 99.457 FP/10k. Это немного улучшает предыдущий verified candidate, но не является basis для auto-promotion. Следующая final evaluation допустима только один раз после явного freeze выбранной configuration; повторно использовать старый final test для подбора нельзя.

### BAF и PaySim weight 0.75 candidates

Оба Kaggle run привязаны к commit `32f288ba3d0756277e33971fc05f461745cfa606`, используют прежние temporal splits и не считают final test. BAF `positive_weight_multiplier=0.75` улучшил validation PR-AUC/Recall/F1 относительно baseline без роста FP budget: `0.164185` / `0.229655` / `0.232542` при 99.937 FP/10k. Он остаётся candidate до явного freeze и одной final evaluation. PaySim `positive_weight_multiplier=0.75` улучшил PR-AUC/F1 и сократил FP, но снизил общую и CASH_OUT Recall; он записан только как trade-off candidate, а baseline сохраняется для recall-приоритета.

ZIP SHA-256: BAF `c8b4384a671517a1e94889e173907f3f586eadd4156e09b1158de8f0b664b4d3`; PaySim `aa3ed5e80912805f851ebfac83ca60b80088a5f104109c925b376407b0b454d3`. Оба прошли ZIP CRC и импортированы `scripts/import_experiment_report.py` без model payload.

### PaySim pre_transaction_v2 candidate

Первый run этой идеи был отозван: арифметический `origin_balance_after_requested_amount = oldbalanceOrg - amount` семантически воспроизводил запрещённый post-operation `newbalanceOrig`. Registry защищён от такого имени и показывает этот report как `invalid`; его хорошие validation-цифры не используются для выбора.

Исправленный controlled experiment меняет только feature set на `pre_transaction_v2`: `amount_to_origin_balance`, `amount_to_destination_balance` и `destination_balance_is_zero`. Все признаки row-local и вычисляются только из уже доступных до решения `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`; target, post-event balances, account IDs, step, history и arithmetic post-balance proxies исключены. Clean report прошёл CRC/provenance checks: SHA-256 `ee5df7210634f28df6b79763df1c5b2a14d58aad6f645eb5772b203119544149`, Kaggle `hoogle57/izqar-paysim-pre-transaction-v2-clean-candidate-v1`, revision `3a328d1699c704341ee398fbb8693093ce697663`. На 973 173 validation операциях: P `0.926991`, R `0.735088`, F1 `0.819961`, PR-AUC `0.873965`, ROC-AUC `0.999868`, 33 FP / 151 FN / 0.339 FP/10k. Это report-only candidate, без final test, promotion или serving-model change.

API уже умеет безопасно обслужить будущий approved v2 artifact, но принимает от клиента только четыре исходных pre-transaction поля. Он локально выводит только три allowlisted ratio/zero признаки и отклоняет любые дополнительные поля и неизвестную feature schema. Выражение `oldbalanceOrg - amount` запрещено и не может быть добавлено как convenience feature.

```powershell
.\.venv\Scripts\python.exe scripts\train_paysim.py `
  --data <PS_20174392719_1491204439457_log.csv> `
  --full-data --mode experiment --feature-set pre_transaction_v2 `
  --model-version paysim-xgboost-v4 --estimators 180 `
  --output models\experiments\paysim_pre_transaction_v2_candidate.joblib
```

Если после независимого review конфигурация будет явно зафиксирована, единственный допустимый final command требует отдельного acknowledgement и нового пустого output-пути. Не запускайте его для новой настройки и не повторяйте с другим порогом:

```powershell
.\.venv\Scripts\python.exe scripts\train_paysim.py `
  --data <PS_20174392719_1491204439457_log.csv> --full-data --mode final `
  --feature-set pre_transaction_v2 --frozen-final-feature-set pre_transaction_v2 `
  --model-version paysim-xgboost-v5 --estimators 180 `
  --output models\paysim_v5_pre_transaction_v2_sealed.joblib
```

Этот sealed run уже успешно выполнен один раз в Kaggle (`hoogle57/izqar-paysim-v2-sealed-final-v1`, revision `60479fc426f911de3d466537626b6b051cf4a939`): test F1 `0.852542`, PR-AUC `0.967377`, Recall `0.752618`, 52 FP и 992 FN на 955 744 synthetic операциях. ZIP SHA-256 `029F607A62055B72DFFFDDA8D8B86704C16840F15B67A1927C6818EBC2646CF0` прошёл CRC. Не запускайте эту команду повторно. До promotion нужно изолированно подтвердить корректный transform trusted scikit-learn `1.6.1` preprocessing в локальном `1.9.1` runtime; текущий secure installer намеренно отказался от несовместимого pickle.

Сравнивайте с active v1 и weight candidates на одном validation period: общий Recall/F1/PR-AUC, FP/10k, срезы TRANSFER/CASH_OUT и preprocessing stress. Не использовать final test для выбора.

### Unified IEEE loss weight 1.25 candidate

Kaggle kernel `hoogle57/izqar-unified-ieee-weight-125-candidate-v3` завершил four-task validation-only run на commit `1525c8cd2155cc59ec1845611d69609892fc4b5c`. Он меняет ровно IEEE loss weight с `1.0` на `1.25`; все четыре task продолжают обновлять shared backbone. ZIP SHA-256 `ea3bc3b6e87266b704814750afa9aea18e9510d6730817dadf6a7d771cb54fff` прошёл CRC, pinned revision и validation-only gates и импортирован без model weights.

PR-AUC вырос для PaySim (`0.038053`), IEEE (`0.190786`) и IBM (`0.013468`), но слегка снизился для BAF (`0.136371`) относительно исходной Unified validation. Поэтому candidate нельзя продвигать автоматически или называть улучшением всей Unified модели. Model Registry и UI показывают четыре task metrics отдельно; final test остаётся sealed.

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
