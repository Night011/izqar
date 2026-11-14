# Протокол оптимизации ML-моделей IZQAR

Этот документ задаёт порядок validation-only исследований. Он не объявляет новые метрики и не разрешает повторно подбирать параметры по ранее просмотренным specialist test-наборам.

## Неподвижные правила

1. Активные PaySim, IEEE-CIS, BAF Base и IBM AML артефакты остаются baseline.
2. Каждый candidate запускается в режиме `experiment`: финальный test не рассчитывается и модель не становится active.
3. Признаки, preprocessing, threshold и алгоритм выбираются только по ранним временным train/validation-окнам.
4. Одинаковые candidate сравниваются с baseline на одинаковой validation-выборке. Улучшение только одной цифры не считается победой без проверки нагрузки очереди и устойчивости по времени.
5. Новый candidate может получить отдельный portable ZIP лишь после заранее зафиксированного критерия. До независимой оценки он остаётся shadow/experiment.

## Что сохраняет каждый новый отчёт

Тренеры PaySim, IEEE-CIS, BAF Base и IBM AML сохраняют `experiment` с ID, код-ревизией, схемой признаков, split, параметрами, временем и доступным high-water RSS. Также есть:

- `validation_ranking_metrics`: Precision@100/500/1000 и Recall@K;
- `validation_calibration`: uncalibrated Brier score и непустые score bins с observed positive rate;
- `validation_operating_profiles`:
  - **monitoring** — максимальный threshold, удерживающий validation Recall ≥ 80%, либо честная пометка, что условие недостижимо;
  - **analyst_review** — validation-selected operating point;
  - **high_confidence** — максимальный Recall при Precision ≥ 80%, либо честная недоступность;
- полный `threshold_analysis`, feature importance и предупреждения о применимости.

Профили не являются банковскими policy: 80% — явная исследовательская настройка для сравнения, а не готовое правило блокировки.
Калибровочные bins — диагностика над validation, а не разрешение называть score вероятностью. Отдельная калибровка требует зафиксированного дополнительного периода.

## Приоритеты исследований

| Направление | Validation-only критерий | Первые безопасные эксперименты |
|---|---|---|
| IBM AML | Precision/PR-AUC и alerts/10k при сохранении Recall, Precision@K | score distribution, payment-format/currency/time slices, causальные account-history/graph признаки без ID в модели, reranking для review queue |
| BAF | Recall и PR-AUC при контроле FP | class weighting, разрешённые признаки заявки, month drift, CatBoost/LightGBM candidate на неизменном split |
| IEEE-CIS | Recall при заранее заданном FP budget | C/D/V/identity availability audit, missingness, time slices, candidate preprocessing |
| PaySim | Recall по `TRANSFER`/`CASH_OUT` и устойчивость | validation error slices, безопасные pre-transaction balance features, unknown category stress |
| Unified MTL | PR-AUC/F1 по каждой task без отрицательного transfer | task/class balancing, shared-width, adapters/heads, gradient-interference audit, early stopping |

## Команды безопасных experiments

Все команды требуют исходные данные из разрешённого Kaggle Input и сохраняют отчёт в игнорируемом `backend/models/experiments/`.

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\train_paysim.py --data <PaySim.csv> --mode experiment --max-rows 500000 --output models\experiments\paysim_candidate_01.joblib
.\.venv\Scripts\python.exe scripts\train_ieee.py --data-dir <ieee-dir> --mode experiment --feature-set extended_cv --max-train-rows 300000 --output models\experiments\ieee_candidate_01.joblib
.\.venv\Scripts\python.exe scripts\train_baf.py --data <Base.csv> --mode experiment --max-train-rows 500000 --output models\experiments\baf_candidate_01.joblib --max-false-alerts-per-10000 100
.\.venv\Scripts\python.exe scripts\train_ibm_aml.py --data <HI-Small_Trans.csv> --mode experiment --max-train-rows 500000 --output models\experiments\ibm_candidate_01.joblib --max-false-alerts-per-10000 100
.\.venv\Scripts\python.exe scripts\train_ibm_aml.py --data <HI-Small_Trans.csv> --mode experiment --feature-set causal_history_v1 --history-max-accounts 500000 --max-train-rows 500000 --output models\experiments\ibm_history_candidate_01.joblib --max-false-alerts-per-10000 100
```

Не запускайте `--mode final` для настройки этих старых holdout. Для Unified остаётся отдельная sealed one-time final процедура из [NEXT_STEPS.md](NEXT_STEPS.md).

Для Unified task-balancing experiment передайте все четыре веса явно, например:

```powershell
.\.venv\Scripts\python.exe scripts\train_unified.py --paysim <PaySim.csv> --ieee-dir <ieee-dir> --baf <Base.csv> --ibm-aml <HI-Small_Trans.csv> --output-dir models\experiments\unified_task_weight_01 --mode experiment --task-weights '{"paysim":1.0,"ieee_cis":1.25,"baf_base":1.25,"ibm_aml":1.5}'
```

Это не ensemble и не routing: веса масштабируют BCE-loss каждой головы, поэтому все задачи продолжают обновлять общий backbone. Конфигурация попадает в `training.task_weights` манифеста. Нельзя передавать этот параметр в sealed `--mode final`.

Для controlled class-imbalance кандидата добавьте `--positive-weight-multiplier 0.75`, `1.25` или `1.5` к одной из первых трёх команд. Это multiplier поверх inverse-prevalence веса; сравнивайте только одинаковые validation split, все operating profile и Precision@K. Любое значение отличное от `1.0` намеренно блокирует `--mode final`.

`scripts/ibm_history_features.py` реализует prior outgoing/incoming counts, amounts и time-since-activity. Его `CausalHistoryStream` удерживает последнюю timestamp-группу между CSV chunks, поэтому одинаковое время не даёт операциям увидеть друг друга. Account остаётся только ключом истории. `train_ibm_aml.py --feature-set causal_history_v1` подключает эти признаки **только** к experiment candidate и блокирует `--mode final`; активный IBM артефакт не изменяется.

## Интерпретация результатов

- **Monitoring** подходит для исследования покрытия; он не означает автоматическую блокировку.
- **Analyst Review** оценивает компромисс false alerts/Recall и размер очереди.
- **High Confidence** не является доказательством мошенничества и может честно отсутствовать, если validation не достигает нужного Precision.
- Периодные, категориальные и графовые срезы — диагностика. Они не дают права менять candidate после просмотра закрытого test.
