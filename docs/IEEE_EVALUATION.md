# IEEE-CIS: протокол экспериментов и финальной оценки

`train_ieee.py` читает только размеченные `train_transaction.csv` и `train_identity.csv`, проверяет уникальность `TransactionID`, соединяет identity левым `one_to_one` join и сортирует по относительному времени `TransactionDT` и идентификатору. Граница validation — timestamp строки 70% полного train CSV; граница финального test — timestamp строки 85%. Равные timestamps целиком относятся к более позднему сегменту. Реальные доли могут немного отличаться от 70/15/15.

Финальный test — последние примерно 15% **всего размеченного** набора. Он не участвует в выборе объёма train, гиперпараметров, списка признаков, baseline, порога или лимита ложных тревог. `--max-train-rows` берёт последние N строк внутри раннего train-сегмента. Validation и test при 100k, 200k, 300k одинаковы. В режиме `experiment` модель и test-метрики не сохраняются: результат — отдельный `.metrics.json` с validation и кривой порогов.

Ранее полученные результаты 100k/200k/300k относились к разным временным test-окнам и потому не служат сравнением при общем holdout. Отчёт 300k: Precision 0.581761, Recall 0.421172, F1 0.488610, PR-AUC 0.502464, ROC-AUC 0.876124, FP 532 и FN 1017 на 45 000 тестовых операциях. Эти числа — исторический запуск, **не** новая финальная оценка.

На локальных IEEE CSV после исправления baseline запущены сопоставимые эксперименты с 100 деревьями. Validation во всех трёх — 88 581 операций, финальный test не оценивался:

| Train | Validation Precision | Recall | F1 | PR-AUC | FP | Итерации baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 100 000 | 0.575622 | 0.380342 | 0.458036 | 0.459228 | 853 | 33 |
| 200 000 | 0.546769 | 0.397765 | 0.460514 | 0.450995 | 1 003 | 15 |
| 300 000 | 0.544993 | 0.388231 | 0.453446 | 0.454491 | 986 | 17 |

Baseline применяет train-fitted standard scaling, ограничивает стандартизованные числовые значения до [-20, 20] и использует `liblinear`. На этих реальных подвыборках он сошёлся без предупреждений при `max_iter=1000`; его F1 по validation около 0.19, поэтому он остаётся только сравнительным baseline. Отчёты лежат в игнорируемом Git каталоге `backend/models/experiments/`.

## Эксперименты

Из `backend/` в PowerShell после установки `requirements-ml.txt` и размещения обоих CSV в `data/ieee/`:

```powershell
.\.venv\Scripts\python.exe scripts\train_ieee.py --data-dir data\ieee --mode experiment --max-train-rows 100000 --estimators 180 --output models\experiments\ieee_100k.joblib
.\.venv\Scripts\python.exe scripts\train_ieee.py --data-dir data\ieee --mode experiment --max-train-rows 200000 --estimators 180 --output models\experiments\ieee_200k.joblib
.\.venv\Scripts\python.exe scripts\train_ieee.py --data-dir data\ieee --mode experiment --max-train-rows 300000 --estimators 180 --output models\experiments\ieee_300k.joblib
```

В отчётах `models/experiments/*.metrics.json` сравнивай validation PR-AUC, Precision, Recall, F1, FP/FN и `threshold_analysis.curve`. Кривая содержит до 101 порога для чтения; выбор по лимиту ложных тревог проходит по **всем** уникальным validation-скорам. `--max-false-alerts-per-10000 100` выбирает порог с максимальным recall при не более 100 ложных тревогах на 10 000 validation-операций. Без опции выбирается максимум F1. Лимит фиксируется до финального test.

`feature_importance.top_30` — важность преобразованных признаков XGBoost. Автоматическая проверка запрещает `isFraud`, `TransactionID`, `TransactionDT`; она не доказывает отсутствие скрытой утечки в анонимизированных полях IEEE. Особо проверь семантику высокоранговых `D`, `V`, `C` и device-признаков до финального запуска. Отбор признаков по test запрещён.

## Однократный финальный запуск

После фиксации набора признаков, `--estimators` и стратегии порога по validation:

```powershell
.\.venv\Scripts\python.exe scripts\train_ieee.py --data-dir data\ieee --mode final --estimators 180 --output models\ieee_final_v2.joblib
```

Если выбран лимит ложных тревог 100 на 10 000, добавь `--max-false-alerts-per-10000 100` к **этой же команде**. Финальный запуск использует весь ранний train-сегмент, фиксированную validation и оценивает финальный test один раз. Результаты сохраняются в `models/ieee_final_v2.metrics.json`; артефакт содержит обученный preprocessing, модель, признаки и порог. Существующие отчёты 100k/200k/300k и `ieee_model.joblib` не перезаписываются. Для API укажи `IZQAR_MODEL_PATH` на новый доверенный артефакт и перезапусти сервер. Не перезапускай финальную оценку с изменёнными параметрами по её результатам.

Для Kaggle Notebook команда уже установлена в `notebooks/ieee_cis_kaggle.ipynb`; экспериментальные запуски можно добавить отдельными ячейками с различными путями `--output` до финальной ячейки. Ноутбук ещё не запускался на полном наборе в этом репозитории.

Ограничения памяти: `read_csv` загружает только выбранные столбцы, числа хранятся как `float32`, identity читается частями и фильтруется по нужным `TransactionID`, модель обучается на CPU. Для 16 ГБ RAM всё равно проверь доступную память перед запуском; более жёсткий лимит задаётся выбором train-подвыборки, а финальный test остаётся неизменным.
