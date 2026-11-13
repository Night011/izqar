# IZQAR — продолжение работы

## Сначала: Unified Intelligence в Kaggle

1. Импортируйте `notebooks/unified_intelligence_kaggle.ipynb` в Kaggle.
2. Через **Add Input** подключите: `ealaxi/paysim1`, `ieee-fraud-detection`, `sgpjesus/bank-account-fraud-dataset-neurips-2022`, `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`. Условия IEEE-CIS принимает только владелец аккаунта.
3. Включите Internet только для клонирования feature-ветки и зависимостей. Исходные CSV не скачивать в локальный репозиторий.
4. Выполните notebook. Он использует validation-only периоды и создаёт `unified_weights.npz`, `unified_preprocessors.joblib`, `unified.metrics.json` в `/kaggle/working/izqar_unified_mtl_v1`.
5. Сохраните вывод ячеек, commit SHA, размеры выборок, validation метрики, `gradient_audit_shared_backbone_l2`, `training.seconds` и `unified_checkpoint.pt`. До сравнения не называйте модель лучше специализированных.

## После экспорта

1. Скопируйте только доверенные три файла в `backend/models/` с именами `unified_weights.npz`, `unified_preprocessors.joblib`, `unified.metrics.json`.
2. Не меняйте `.env`, если используется путь по умолчанию. Для другого манифеста укажите `IZQAR_UNIFIED_MODEL_PATH=<путь к unified.metrics.json>`.
3. Запустите:

```powershell
cd C:\Users\a\Documents\IZQAR-MVP\backend
.\.venv\Scripts\python.exe -m pytest -q
```

4. Проверьте `GET /api/v1/model/status` и отдельный `POST /api/v1/unified/<task>/predict` с точной схемой `task_metadata.<task>.feature_schema` из манифеста.

## Локальные проверки и запуск

```powershell
cd C:\Users\a\Documents\IZQAR-MVP
powershell -ExecutionPolicy Bypass -File .\scripts\start_local.ps1 -ConfigureOnly
cd backend
.\.venv\Scripts\python.exe -m pytest -q
cd ..\frontend
npm run build
```

## Незакоммиченные на момент этого документа

Unified training script, Kaggle notebook, Unified API integration, статусы и тесты. Перед следующей крупной задачей нужно завершить проверки, создать Conventional Commit и push feature-ветки.
