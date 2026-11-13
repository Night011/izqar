# IZQAR — продолжение работы

## Сначала: Unified Intelligence в Kaggle

1. Notebook уже выполнен в Kaggle на commit `6f91892ac1062425f2a34a822dd617ee38f3b8cc`; не перезапускайте его ради подбора параметров.
2. Через **Add Input** подключите: `ealaxi/paysim1`, `ieee-fraud-detection`, `sgpjesus/bank-account-fraud-dataset-neurips-2022`, `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`. Условия IEEE-CIS принимает только владелец аккаунта.
3. Включите Internet только для клонирования feature-ветки и зависимостей. Исходные CSV не скачивать в локальный репозиторий.
4. В завершённой сессии скачайте из `/kaggle/working/izqar_unified_mtl_v1` `unified_weights.npz`, `unified_preprocessors.joblib`, `unified.metrics.json` и `unified_checkpoint.pt`. При необходимости создайте ZIP только из этих четырёх файлов.
5. Сохраните вывод ячеек, commit SHA, размеры выборок, validation метрики, `gradient_audit_shared_backbone_l2`, `training.seconds` и checkpoint. До сравнения не называйте модель лучше специализированных.

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

## Текущий handoff

Kaggle вкладка оставлена открытой с успешным validation-only output и четырьмя файлами в `/kaggle/working/izqar_unified_mtl_v1`. Локально требуется только доверенный экспорт артефактов; `.vscode/` остаётся пользовательским неотслеживаемым каталогом.
