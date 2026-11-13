# IZQAR — статус разработки

Обновлено: 14 ноября 2026, Asia/Qyzylorda. Рабочая ветка: `feat/ml-pipelines-and-startup`; `main` не изменялась.

## Модели

| Компонент | Статус | Подтверждённый результат | Ограничение |
|---|---|---|---|
| PaySim v2 | Установлена, API проверен | test: F1 0.780426, PR-AUC 0.833058, 314 FP / 1 243 FN | Синтетические переводы; test ранее просматривался; скор не калиброван |
| IEEE-CIS v2 | Установлена, API проверен | test: F1 0.464247, PR-AUC 0.444182, 1 148 FP / 1 804 FN | Онлайн-платежи, не граф переводов; скор не калиброван |
| BAF Base v1 | Установлена, API проверен | test: F1 0.241141, PR-AUC 0.199692, 731 FP / 1 132 FN | Синтетические заявки; низкий Recall 0.207283 |
| IBM AML HI-Small v1 | Установлена, API проверен | test: F1 0.115342, PR-AUC 0.101293, 15 465 FP / 519 FN | На test 202.925 ложных тревог/10k против validation-бюджета 100; не менять порог по этому test |
| Unified Intelligence v1 | Код, CPU-inference и Kaggle notebook подготовлены; IEEE adapter включает проверяемый `one_to_one` identity merge; обучение не запускалось | Реальных метрик нет | Нужен Kaggle запуск четырёх входов; компонент экспериментальный |

## Платформа

- React: мониторинг, Model Registry, ML Analytics, интерактивный NetworkX/Cytoscape граф и рабочее место расследований. Граф ограниченно показывает fan-in/fan-out, быстрые цепочки и направленные трёхузловые циклы как сигналы, а не доказательства.
- Backend: четыре специализированных endpoint-инференса; очередь расследований, статусы, комментарии и append-only `investigation_events` для синтетических переводов.
- Unified endpoint: `POST /api/v1/unified/{paysim|ieee_cis|baf_base|ibm_aml}/predict` подготовлен, но честно возвращает 503 до установки совместно обученного артефакта.
- Локальная проверка: 36 backend-тестов прошли; React build прошёл. В том числе проверены Unified adapter protocol, NumPy shared-backbone inference, 503 при отсутствии артефакта и загрузка тестового доверенного экспорта.

## Последние проверенные коммиты

- `00421b7 fix: preserve existing demo secrets during startup`
- `e262abb feat: add portable unified network inference kernel`
- `b3d598b feat: add persistent investigation workspace and audit events`

## Не закрыто

1. Запустить и зафиксировать validation-only Unified MTL в Kaggle. Не использовать ранее просмотренные final test для выбора параметров.
2. Установить только доверенный экспорт, проверить Unified API и добавить его в Model Registry и React после наличия реальных метрик.
3. Продолжить validation-only исследования улучшения IBM AML, BAF, IEEE и PaySim; прежние финальные test не переиспользовать для выбора.
4. Ввести ML-оценки как самостоятельные сохраняемые инциденты без подмены их Rule Engine переводами.
5. Расширить временный граф IBM AML на данных Kaggle, не выдавая графовый сигнал за подтверждённое отмывание.
