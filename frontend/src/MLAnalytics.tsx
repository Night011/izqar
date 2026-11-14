import { useEffect, useState } from 'react';
import { request } from './api';
import type { ModelDetail, ModelPrediction, ModelRegistry, ModelStatus, RegistryEntry } from './api';
import './ml-analytics.css';
import './model-registry.css';

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const examplePaySim = JSON.stringify({ type: 'TRANSFER', amount: 150000, oldbalanceOrg: 180000, oldbalanceDest: 20000 }, null, 2);
const names: Record<string, string> = { paysim: 'PaySim', ieee_cis: 'IEEE-CIS', baf_base: 'BAF Base', ibm_aml: 'IBM AML', izqar_unified_mtl: 'Unified Intelligence' };
type SpecialistKind = 'paysim' | 'ieee_cis' | 'baf_base' | 'ibm_aml';
type SelectedModel = SpecialistKind | 'unified_paysim' | 'unified_ieee_cis' | 'unified_baf_base' | 'unified_ibm_aml';
const unifiedTask = (kind: SelectedModel): SpecialistKind | null => kind.startsWith('unified_') ? kind.slice('unified_'.length) as SpecialistKind : null;

function RegistryRow({ entry }: { entry: RegistryEntry }) {
  const label = entry.state === 'installed' ? 'Артефакт установлен' : entry.state === 'report_only' ? 'Только отчёт' : 'Ошибка метаданных';
  return <tr><td><strong>{names[entry.model_kind ?? entry.active_for ?? ''] ?? entry.model_kind ?? 'Неизвестная модель'}</strong><small>{entry.active_for ? 'Активная' : 'История'}</small></td>
    <td>{entry.model_version ?? '—'}</td><td>{label}{entry.error && <small>{entry.error}</small>}</td>
    <td>{entry.test_metrics ? `F1 ${entry.test_metrics.f1.toFixed(3)} · PR-AUC ${entry.test_metrics.average_precision_pr_auc.toFixed(3)}` : 'Нет test-метрик'}</td>
    <td><details><summary>Происхождение</summary><p>{entry.dataset ?? 'Источник не указан'}</p><p>Манифест: {entry.manifest}</p><p>SHA-256: {entry.manifest_sha256 ?? '—'}</p>
      {entry.data_files?.map(file => <p key={file.filename}>Данные: {file.filename} · SHA-256 {file.sha256 ?? 'не указан'}</p>)}
      {entry.components?.map(file => <p key={file.filename}>Компонент: {file.filename} · SHA-256 {file.sha256}</p>)}</details></td></tr>;
}

function ModelCard({ title, detail, error }: { title: string; detail?: ModelDetail | null; error?: string | null }) {
  if (!detail) return <section className="panel ml-card"><h2>{title}</h2><p>{error ? `Ошибка артефакта: ${error}` : 'Обученный артефакт не установлен'}</p></section>;
  const m = detail.test_metrics;
  return <section className="panel ml-card">
    <div className="ml-card-head"><div><h2>{title}</h2><p>{detail.dataset}</p></div><span className="count-tag">{detail.model_version}</span></div>
    <p>Обучена: {detail.trained_at_utc || 'не указано'} · Признаков: {detail.features}</p>
    {m ? <><div className="ml-metrics">
      <div><span>Precision</span><strong>{pct(m.precision)}</strong></div><div><span>Recall</span><strong>{pct(m.recall)}</strong></div>
      <div><span>F1</span><strong>{pct(m.f1)}</strong></div><div><span>PR-AUC</span><strong>{pct(m.average_precision_pr_auc)}</strong></div>
      <div><span>ROC-AUC</span><strong>{pct(m.roc_auc)}</strong></div><div><span>Ложных тревог / 10k</span><strong>{m.false_alerts_per_10000_operations.toFixed(1)}</strong></div>
    </div><p>Отложенный test: {m.cases.toLocaleString('ru-RU')} записей · матрица TN / FP / FN / TP: {m.confusion_matrix_tn_fp_fn_tp.join(' / ')}</p></>
      : <p>Проверенных test-метрик в артефакте нет.</p>}
    <p>Порог решения: {detail.threshold.toFixed(4)} · Скор не откалиброван как вероятность.</p>
    {detail.feature_importance.length > 0 && <details><summary>Важность признаков XGBoost</summary><ol>{detail.feature_importance.map(item => <li key={item.feature}>{item.feature}: {item.importance.toFixed(4)}</li>)}</ol><p>Важность не доказывает причинность и отсутствие скрытой утечки.</p></details>}
  </section>;
}

function UnifiedCard({ detail, error }: { detail?: import('./api').UnifiedDetail | null; error?: string | null }) {
  if (!detail) return <section className="panel ml-card"><h2>Unified Intelligence · эксперимент</h2><p>{error ? `Ошибка артефакта: ${error}` : 'Совместно обученный артефакт ещё не установлен'}</p><p>Четыре специализированные модели остаются рабочими детекторами.</p></section>;
  const useFinal = detail.final_evaluation_available && Object.values(detail.tasks).every(item => item.final_test_metrics);
  return <section className="panel ml-card"><div className="ml-card-head"><div><h2>Unified Intelligence · эксперимент</h2><p>{detail.model_version} · общий backbone и четыре независимые головы</p></div><span className="count-tag">{useFinal ? 'one-time final test' : 'validation only'}</span></div><p>{useFinal ? 'Ниже показана однократная финальная оценка с порогами, уже выбранными на validation. Её нельзя использовать для новой настройки модели.' : 'Метрики ниже получены на validation-периодах и не сопоставимы с ранее просмотренными финальными test специализированных моделей.'}</p><div className="registry-scroll"><table className="registry-table"><thead><tr><th>Задача</th><th>Precision</th><th>Recall</th><th>F1</th><th>PR-AUC</th><th>ROC-AUC</th><th>FPR / FNR</th><th>Ложн. тревоги / 10k</th><th>TN / FP / FN / TP</th></tr></thead><tbody>{Object.entries(detail.tasks).map(([task, item]) => { const m = useFinal ? item.final_test_metrics! : item.validation_metrics; return <tr key={task}><td>{names[task] ?? task}<small>{m.cases.toLocaleString('ru-RU')} {useFinal ? 'test' : 'validation'} записей</small></td><td>{pct(m.precision)}</td><td>{pct(m.recall)}</td><td>{pct(m.f1)}</td><td>{pct(m.average_precision_pr_auc)}</td><td>{pct(m.roc_auc)}</td><td>{pct(m.false_positive_rate)} / {pct(m.false_negative_rate)}</td><td>{m.false_alerts_per_10000_operations.toFixed(1)}</td><td>{m.confusion_matrix_tn_fp_fn_tp.join(' / ')}</td></tr>; })}</tbody></table></div>{detail.preprocessing_compatibility && <p>Preprocessing загружен через проверенный адаптер совместимости {detail.preprocessing_compatibility}; модель не переобучалась.</p>}{detail.final_evaluation_error && <p className="ml-error">{detail.final_evaluation_error}</p>}<p>Экспериментальный некалиброванный ML-скор; графовые и rule-based сигналы с ним не объединяются. Отрицательный перенос проверяется сравнением на соответствующих периодах.</p></section>;
}

function ModelComparison({ status }: { status: ModelStatus | null }) {
  const unified = status?.unified_mtl;
  if (!unified) return <section className="panel ml-card"><h2>Model Comparison</h2><p>Сравнение появится после установки проверенного Unified-артефакта. Специализированные модели остаются основными рабочими детекторами.</p></section>;
  const useFinal = unified.final_evaluation_available && Object.values(unified.tasks).every(item => item.final_test_metrics);
  const rows = (['paysim', 'ieee_cis', 'baf_base', 'ibm_aml'] as SpecialistKind[]).map(task => ({
    task, specialist: status?.[task]?.test_metrics ?? null,
    unified: useFinal ? unified.tasks[task].final_test_metrics! : unified.tasks[task].validation_metrics,
  }));
  return <section className="panel ml-card"><div className="ml-card-head"><div><h2>Model Comparison</h2><p>Рядом приведены сохранённые метрики соответствующего источника, без смешивания задач.</p></div><span className="count-tag">{useFinal ? 'Unified final' : 'Unified validation'}</span></div><div className="registry-scroll"><table className="registry-table comparison-table"><thead><tr><th>Источник</th><th>Специализированная</th><th>Unified Intelligence</th><th>Статус сопоставления</th></tr></thead><tbody>{rows.map(row => <tr key={row.task}><td>{names[row.task]}</td><td>{row.specialist ? <><strong>F1 {row.specialist.f1.toFixed(3)} · PR-AUC {row.specialist.average_precision_pr_auc.toFixed(3)}</strong><small>test · {row.specialist.cases.toLocaleString('ru-RU')} записей</small></> : 'Нет проверенных test-метрик'}</td><td><strong>F1 {row.unified.f1.toFixed(3)} · PR-AUC {row.unified.average_precision_pr_auc.toFixed(3)}</strong><small>{useFinal ? 'one-time final test' : 'validation'} · {row.unified.cases.toLocaleString('ru-RU')} записей</small></td><td><span className="comparison-hold">Нужен одинаковый frozen holdout</span></td></tr>)}</tbody></table></div><p>Числа не объявляются «победой» одной модели: текущие артефакты могут использовать разные временные границы и объёмы. Вывод о положительном или отрицательном transfer допустим только после независимой оценки на идентичной, заранее зафиксированной выборке каждой задачи.</p></section>;
}

export function MLAnalytics({ status, apiKey }: { status: ModelStatus | null; apiKey: string }) {
  const [kind, setKind] = useState<SelectedModel>('paysim');
  const [input, setInput] = useState(examplePaySim);
  const [result, setResult] = useState<ModelPrediction | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [registry, setRegistry] = useState<ModelRegistry | null>(null);
  const [registryError, setRegistryError] = useState('');
  useEffect(() => {
    let mounted = true;
    request<ModelRegistry>('/api/v1/model/registry', apiKey).then(data => { if (mounted) { setRegistry(data); setRegistryError(''); } })
      .catch((exc: Error) => { if (mounted) setRegistryError(exc.message); });
    return () => { mounted = false; };
  }, [apiKey, status]);
  const selectedUnifiedTask = unifiedTask(kind);
  const specialistKind: SpecialistKind | null = selectedUnifiedTask ? null : kind as SpecialistKind;
  const detail = specialistKind ? status?.[specialistKind] : null;
  const selectedFeatures = selectedUnifiedTask ? status?.unified_mtl?.task_schemas[selectedUnifiedTask] ?? [] : detail?.feature_names ?? [];
  const changeKind = (next: SelectedModel) => { setKind(next); setResult(null); setError(''); setInput(next === 'paysim' ? examplePaySim : ''); };
  const template = () => setInput(JSON.stringify(Object.fromEntries(selectedFeatures.map(name => [name,
    kind === 'ibm_aml' ? ({ amount_paid: 150, log_amount_paid: Math.log1p(150), same_bank: 0, same_currency: 1, hour: 12, weekday: 0,
      receiving_currency: 'USD', payment_currency: 'USD', payment_format: 'Wire' } as Record<string, number | string>)[name] :
    name === 'TransactionAmt' ? 150 : name === 'ProductCD' ? 'W' : null])), null, 2));
  const predict = async () => {
    setBusy(true); setResult(null); setError('');
    try {
      const features = JSON.parse(input) as unknown;
      if (!features || typeof features !== 'object' || Array.isArray(features)) throw new Error('Введите JSON-объект признаков');
      const path = selectedUnifiedTask ? `/api/v1/unified/${selectedUnifiedTask}/predict` : kind === 'paysim' ? '/api/v1/paysim/predict' : kind === 'baf_base' ? '/api/v1/baf/predict' : kind === 'ibm_aml' ? '/api/v1/ibm-aml/predict' : '/api/v1/ieee/predict';
      setResult(await request<ModelPrediction>(path, apiKey, { method: 'POST', body: JSON.stringify({ features }) }));
    } catch (exc) { setError((exc as Error).message); }
    finally { setBusy(false); }
  };
  return <div className="ml-page">
    <p className="ml-note">Метрики получены из сохранённых артефактов соответствующих датасетов. Демо-транзакции в журнале оценивает отдельный Rule Engine.</p>
    <div className="ml-grid"><ModelCard title="PaySim · синтетические переводы" detail={status?.paysim} error={status?.artifact_errors?.paysim} /><ModelCard title="IEEE-CIS · онлайн-платежи" detail={status?.ieee_cis} error={status?.artifact_errors?.ieee_cis} /><ModelCard title="BAF Base · синтетические заявки" detail={status?.baf_base} error={status?.artifact_errors?.baf_base} /><ModelCard title="IBM AML · синтетические операции" detail={status?.ibm_aml} error={status?.artifact_errors?.ibm_aml} /><UnifiedCard detail={status?.unified_mtl} error={status?.artifact_errors?.unified_mtl} /></div>
    <ModelComparison status={status} />
    <section className="panel ml-card"><h2>Model Registry</h2><p>Активные артефакты и история экспериментов. Контрольные суммы подтверждают идентичность файлов, а метрики взяты из сохранённых отчётов.</p>
      {registryError && <p className="ml-error">Ошибка реестра: {registryError}</p>}
      {registry && <><div className="registry-scroll"><table className="registry-table"><thead><tr><th>Модель</th><th>Версия</th><th>Состояние</th><th>Test</th><th>Аудит</th></tr></thead><tbody>
        {registry.entries.filter(entry => entry.active_for).map(entry => <RegistryRow key={`${entry.active_for}-${entry.manifest}`} entry={entry} />)}
      </tbody></table></div><details className="registry-history"><summary>История отчётов ({registry.entries.filter(entry => !entry.active_for).length})</summary><div className="registry-scroll"><table className="registry-table"><thead><tr><th>Модель</th><th>Версия</th><th>Состояние</th><th>Test</th><th>Аудит</th></tr></thead><tbody>
        {registry.entries.filter(entry => !entry.active_for).map(entry => <RegistryRow key={entry.manifest} entry={entry} />)}
      </tbody></table></div></details></>}
    </section>
    <section className="panel ml-card"><h2>Проверка ML-инференса</h2><p>Введите поля именно выбранной модели. Шаблон заполнен значениями для проверки схемы, а не реалистичной банковской операцией.</p>
      <div className="ml-actions"><select aria-label="Выбор ML-модели" value={kind} onChange={e => changeKind(e.target.value as SelectedModel)}><option value="paysim">PaySim</option><option value="ieee_cis">IEEE-CIS</option><option value="baf_base">BAF Base</option><option value="ibm_aml">IBM AML</option>{status?.unified_mtl && <optgroup label="Unified Intelligence · эксперимент"><option value="unified_paysim">Unified · PaySim</option><option value="unified_ieee_cis">Unified · IEEE-CIS</option><option value="unified_baf_base">Unified · BAF Base</option><option value="unified_ibm_aml">Unified · IBM AML</option></optgroup>}</select><button type="button" onClick={template} disabled={selectedFeatures.length === 0}>Шаблон признаков</button></div>
      <textarea aria-label="JSON признаков" spellCheck={false} value={input} onChange={e => setInput(e.target.value)} rows={9} />
      <button className="primary" disabled={selectedFeatures.length === 0 || busy} onClick={predict}>{busy ? 'Оцениваем…' : 'Получить оценку модели'}</button>
      {error && <p className="ml-error">{error}</p>}
      {result && <div className="ml-result"><strong>{result.flagged ? 'Выше порога' : 'Ниже порога'} · скор {result.risk_score.toFixed(2)} / 100</strong><span>{result.source} · {result.model_version} · порог {result.threshold.toFixed(4)}</span><p>{result.warning}</p></div>}
    </section>
  </div>;
}
