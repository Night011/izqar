import { useEffect, useState } from 'react';
import { request } from './api';
import type { ModelDetail, ModelPrediction, ModelRegistry, ModelStatus, RegistryEntry } from './api';
import './ml-analytics.css';
import './model-registry.css';

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const examplePaySim = JSON.stringify({ type: 'TRANSFER', amount: 150000, oldbalanceOrg: 180000, oldbalanceDest: 20000 }, null, 2);
const names: Record<string, string> = { paysim: 'PaySim', ieee_cis: 'IEEE-CIS', baf_base: 'BAF Base', ibm_aml: 'IBM AML', izqar_unified_mtl: 'Unified Intelligence' };

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
  return <section className="panel ml-card"><div className="ml-card-head"><div><h2>Unified Intelligence · эксперимент</h2><p>{detail.model_version} · общий backbone и четыре независимые головы</p></div><span className="count-tag">validation only</span></div><p>Метрики ниже нельзя сопоставлять с ранее просмотренными финальными test специализированных моделей.</p><div className="ml-metrics">{Object.entries(detail.tasks).map(([task, item]) => <div key={task}><span>{names[task]}</span><strong>F1 {item.validation_metrics.f1.toFixed(3)}</strong><small>PR-AUC {item.validation_metrics.average_precision_pr_auc.toFixed(3)}</small></div>)}</div><p>Экспериментальный uncalibrated score. Отрицательный перенос проверяется сравнением на соответствующих validation-периодах.</p></section>;
}

export function MLAnalytics({ status, apiKey }: { status: ModelStatus | null; apiKey: string }) {
  const [kind, setKind] = useState<'paysim' | 'ieee_cis' | 'baf_base' | 'ibm_aml'>('paysim');
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
  const detail = status?.[kind];
  const changeKind = (next: 'paysim' | 'ieee_cis' | 'baf_base' | 'ibm_aml') => { setKind(next); setResult(null); setError(''); setInput(next === 'paysim' ? examplePaySim : ''); };
  const template = () => setInput(JSON.stringify(Object.fromEntries((detail?.feature_names ?? []).map(name => [name,
    kind === 'ibm_aml' ? ({ amount_paid: 150, log_amount_paid: Math.log1p(150), same_bank: 0, same_currency: 1, hour: 12, weekday: 0,
      receiving_currency: 'USD', payment_currency: 'USD', payment_format: 'Wire' } as Record<string, number | string>)[name] :
    name === 'TransactionAmt' ? 150 : name === 'ProductCD' ? 'W' : null])), null, 2));
  const predict = async () => {
    setBusy(true); setResult(null); setError('');
    try {
      const features = JSON.parse(input) as unknown;
      if (!features || typeof features !== 'object' || Array.isArray(features)) throw new Error('Введите JSON-объект признаков');
      const path = kind === 'paysim' ? '/api/v1/paysim/predict' : kind === 'baf_base' ? '/api/v1/baf/predict' : kind === 'ibm_aml' ? '/api/v1/ibm-aml/predict' : '/api/v1/ieee/predict';
      setResult(await request<ModelPrediction>(path, apiKey, { method: 'POST', body: JSON.stringify({ features }) }));
    } catch (exc) { setError((exc as Error).message); }
    finally { setBusy(false); }
  };
  return <div className="ml-page">
    <p className="ml-note">Метрики получены из сохранённых артефактов соответствующих датасетов. Демо-транзакции в журнале оценивает отдельный Rule Engine.</p>
    <div className="ml-grid"><ModelCard title="PaySim · синтетические переводы" detail={status?.paysim} error={status?.artifact_errors?.paysim} /><ModelCard title="IEEE-CIS · онлайн-платежи" detail={status?.ieee_cis} error={status?.artifact_errors?.ieee_cis} /><ModelCard title="BAF Base · синтетические заявки" detail={status?.baf_base} error={status?.artifact_errors?.baf_base} /><ModelCard title="IBM AML · синтетические операции" detail={status?.ibm_aml} error={status?.artifact_errors?.ibm_aml} /><UnifiedCard detail={status?.unified_mtl} error={status?.artifact_errors?.unified_mtl} /></div>
    <section className="panel ml-card"><h2>Model Registry</h2><p>Активные артефакты и история экспериментов. Контрольные суммы подтверждают идентичность файлов, а метрики взяты из сохранённых отчётов.</p>
      {registryError && <p className="ml-error">Ошибка реестра: {registryError}</p>}
      {registry && <><div className="registry-scroll"><table className="registry-table"><thead><tr><th>Модель</th><th>Версия</th><th>Состояние</th><th>Test</th><th>Аудит</th></tr></thead><tbody>
        {registry.entries.filter(entry => entry.active_for).map(entry => <RegistryRow key={`${entry.active_for}-${entry.manifest}`} entry={entry} />)}
      </tbody></table></div><details className="registry-history"><summary>История отчётов ({registry.entries.filter(entry => !entry.active_for).length})</summary><div className="registry-scroll"><table className="registry-table"><thead><tr><th>Модель</th><th>Версия</th><th>Состояние</th><th>Test</th><th>Аудит</th></tr></thead><tbody>
        {registry.entries.filter(entry => !entry.active_for).map(entry => <RegistryRow key={entry.manifest} entry={entry} />)}
      </tbody></table></div></details></>}
    </section>
    <section className="panel ml-card"><h2>Проверка ML-инференса</h2><p>Введите поля именно выбранной модели. Примерные значения в шаблоне служат только технической проверке API.</p>
      <div className="ml-actions"><select aria-label="Выбор ML-модели" value={kind} onChange={e => changeKind(e.target.value as 'paysim' | 'ieee_cis' | 'baf_base' | 'ibm_aml')}><option value="paysim">PaySim</option><option value="ieee_cis">IEEE-CIS</option><option value="baf_base">BAF Base</option><option value="ibm_aml">IBM AML</option></select><button type="button" onClick={template} disabled={!detail}>Шаблон признаков</button></div>
      <textarea aria-label="JSON признаков" spellCheck={false} value={input} onChange={e => setInput(e.target.value)} rows={9} />
      <button className="primary" disabled={!detail || busy} onClick={predict}>{busy ? 'Оцениваем…' : 'Получить оценку модели'}</button>
      {error && <p className="ml-error">{error}</p>}
      {result && <div className="ml-result"><strong>{result.flagged ? 'Выше порога' : 'Ниже порога'} · скор {result.risk_score.toFixed(2)} / 100</strong><span>{result.source} · {result.model_version} · порог {result.threshold.toFixed(4)}</span><p>{result.warning}</p></div>}
    </section>
  </div>;
}
