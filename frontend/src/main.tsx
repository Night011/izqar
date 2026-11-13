import { useCallback, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, AlertTriangle, ArrowDownLeft, ArrowRight, ArrowUpRight, BarChart3, Check, ChevronRight, CircleHelp, Clock, Cpu, Fingerprint, GitBranch, KeyRound, LockKeyhole, LogOut, Play, RefreshCcw, Search, Shield, ShieldAlert, ShieldCheck, Wallet, X } from 'lucide-react';
import { Graph } from './Graph';
import { MLAnalytics } from './MLAnalytics';
import { money, request, shortDate } from './api';
import type { GraphData, Metrics, ModelStatus, Review, Scenario, Transaction } from './api';
import './styles.css';
import './model-summary.css';

const scenarios: { type: Scenario; label: string; description: string }[] = [
  { type: 'normal', label: 'Обычная оплата', description: 'Небольшая покупка' },
  { type: 'new_recipient', label: 'Подозрительный перевод', description: 'Крупная сумма, новое устройство' },
  { type: 'velocity', label: 'Серия переводов', description: 'Много операций за короткое время' },
  { type: 'fan_in', label: 'Сеть получателей', description: 'Несколько отправителей → один счёт' },
  { type: 'chain', label: 'Цепочка переводов', description: 'Последовательное движение средств' },
];
const riskLabels = { low: 'Низкий риск', medium: 'Средний риск', high: 'Высокий риск' };
const decisionLabels: Record<string, string> = { allow: 'Разрешено', step_up: 'Доп. проверка', review: 'На рассмотрении' };
const reviewLabels: Record<Review, string> = { pending: 'Не проверено', confirmed_fraud: 'Подтверждено аналитиком', false_positive: 'Ложная тревога' };
const blank: Metrics = { transactions: 0, simulated_volume_kzt: 0, high: 0, medium: 0, low: 0, reviewed: 0, disclaimer: '' };
const emptyGraph: GraphData = { nodes: [], edges: [], indicators: { high_fan_in_accounts: [], high_fan_out_accounts: [], rapid_transfer_chains: [], window_minutes: 1440, note: '' } };

function App() {
  const [key, setKey] = useState(() => sessionStorage.getItem('izqar-api-key') ?? '');
  const [draftKey, setDraftKey] = useState('');
  const [connected, setConnected] = useState(false);
  const [metrics, setMetrics] = useState<Metrics>(blank);
  const [rows, setRows] = useState<Transaction[]>([]);
  const [graph, setGraph] = useState<GraphData>(emptyGraph);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [selected, setSelected] = useState<Transaction | null>(null);
  const [tab, setTab] = useState<'overview' | 'network' | 'ml'>('overview');
  const [busy, setBusy] = useState<Scenario | null>(null);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [search, setSearch] = useState('');

  const refresh = useCallback(async (secret: string) => {
    const [m, t, g, ml] = await Promise.all([
      request<Metrics>('/api/v1/metrics', secret),
      request<Transaction[]>('/api/v1/transactions?limit=100', secret),
      request<GraphData>('/api/v1/graph', secret),
      request<ModelStatus>('/api/v1/model/status', secret),
    ]);
    setMetrics(m); setRows(t); setGraph(g); setModel(ml);
    setSelected(previous => previous ? t.find(tx => tx.id === previous.id) ?? null : null);
    setConnected(true);
    setError('');
  }, []);

  useEffect(() => { if (key) refresh(key).catch((err: Error) => { setError(err.message); setConnected(false); }); }, [key, refresh]);

  const login = () => {
    const next = draftKey.trim();
    sessionStorage.setItem('izqar-api-key', next);
    setKey(next);
  };
  const logout = () => { sessionStorage.removeItem('izqar-api-key'); setKey(''); setConnected(false); setDraftKey(''); setError(''); };
  const simulate = async (type: Scenario) => {
    setBusy(type); setError('');
    try {
      const added = await request<Transaction[]>('/api/v1/demo/simulate', key, { method: 'POST', body: JSON.stringify({ scenario: type }) });
      await refresh(key);
      setSelected(added[added.length - 1]);
    } catch (err) { setError((err as Error).message); }
    finally { setBusy(null); }
  };
  const updateReview = async (tx: Transaction, state: Review) => {
    try {
      const updated = await request<Transaction>(`/api/v1/transactions/${tx.id}/review`, key,
        { method: 'PATCH', body: JSON.stringify({ review_state: state }) });
      setSelected(updated); await refresh(key);
    } catch (err) { setError((err as Error).message); }
  };
  const visibleRows = rows.filter(tx => (filter === 'all' || tx.risk_level === filter)
    && `${tx.sender} ${tx.recipient} ${tx.scenario}`.toLowerCase().includes(search.toLowerCase()));

  if (!connected) return <div className="login-page">
    <div className="login-glow" /><div className="login-card">
      <Brand large />
      <div className="eyebrow">ANALYST ACCESS / LOCAL DEMO</div>
      <h1>Интеллектуальная защита финансовых операций</h1>
      <p>Демо-консоль IZQAR. Для доступа введите API-ключ из файла <code>.env</code>.</p>
      <label htmlFor="secret">Ключ доступа</label>
      <div className="secret-field"><KeyRound size={17} /><input autoComplete="off" id="secret" type="password" value={draftKey} onChange={e => setDraftKey(e.target.value)} onKeyDown={e => e.key === 'Enter' && login()} placeholder="IZQAR_API_KEY" /></div>
      <button className="primary wide" onClick={login} disabled={!draftKey.trim()}>Открыть панель <ArrowRight size={17} /></button>
      {error && <p className="error">{error}</p>}
      <p className="login-footnote"><LockKeyhole size={13} /> Секрет хранится только в текущей сессии браузера. Не публикуйте демо в интернете.</p>
    </div>
  </div>;

  return <div className="app-shell">
    <aside className="sidebar">
      <Brand />
      <div className="side-label">РАБОЧЕЕ ПРОСТРАНСТВО</div>
      <nav>
        <button className={'nav-item ' + (tab === 'overview' ? 'active' : '')} onClick={() => setTab('overview')}><BarChart3 size={18} /> Обзор системы <ChevronRight size={15} /></button>
        <button className={'nav-item ' + (tab === 'network' ? 'active' : '')} onClick={() => setTab('network')}><GitBranch size={18} /> Граф связей <ChevronRight size={15} /></button>
        <button className={'nav-item ' + (tab === 'ml' ? 'active' : '')} onClick={() => setTab('ml')}><Cpu size={18} /> ML Analytics <ChevronRight size={15} /></button>
      </nav>
      <div className="side-divider" />
      <div className="side-label">СТАТУС СИСТЕМЫ</div>
      <div className="system-row"><span className="status-dot" /> API-сервис <span>Активен</span></div>
      <div className="system-row"><span className="status-dot" /> Rule Engine <span>Активен</span></div>
      <div className="system-row"><span className={'status-dot ' + (model?.ieee_cis ? '' : 'amber')} /> IEEE-CIS ML <span>{model?.ieee_cis ? 'Обучена' : 'Нет модели'}</span></div>
      <div className="system-row"><span className={'status-dot ' + (model?.paysim ? '' : 'amber')} /> PaySim ML <span>{model?.paysim ? 'Обучена' : 'Нет модели'}</span></div>
      <div className="system-row"><span className={'status-dot ' + (model?.baf_base ? '' : 'amber')} /> BAF Base ML <span>{model?.baf_base ? 'Обучена' : 'Нет модели'}</span></div>
      <div className="side-bottom"><div className="mini-shield"><ShieldCheck size={17} /></div><div><strong>Hackathon MVP</strong><span>Локальная симуляция</span></div></div>
    </aside>

    <div className="main-frame">
      <header className="topbar">
        <div className="breadcrumb">Платформа <ChevronRight size={14} /> <strong>{tab === 'overview' ? 'Обзор системы' : tab === 'network' ? 'Граф связей' : 'ML Analytics'}</strong></div>
        <div className="top-actions"><span className="demo-tag"><span className="status-dot" /> DEMO ENVIRONMENT</span><button title="Обновить" className="icon-button" onClick={() => refresh(key).catch((e: Error) => setError(e.message))}><RefreshCcw size={18} /></button><button title="Выйти" className="icon-button" onClick={logout}><LogOut size={18} /></button></div>
      </header>
      <main className="content">
        <div className="page-heading"><div><div className="eyebrow">IZQAR / FRAUD INTELLIGENCE</div><h1>{tab === 'overview' ? 'Центр мониторинга' : tab === 'network' ? 'Анализ связей' : 'Аналитика моделей'}</h1><p>{tab === 'overview' ? 'Оценка рисков и расследование подозрительных операций.' : tab === 'network' ? 'Исследование структуры переводов между синтетическими счетами.' : 'Фактические тестовые метрики и отдельные ML-прогнозы.'}</p></div><span className="date-stamp"><Clock size={15} /> Демо-режим · KZT</span></div>
        <div className="disclaimer"><CircleHelp size={16} /> <span><strong>Исследовательский прототип.</strong> Все операции синтетические. Оценки в таблице рассчитаны правилами, не ML-моделью. Банк и реальные клиенты не подключены.</span></div>
        {error && <div className="error-banner"><AlertTriangle size={17}/>{error}<button onClick={() => setError('')} aria-label="Закрыть"><X size={15}/></button></div>}

        {tab === 'overview' ? <>
          <div className="stats-grid">
            <Stat label="Обработано переводов" value={String(metrics.transactions)} info="Синтетические операции" icon={<Activity size={19}/>} />
            <Stat label="Высокий риск" value={String(metrics.high)} info="Требуют рассмотрения" icon={<ShieldAlert size={19}/>} danger />
            <Stat label="Средний риск" value={String(metrics.medium)} info="Дополнительная проверка" icon={<Fingerprint size={19}/>} />
            <Stat label="Сумма переводов" value={money(metrics.simulated_volume_kzt)} info="Демо-объём, не потери" icon={<Wallet size={19}/>} />
          </div>
          <div className="panel-grid">
            <section className="panel">
              <div className="panel-header"><div><h2>Симулятор операций</h2><p>Проверьте, как система оценивает разные сценарии</p></div><Play size={18} className="soft-icon" /></div>
              <div className="scenarios">{scenarios.map(s => <button key={s.type} disabled={busy !== null} className="scenario" onClick={() => simulate(s.type)}><span className="scenario-play"><Play size={13} fill="currentColor" /></span><span><strong>{s.label}</strong><small>{s.description}</small></span><ArrowUpRight size={16} className="scenario-arrow" /></button>)}</div>
            </section>
            <section className="panel">
              <div className="panel-header"><div><h2>Распределение рисков</h2><p>По выполненным сценариям</p></div><BarChart3 size={18} className="soft-icon" /></div>
              <div className="distribution">
                {(['low','medium','high'] as const).map(level => {
                  const count = metrics[level]; const width = metrics.transactions ? 100 * count / metrics.transactions : 0;
                  return <div className="dist-row" key={level}><div className="dist-label"><span>{riskLabels[level]}</span><strong>{count}</strong></div><div className="dist-track"><div className={'dist-fill ' + level} style={{ width: `${width}%` }} /></div></div>;
                })}
              </div>
              <div className="engine-info"><Cpu size={20} /><div><strong>Демонстрационные правила активны</strong><span>Оценка операций в таблице рассчитана Rule Engine.</span></div></div>
              <div className="model-summary"><strong>ML-модели · отдельная оценка</strong>{(['paysim', 'ieee_cis'] as const).map(kind => {
                const entry = model?.[kind];
                return <div key={kind}><span>{kind === 'paysim' ? 'PaySim' : 'IEEE-CIS'}</span><span>{entry ? `Доступна · ${entry.test_metrics ? `F1 test ${entry.test_metrics.f1.toFixed(3)}` : 'тестовых метрик нет'}` : 'Нет обученной модели'}</span></div>;
              })}<small>Метрики показываются только из обученных артефактов соответствующего датасета.</small></div>
            </section>
          </div>
          <section className="panel transactions-panel">
            <div className="panel-header"><div><h2>Последние операции</h2><p>Нажмите на запись, чтобы изучить объяснение риска</p></div><span className="count-tag">{visibleRows.length} операций</span></div>
            <div className="table-tools"><div className="search"><Search size={16} /><input aria-label="Поиск счёта" value={search} onChange={e => setSearch(e.target.value)} placeholder="Поиск счёта или сценария" /></div><div className="filter-row">{(['all','low','medium','high'] as const).map(f => <button className={'filter-btn '+(filter===f?'on':'')} key={f} onClick={() => setFilter(f)}>{f==='all'?'Все':riskLabels[f]}</button>)}</div></div>
            <div className="table-wrap"><table><thead><tr><th>ВРЕМЯ</th><th>ОТПРАВИТЕЛЬ / ПОЛУЧАТЕЛЬ</th><th>СУММА</th><th>РИСК</th><th>РЕШЕНИЕ</th><th /></tr></thead><tbody>{visibleRows.map(tx => <tr key={tx.id} onClick={() => setSelected(tx)} className="transaction-row"><td className="muted">{shortDate(tx.created_at)}</td><td><div className="account-flow">{tx.sender}<ArrowRight size={13}/>{tx.recipient}</div></td><td className="amount-cell">{money(tx.amount)}</td><td><span className={'risk-chip '+tx.risk_level}><i/>{riskLabels[tx.risk_level]} · {tx.risk_score}</span></td><td className="muted">{decisionLabels[tx.decision]}</td><td><ChevronRight size={16}/></td></tr>)}</tbody></table>{!visibleRows.length && <div className="empty-state"><Activity size={25}/><strong>Пока нет операций</strong><span>Запустите сценарий выше, чтобы начать демонстрацию.</span></div>}</div>
          </section>
        </> : tab === 'network' ? <>
          <div className="graph-summary"><Stat label="Счетов в выборке" value={String(graph.nodes.length)} info="Уникальные узлы" icon={<Fingerprint size={19}/>} /><Stat label="Направлений переводов" value={String(graph.edges.length)} info="Рёбра графа" icon={<GitBranch size={19}/>} /><Stat label="Высокий входящий поток" value={String(graph.indicators.high_fan_in_accounts.length)} info="3+ разных отправителя" icon={<ArrowDownLeft size={19}/>} /><Stat label="Высокий исходящий поток" value={String(graph.indicators.high_fan_out_accounts.length)} info="3+ разных получателя" icon={<ArrowUpRight size={19}/>} /></div>
          <section className="panel graph-panel"><div className="panel-header"><div><h2>Финансовая сеть</h2><p>Оранжевым отмечены узлы с повышенной входящей или исходящей связностью</p></div><span className="count-tag">NetworkX + Cytoscape</span></div>{graph.nodes.length ? <Graph data={graph} /> : <div className="empty-state big"><GitBranch size={27}/><strong>Нет связей</strong><span>В разделе «Обзор системы» запустите сценарий «Сеть получателей».</span></div>}</section>
          <div className="panel graph-note"><Shield size={20}/><p>{graph.indicators.note || 'Структурные сигналы графа не означают доказанное мошенничество.'}</p></div>
        </> : <MLAnalytics status={model} apiKey={key} />}
        <footer>IZQAR · Intelligent Fraud Prevention Platform <span>Research prototype v0.1.0</span></footer>
      </main>
    </div>
    {selected && <div className="drawer-overlay" onMouseDown={() => setSelected(null)}><aside className="drawer" onMouseDown={e => e.stopPropagation()}><div className="drawer-head"><div><div className="eyebrow">ТРАНЗАКЦИЯ / АНАЛИЗ</div><h2>Детали операции</h2></div><button className="icon-button" onClick={() => setSelected(null)} aria-label="Закрыть"><X size={20}/></button></div>
      <div className={'score-box '+selected.risk_level}><span>Демо-оценка риска по правилам</span><strong>{selected.risk_score}<small>/ 100</small></strong><div>{riskLabels[selected.risk_level]}</div></div>
      <div className="drawer-data"><div><span>Сумма</span><strong>{money(selected.amount)}</strong></div><div><span>Отправитель</span><strong>{selected.sender}</strong></div><div><span>Получатель</span><strong>{selected.recipient}</strong></div><div><span>Дата операции</span><strong>{shortDate(selected.created_at)}</strong></div><div><span>Решение системы</span><strong>{decisionLabels[selected.decision]}</strong></div><div><span>Статус проверки</span><strong>{reviewLabels[selected.review_state]}</strong></div></div>
      <h3>Почему система присвоила этот риск</h3><div className="reasons">{selected.reasons.map((r, i)=><div key={i}><AlertTriangle size={16}/>{r}</div>)}</div>
      <div className="drawer-disclaimer">Скор не является статистически откалиброванной вероятностью мошенничества. Решение демонстрационное.</div>
      <h3>Решение аналитика</h3><div className="review-actions"><button onClick={() => updateReview(selected, 'confirmed_fraud')}><ShieldAlert size={16}/> Подтвердить риск</button><button onClick={() => updateReview(selected, 'false_positive')}><Check size={16}/> Ложная тревога</button></div>
    </aside></div>}
  </div>;
}

function Brand({ large = false }: { large?: boolean }) { return <div className={'brand '+(large?'brand-large':'')}><div className="brand-icon"><Shield size={27} strokeWidth={1.9}/><span /></div><div><div className="brand-name">IZQAR<span>.</span></div><div className="brand-sub">FRAUD INTELLIGENCE</div></div></div>; }
function Stat({ label, value, info, icon, danger = false }: { label: string, value: string, info: string, icon: React.ReactNode, danger?: boolean }) { return <div className="stat"><div className="stat-top"><span>{label}</span><div className={danger ? 'stat-icon danger':'stat-icon'}>{icon}</div></div><strong>{value}</strong><small>{info}</small></div>; }
createRoot(document.getElementById('root')!).render(<App />);
