import { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, ArrowRight, Check, Clock, GitBranch, MessageSquare, Search, ShieldAlert } from 'lucide-react';
import { Graph } from './Graph';
import { money, request, shortDate } from './api';
import type { GraphData, InvestigationDetail, InvestigationList, Review, Transaction } from './api';
import './investigations.css';

const stateText: Record<Review, string> = {
  pending: 'Ожидает проверки', in_review: 'Изучается', step_up: 'Доп. подтверждение',
  escalated: 'Передано на расследование', allowed: 'Разрешено аналитиком',
  confirmed_fraud: 'Подтверждено аналитиком', false_positive: 'Ложная тревога',
};
const actions: { state: Review; label: string }[] = [
  { state: 'in_review', label: 'Начать проверку' },
  { state: 'step_up', label: 'Запросить подтверждение' },
  { state: 'escalated', label: 'Передать на расследование' },
  { state: 'allowed', label: 'Разрешить' },
  { state: 'confirmed_fraud', label: 'Подтвердить мошенничество' },
  { state: 'false_positive', label: 'Ложная тревога' },
];

function caseGraph(detail: InvestigationDetail): GraphData {
  const transactions = [detail.transaction, ...detail.related_transactions];
  const adjacency = new Map<string, { in: Set<string>; out: Set<string> }>();
  const edges = new Map<string, GraphData['edges'][number]>();
  for (const tx of transactions) {
    for (const id of [tx.sender, tx.recipient]) {
      if (!adjacency.has(id)) adjacency.set(id, { in: new Set(), out: new Set() });
    }
    adjacency.get(tx.sender)!.out.add(tx.recipient);
    adjacency.get(tx.recipient)!.in.add(tx.sender);
    const key = `${tx.sender}\u0000${tx.recipient}`;
    const previous = edges.get(key);
    edges.set(key, previous ? {
      ...previous, count: previous.count + 1,
      total_amount: previous.total_amount + tx.amount,
      latest_at: previous.latest_at > tx.created_at ? previous.latest_at : tx.created_at,
    } : { source: tx.sender, target: tx.recipient, count: 1, total_amount: tx.amount, latest_at: tx.created_at });
  }
  const nodes = [...adjacency.entries()].map(([id, degree]) => ({
    id, in_degree: degree.in.size, out_degree: degree.out.size,
    flagged: degree.in.size >= 3 || degree.out.size >= 3,
  }));
  return {
    nodes, edges: [...edges.values()],
    indicators: {
      high_fan_in_accounts: nodes.filter(n => n.in_degree >= 3).map(n => n.id),
      high_fan_out_accounts: nodes.filter(n => n.out_degree >= 3).map(n => n.id),
      rapid_transfer_chains: [], window_minutes: 0,
      note: 'Показаны только операция и до 50 связанных записей из базы. Графовая связь не доказывает мошенничество.',
    },
  };
}

export function InvestigationWorkspace({ apiKey, initialId, onChanged }: {
  apiKey: string; initialId: string | null; onChanged: () => Promise<void>;
}) {
  const [list, setList] = useState<InvestigationList>({ total: 0, items: [] });
  const [detail, setDetail] = useState<InvestigationDetail | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(initialId);
  const [status, setStatus] = useState<Review | 'all'>('all');
  const [risk, setRisk] = useState<'all' | 'medium' | 'high'>('all');
  const [sort, setSort] = useState<'newest' | 'highest_risk'>('highest_risk');
  const [search, setSearch] = useState('');
  const [offset, setOffset] = useState(0);
  const [comment, setComment] = useState('');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadList = useCallback(async () => {
    const query = new URLSearchParams({ status, risk_level: risk, search, sort, limit: '50', offset: String(offset) });
    setList(await request<InvestigationList>(`/api/v1/investigations?${query}`, apiKey));
  }, [apiKey, status, risk, search, sort, offset]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setLoading(true);
      loadList().catch((e: Error) => setError(e.message)).finally(() => setLoading(false));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [loadList]);

  useEffect(() => {
    if (!selectedId) { setDetail(null); return; }
    setDetail(null);
    let active = true;
    request<InvestigationDetail>(`/api/v1/investigations/${encodeURIComponent(selectedId)}`, apiKey)
      .then(value => { if (active) setDetail(value); })
      .catch((e: Error) => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [apiKey, selectedId]);

  const decide = async (review_state: Review) => {
    if (!selectedId) return;
    setBusy(true); setError('');
    try {
      await request<Transaction>(`/api/v1/transactions/${encodeURIComponent(selectedId)}/review`, apiKey,
        { method: 'PATCH', body: JSON.stringify({ review_state }) });
      const updated = await request<InvestigationDetail>(`/api/v1/investigations/${encodeURIComponent(selectedId)}`, apiKey);
      setDetail(updated);
      await Promise.all([loadList(), onChanged()]);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const addComment = async () => {
    if (!selectedId || !comment.trim()) return;
    setBusy(true); setError('');
    try {
      await request(`/api/v1/investigations/${encodeURIComponent(selectedId)}/comments`, apiKey,
        { method: 'POST', body: JSON.stringify({ comment: comment.trim() }) });
      setDetail(await request<InvestigationDetail>(`/api/v1/investigations/${encodeURIComponent(selectedId)}`, apiKey));
      setComment('');
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };

  const graph = useMemo(() => detail ? caseGraph(detail) : null, [detail]);
  return <div className="investigation-layout">
    <section className="panel investigation-queue">
      <div className="panel-header"><div><h2>Очередь расследований</h2><p>Средний и высокий риск, а также принятые решения</p></div><span className="count-tag">{list.total}</span></div>
      <div className="investigation-filters">
        <label className="search"><Search size={15}/><input aria-label="Поиск расследования" value={search} onChange={e => { setSearch(e.target.value); setOffset(0); }} placeholder="Счёт или ID" /></label>
        <select aria-label="Статус расследования" value={status} onChange={e => { setStatus(e.target.value as Review | 'all'); setOffset(0); }}>
          <option value="all">Все статусы</option>{Object.entries(stateText).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
        <select aria-label="Уровень риска" value={risk} onChange={e => { setRisk(e.target.value as typeof risk); setOffset(0); }}>
          <option value="all">Любой риск</option><option value="high">Высокий</option><option value="medium">Средний</option>
        </select>
        <select aria-label="Сортировка расследований" value={sort} onChange={e => { setSort(e.target.value as typeof sort); setOffset(0); }}>
          <option value="highest_risk">По риску</option><option value="newest">Сначала новые</option>
        </select>
      </div>
      {loading ? <div className="investigation-empty">Загрузка очереди…</div> : list.items.length ?
        <div className="case-list">{list.items.map(tx => <button key={tx.id} className={'case-row '+(selectedId === tx.id ? 'active' : '')} onClick={() => setSelectedId(tx.id)}>
          <span><strong>{tx.sender} <ArrowRight size={12}/> {tx.recipient}</strong><small>{shortDate(tx.created_at)} · {money(tx.amount)}</small></span>
          <span className={'risk-chip '+tx.risk_level}>{tx.risk_score}</span><small>{stateText[tx.review_state]}</small>
        </button>)}</div> : <div className="investigation-empty">Подозрительных операций по фильтру нет. Запустите сценарий в обзоре.</div>}
      <div className="case-pages"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>Назад</button><span>{list.total ? `${offset + 1}–${Math.min(offset + 50, list.total)} из ${list.total}` : '0'}</span><button disabled={offset + 50 >= list.total} onClick={() => setOffset(offset + 50)}>Далее</button></div>
    </section>
    <section className="panel investigation-detail">
      {!detail ? <div className="investigation-empty">Выберите операцию для расследования.</div> : <>
        <div className="panel-header"><div><h2>Инцидент · {detail.transaction.id.slice(0, 8)}</h2><p>Синтетический перевод · оценка Rule Engine</p></div><span className={'risk-chip '+detail.transaction.risk_level}>{detail.transaction.risk_score}/100</span></div>
        <div className="case-body">
          <div className="case-meta"><span>Направление</span><strong>{detail.transaction.sender} → {detail.transaction.recipient}</strong><span>Сумма</span><strong>{money(detail.transaction.amount)}</strong><span>Статус</span><strong>{stateText[detail.transaction.review_state]}</strong><span>Рекомендация правил</span><strong>{detail.transaction.decision}</strong></div>
          <h3><ShieldAlert size={16}/> Причины сигнала</h3>
          <ul className="case-reasons">{detail.transaction.reasons.map((reason, index) => <li key={index}>{reason}</li>)}</ul>
          <p className="case-caveat">Риск рассчитан демонстрационными правилами. Наличие сигнала или связи не подтверждает мошенничество.</p>
          <h3><GitBranch size={16}/> Связанные счета и операции</h3>
          {graph && <Graph data={graph} />}
          <div className="case-related">{detail.related_transactions.length ? detail.related_transactions.map(tx => <button key={tx.id} onClick={() => setSelectedId(tx.id)}>
            <span>{tx.sender} → {tx.recipient}</span><span>{money(tx.amount)} · {shortDate(tx.created_at)}</span>
          </button>) : <p>Других операций по этим счетам в базе нет.</p>}</div>
          <h3><Check size={16}/> Решение аналитика</h3>
          <div className="case-actions">{actions.map(action => <button key={action.state} disabled={busy} onClick={() => decide(action.state)}>{action.label}</button>)}</div>
          <h3><MessageSquare size={16}/> Комментарий</h3>
          <div className="case-comment"><textarea maxLength={2000} value={comment} onChange={e => setComment(e.target.value)} placeholder="Факты и следующие шаги расследования" /><button disabled={busy || !comment.trim()} onClick={addComment}>Сохранить комментарий</button></div>
          <h3><Clock size={16}/> Журнал действий</h3>
          {detail.events_total > detail.events.length && <p className="case-caveat">Показаны последние {detail.events.length} из {detail.events_total} событий.</p>}
          <div className="case-events">{detail.events.length ? detail.events.map(event => <div key={event.id}><strong>{event.action === 'decision' ? `Статус: ${stateText[event.to_state!]}` : 'Комментарий'}</strong><small>{shortDate(event.created_at)} · локальный демо-ключ</small>{event.comment && <p>{event.comment}</p>}</div>) : <p>Решений и комментариев пока нет.</p>}</div>
          <p className="case-caveat">Общий API-ключ не устанавливает личность аналитика. Журнал предназначен только для локального демо.</p>
        </div>
      </>}
      {error && <div className="case-error"><AlertTriangle size={15}/>{error}</div>}
    </section>
  </div>;
}
