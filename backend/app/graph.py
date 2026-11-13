"""Time-aware graph representation of synthetic transfers and investigation leads."""
from datetime import datetime, timedelta, timezone
import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session
from .models import Transaction


def build_graph(db: Session, limit: int = 500, window_minutes: int = 1440) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
    rows = list(db.scalars(select(Transaction).where(Transaction.created_at >= cutoff).order_by(Transaction.created_at.desc(), Transaction.id.desc()).limit(limit)))
    graph = nx.DiGraph()
    for tx in rows:
        if graph.has_edge(tx.sender, tx.recipient):
            edge = graph[tx.sender][tx.recipient]
            edge['count'] += 1
            edge['total_amount'] += tx.amount
            edge['latest_at'] = max(edge['latest_at'], tx.created_at)
        else:
            graph.add_edge(tx.sender, tx.recipient, count=1, total_amount=tx.amount, latest_at=tx.created_at)
    nodes = []
    for node in graph.nodes:
        inbound, outbound = graph.in_degree(node), graph.out_degree(node)
        watched = inbound >= 3 or outbound >= 3
        nodes.append({'id': node, 'in_degree': inbound, 'out_degree': outbound, 'flagged': watched})
    edges = [{'source': source, 'target': target, 'count': data['count'],
              'total_amount': round(data['total_amount'], 2), 'latest_at': data['latest_at'].isoformat()} for source, target, data in graph.edges(data=True)]
    chronological = sorted(rows, key=lambda tx: (tx.created_at, tx.id))
    rapid_chains = []
    recent_inbound = {}
    for tx in chronological:
        first = recent_inbound.get(tx.sender)
        if first is not None:
            delta = (tx.created_at - first.created_at).total_seconds()
            if 0 <= delta <= 900 and len(rapid_chains) < 20:
                rapid_chains.append({'path': [first.sender, first.recipient, tx.recipient], 'minutes_between': round(delta / 60, 2), 'amounts': [round(first.amount, 2), round(tx.amount, 2)]})
        recent_inbound[tx.recipient] = tx
    indicators = {
        'high_fan_in_accounts': [n['id'] for n in nodes if n['in_degree'] >= 3],
        'high_fan_out_accounts': [n['id'] for n in nodes if n['out_degree'] >= 3],
        'rapid_transfer_chains': rapid_chains,
        'window_minutes': window_minutes,
        'note': 'Граф отражает временные и структурные признаки, но не доказывает мошенничество.'
    }
    return {'nodes': nodes, 'edges': edges, 'indicators': indicators}
