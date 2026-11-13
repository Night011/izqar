"""Graph representation of synthetic transfers and structural indicators."""
import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session
from .models import Transaction


def build_graph(db: Session, limit: int = 500) -> dict:
    rows = list(db.scalars(select(Transaction).order_by(Transaction.created_at.desc(), Transaction.id.desc()).limit(limit)))
    graph = nx.DiGraph()
    for tx in rows:
        if graph.has_edge(tx.sender, tx.recipient):
            graph[tx.sender][tx.recipient]['count'] += 1
            graph[tx.sender][tx.recipient]['total_amount'] += tx.amount
        else:
            graph.add_edge(tx.sender, tx.recipient, count=1, total_amount=tx.amount)
    nodes = []
    for node in graph.nodes:
        inbound, outbound = graph.in_degree(node), graph.out_degree(node)
        watched = inbound >= 3 or outbound >= 3
        nodes.append({'id': node, 'in_degree': inbound, 'out_degree': outbound, 'flagged': watched})
    edges = [{'source': source, 'target': target, 'count': data['count'],
              'total_amount': round(data['total_amount'], 2)} for source, target, data in graph.edges(data=True)]
    indicators = {
        'high_fan_in_accounts': [n['id'] for n in nodes if n['in_degree'] >= 3],
        'high_fan_out_accounts': [n['id'] for n in nodes if n['out_degree'] >= 3],
        'note': 'Граф отражает структурные признаки, но не доказывает мошенничество.'
    }
    return {'nodes': nodes, 'edges': edges, 'indicators': indicators}
