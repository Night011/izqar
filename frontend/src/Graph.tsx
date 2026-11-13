import { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';
import type { GraphData } from './api';

export function Graph({ data }: { data: GraphData }) {
  const root = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!root.current) return;
    const nodes = data.nodes.map(n => ({ data: { id: n.id, label: n.id, flagged: n.flagged } }));
    const edges = data.edges.map((e, i) => ({ data: { id: `edge-${i}`, source: e.source, target: e.target, weight: e.count } }));
    const cy = cytoscape({
      container: root.current,
      elements: [...nodes, ...edges],
      style: [
        { selector: 'node', style: { 'background-color': '#2f8c96', 'border-width': 1, 'border-color': '#5cc5c7', 'label': 'data(label)', 'color': '#dcebf2', 'font-size': 9, 'text-margin-y': -13, 'text-outline-color': '#0c1722', 'text-outline-width': 1, 'width': 22, 'height': 22 } },
        { selector: 'node[?flagged]', style: { 'background-color': '#e49d57', 'border-color': '#ffc895', 'width': 32, 'height': 32 } },
        { selector: 'edge', style: { 'line-color': '#516b83', 'width': 1.5, 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#516b83', 'arrow-scale': .8 } },
      ],
      layout: { name: 'cose', animate: false, fit: true, padding: 35 },
      minZoom: .2,
      maxZoom: 2.5,
    });
    return () => cy.destroy();
  }, [data]);
  return <div className="graph" role="img" aria-label="Интерактивный граф счетов и направлений переводов" ref={root} />;
}
