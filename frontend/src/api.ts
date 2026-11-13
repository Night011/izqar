export type RiskLevel = 'low' | 'medium' | 'high';
export type Scenario = 'normal' | 'new_recipient' | 'velocity' | 'fan_in' | 'chain';
export type Review = 'pending' | 'confirmed_fraud' | 'false_positive';

export interface Transaction {
  id: string;
  created_at: string;
  sender: string;
  recipient: string;
  amount: number;
  currency: string;
  scenario: string;
  device_changed: boolean;
  external_flag: boolean;
  risk_score: number;
  risk_level: RiskLevel;
  decision: string;
  reasons: string[];
  indicators: Record<string, unknown>;
  review_state: Review;
}
export interface Metrics {
  transactions: number; simulated_volume_kzt: number;
  high: number; medium: number; low: number; reviewed: number; disclaimer: string;
}
export interface GraphData {
  nodes: { id: string; in_degree: number; out_degree: number; flagged: boolean }[];
  edges: { source: string; target: string; count: number; total_amount: number }[];
  indicators: { high_fan_in_accounts: string[]; high_fan_out_accounts: string[]; note: string };
}
export interface ModelStatus {
  ready: boolean; mode: string; message?: string;
  features?: number; test_metrics?: Record<string, number | number[]>; threshold?: number;
}

const ROOT = (import.meta.env.VITE_API_BASE as string | undefined) ?? '';
export async function request<T>(path: string, key: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${ROOT}${path}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', 'X-API-Key': key, ...(options.headers ?? {}) },
  });
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const error = await response.json() as { detail?: string };
      message = typeof error.detail === 'string' ? error.detail : message;
    } catch { /* response was not JSON */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
export const money = (num: number) => new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 }).format(num) + ' ₸';
export const shortDate = (iso: string) => new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }).format(new Date(iso));
