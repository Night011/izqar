export type RiskLevel = 'low' | 'medium' | 'high';
export type Scenario = 'normal' | 'new_recipient' | 'velocity' | 'fan_in' | 'chain' | 'cycle';
export type Review = 'pending' | 'in_review' | 'step_up' | 'escalated' | 'allowed' | 'confirmed_fraud' | 'false_positive';

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
  edges: { source: string; target: string; count: number; total_amount: number; latest_at: string }[];
  indicators: { high_fan_in_accounts: string[]; high_fan_out_accounts: string[]; rapid_transfer_chains: unknown[]; directed_three_node_cycles: string[][]; window_minutes: number; note: string };
}
export interface ModelStatus {
  ready: boolean; mode: string; message?: string;
  ieee_cis?: ModelDetail | null;
  paysim?: ModelDetail | null;
  baf_base?: ModelDetail | null;
  ibm_aml?: ModelDetail | null;
  unified_mtl?: UnifiedDetail | null;
  artifact_errors?: { ieee_cis?: string | null; paysim?: string | null; baf_base?: string | null; ibm_aml?: string | null; unified_mtl?: string | null };
}
export interface UnifiedDetail {
  model_version: string; experimental: true; score_kind: string;
  tasks: Record<string, { threshold: number; validation_metrics: TestMetrics }>;
  task_schemas: Record<string, string[]>;
}
export interface TestMetrics {
  precision: number; recall: number; f1: number; average_precision_pr_auc: number; roc_auc: number;
  confusion_matrix_tn_fp_fn_tp: [number, number, number, number];
  false_positive_rate: number; false_negative_rate: number;
  false_alerts_per_10000_operations: number; cases: number; fraud_cases?: number; positive_cases?: number;
}
export interface ModelDetail {
  features: number; feature_names: string[]; test_metrics: TestMetrics | null; threshold: number;
  model_version: string; dataset: string; trained_at_utc?: string;
  split_counts: Record<string, number>; score_kind: string;
  feature_importance: { feature: string; importance: number }[];
}
export interface ModelPrediction {
  risk_score: number; threshold: number; flagged: boolean; source: string;
  model_version: string; score_kind: string; warning: string;
}
export interface RegistryEntry {
  manifest: string; active_for: string | null; state: 'installed' | 'report_only' | 'invalid'; error?: string | null;
  model_kind?: string; model_version?: string; dataset?: string; trained_at_utc?: string;
  manifest_sha256?: string; feature_count?: number;
  data_files?: { filename: string; size_bytes?: number; sha256?: string }[];
  components?: { filename: string; bytes: number; sha256: string }[];
  validation_metrics?: TestMetrics | null; test_metrics?: TestMetrics | null;
}
export interface ModelRegistry { registry_version: number; entries: RegistryEntry[]; note: string }
export interface InvestigationEvent {
  id: string; transaction_id: string; created_at: string; action: 'decision' | 'comment';
  from_state: Review | null; to_state: Review | null; comment: string | null;
  actor_source: string;
}
export interface InvestigationList { total: number; items: Transaction[] }
export interface InvestigationDetail {
  transaction: Transaction; events: InvestigationEvent[]; events_total: number;
  related_transactions: Transaction[];
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
