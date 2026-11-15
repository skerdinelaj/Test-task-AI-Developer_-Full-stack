// Mirrors eu261_claim_agent.schemas.ClaimResult (the API contract).
export type ClaimStatus =
  | 'eligible' | 'not_eligible' | 'needs_info' | 'needs_review' | 'out_of_scope' | 'rejected'
export type ReviewState = 'not_required' | 'pending' | 'approved' | 'rejected'

export interface RiskFlag { code: string; severity: 'info' | 'warning' | 'high'; message: string }
export interface Decision {
  status: ClaimStatus; eligible: boolean | null; amount_eur: number; distance_km: number | null
  arrival_delay_minutes: number | null; reasons: string[]; rule_ids: string[]
}
export interface ToolCallRecord {
  tool: string; arguments: Record<string, unknown>; ok: boolean; duration_ms: number
  error: string | null; source: 'planner' | 'backfill'
}
export interface ClaimResult {
  run_id: string; status: ClaimStatus; review_state: ReviewState; needs_human_review: boolean
  summary: string; decision: Decision | null; missing_fields: string[]; risk_flags: RiskFlag[]
  letter: { subject: string; body: string } | null
  review: { approved: boolean; reviewer: string; note: string | null } | null
  tool_calls: ToolCallRecord[]; steps: { node: string; duration_ms: number }[]
  usage: { input_tokens: number; output_tokens: number; llm_calls: number; cost_usd: number }
  provider: string
}
