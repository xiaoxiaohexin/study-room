/**
 * Web / 临时知识库（Staging）API client.
 *
 * 临时库不参与标准 RAG：先暂存，再提纯成候选，最后由用户确认晋升到标准知识库。
 */

import { learningHeaders, learningJsonHeaders } from './learning-context';

export interface WebStatus {
	enabled: boolean;
	search_available: boolean;
	search_providers: string[];
	search_primary: string;
	search_fallback: string;
	search_base_url: string;
	mcp_configured: boolean;
	fallback_mode: string;
	allowed_domains: string[];
	max_bytes: number;
	staging_ttl_days: number;
}

export interface WebPreviewResult {
	status: string;
	url: string;
	final_url: string;
	title: string;
	chars: number;
	preview: string;
	content_hash: string;
	fetched_at: number;
	content_type: string;
}

export interface StagingDocument {
	doc_id: string;
	filename: string;
	title: string;
	namespace: string;
	tier: string;
	source_type: string;
	url: string;
	origin: string;
	session_id: string;
	chunks: number;
	page_count: number;
	quality_score: number;
	status: string;
	created_at: number;
	fetched_at: number;
	target_namespace: string;
	file_available: boolean;
	promoted_doc_id: string;
	reused?: boolean;
	excerpt?: string;
}

export interface StagingCandidate {
	id: string;
	doc_id: string;
	title: string;
	summary: string;
	content: string;
	raw_quote: string;
	difficulty: number;
	quality_score: number;
	status: string;
	suggested_course_id: string;
	promoted_kp_id: string;
	created_at: number;
	source_title: string;
	source_url: string;
}

export interface PromotionResult {
	ok: boolean;
	target_course_id: string;
	target_namespace: string;
	promoted_documents: Array<Record<string, unknown>>;
	promoted_knowledge_points: Array<Record<string, unknown>>;
	errors: string[];
}

async function requestJson<T>(
	url: string,
	init: RequestInit & { json?: unknown } = {}
): Promise<T> {
	const headers = init.json !== undefined ? learningJsonHeaders() : learningHeaders();
	const response = await fetch(url, {
		...init,
		headers: { ...headers, ...(init.headers as Record<string, string> | undefined) },
		body: init.json !== undefined ? JSON.stringify(init.json) : init.body
	});
	const data = (await response.json().catch(() => ({}))) as T & { detail?: string };
	if (!response.ok) {
		throw new Error(data?.detail || `请求失败（HTTP ${response.status}）`);
	}
	return data;
}

export function getWebStatus(): Promise<WebStatus> {
	return requestJson<WebStatus>('/api/web/status');
}

export function previewUrl(url: string): Promise<WebPreviewResult> {
	return requestJson<WebPreviewResult>('/api/web/preview', {
		method: 'POST',
		json: { url }
	});
}

export function stageUrl(
	url: string,
	options: { sessionId?: string; conversationId?: string; query?: string; targetCourseId?: string } = {}
): Promise<{ status: string; document: StagingDocument }> {
	return requestJson('/api/web/stage', {
		method: 'POST',
		json: {
			url,
			session_id: options.sessionId ?? '',
			conversation_id: options.conversationId ?? '',
			query: options.query ?? '',
			target_course_id: options.targetCourseId ?? ''
		}
	});
}

export function searchWeb(
	query: string,
	options: { topK?: number; stage?: boolean; sessionId?: string } = {}
): Promise<{
	status: string;
	results: Array<{ title: string; url: string; snippet: string; provider: string }>;
	staged: StagingDocument[];
}> {
	return requestJson('/api/web/search', {
		method: 'POST',
		json: {
			query,
			top_k: options.topK ?? 5,
			stage: options.stage ?? false,
			session_id: options.sessionId ?? ''
		}
	});
}

export function listStagingDocuments(
	options: { sessionId?: string; status?: string; limit?: number } = {}
): Promise<{ status: string; count: number; documents: StagingDocument[] }> {
	const params = new URLSearchParams();
	if (options.sessionId) params.set('session_id', options.sessionId);
	if (options.status) params.set('status', options.status);
	if (options.limit) params.set('limit', String(options.limit));
	return requestJson(`/api/staging/documents?${params.toString()}`);
}

export function deleteStagingDocument(docId: string): Promise<{ status: string; deleted: boolean }> {
	return requestJson(`/api/staging/documents/${encodeURIComponent(docId)}`, {
		method: 'DELETE'
	});
}

export function distillStagingDocument(
	docId: string
): Promise<{ status: string; doc_id: string; quality_score: number; candidates: StagingCandidate[] }> {
	return requestJson(`/api/staging/documents/${encodeURIComponent(docId)}/distill`, {
		method: 'POST'
	});
}

export function listStagingCandidates(
	options: { sessionId?: string; status?: string; limit?: number } = {}
): Promise<{ status: string; count: number; candidates: StagingCandidate[] }> {
	const params = new URLSearchParams();
	if (options.sessionId) params.set('session_id', options.sessionId);
	if (options.status !== undefined) params.set('status', options.status);
	if (options.limit) params.set('limit', String(options.limit));
	return requestJson(`/api/staging/candidates?${params.toString()}`);
}

export function rejectStagingCandidate(candidateId: string): Promise<{ status: string }> {
	return requestJson(`/api/staging/candidates/${encodeURIComponent(candidateId)}/reject`, {
		method: 'POST'
	});
}

export function promoteStaging(options: {
	stagingDocIds: string[];
	candidateIds: string[];
	targetCourseId: string;
	targetNamespace?: string;
}): Promise<PromotionResult> {
	return requestJson<PromotionResult>('/api/staging/promote', {
		method: 'POST',
		json: {
			staging_doc_ids: options.stagingDocIds,
			candidate_ids: options.candidateIds,
			target_course_id: options.targetCourseId,
			target_namespace: options.targetNamespace ?? null
		}
	});
}

export function revokePromotion(stagingDocId: string): Promise<{
	ok: boolean;
	removed_doc_id: string;
	revoked_knowledge_points: string[];
}> {
	return requestJson('/api/staging/revoke', {
		method: 'POST',
		json: { staging_doc_id: stagingDocId }
	});
}

export function stagingStats(
	options: { sessionId?: string } = {}
): Promise<{
	status: string;
	documents: number;
	pending_candidates: number;
	search_available: boolean;
	ttl_days: number;
}> {
	const params = new URLSearchParams();
	if (options.sessionId) params.set('session_id', options.sessionId);
	return requestJson(`/api/staging/stats?${params.toString()}`);
}
