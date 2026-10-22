/**
 * Pure helpers that translate backend knowledge points/relations into canvas
 * structures. Kept outside the Svelte component so unit tests exercise the real
 * implementation instead of a copied "shadow" version.
 */

import type {
	CanvasEdge,
	CustomCanvasNode,
	KnowledgeRevision,
	ProvenanceAnchor
} from '$lib/types/canvas';

export interface BackendKnowledgePoint {
	id: string;
	course_id?: string;
	user_id?: string;
	name: string;
	summary?: string;
	difficulty?: number;
	raw_quote?: string;
	content?: string;
	version?: number;
	provenance?: Record<string, unknown>;
	history?: Array<Record<string, unknown>>;
	created_at?: number;
	updated_at?: number;
}

export interface BackendRelation {
	id?: string;
	src_id: string;
	rel_type: 'prerequisite' | 'causal' | 'similar' | 'branch_followup';
	dst_id: string;
	metadata?: Record<string, unknown>;
}

const CHANGE_TYPE_MAP: Record<string, KnowledgeRevision['changeType']> = {
	create: 'create',
	initial: 'initial',
	edit: 'edit',
	edit_content: 'edit_content',
	append: 'append_quote',
	append_quote: 'append_quote',
	rollback: 'revert',
	revert: 'revert'
};

function toMillis(value: unknown): number {
	const num = Number(value);
	if (!Number.isFinite(num) || num <= 0) return Date.now();
	// Backend timestamps are seconds; local ones are milliseconds.
	return num < 1e12 ? Math.round(num * 1000) : Math.round(num);
}

function normalizeChangeType(raw: unknown): KnowledgeRevision['changeType'] {
	const key = String(raw ?? 'edit').toLowerCase();
	if (key.startsWith('rollback')) return 'revert';
	return CHANGE_TYPE_MAP[key] ?? 'edit';
}

/** Map backend `history_json` snapshots to local revision records. */
export function mapBackendHistoryToRevisions(
	history: unknown,
	fallbackTitle = ''
): KnowledgeRevision[] {
	if (!Array.isArray(history) || history.length === 0) return [];
	return history
		.filter((entry): entry is Record<string, unknown> => !!entry && typeof entry === 'object')
		.map((entry) => ({
			version: Number(entry.version) || 1,
			timestamp: toMillis(entry.timestamp),
			editor: 'ai' as const,
			changeType: normalizeChangeType(entry.change_type ?? entry.changeType),
			summary: String(entry.summary ?? entry.change_type ?? ''),
			content: String(entry.content ?? ''),
			title: String(entry.title ?? fallbackTitle ?? ''),
			quoteSnippet: String(entry.raw_quote ?? ''),
			rawQuote: String(entry.raw_quote ?? ''),
			difficulty: typeof entry.difficulty === 'number' ? entry.difficulty : undefined
		}))
		.sort((a, b) => a.version - b.version);
}

function anchorFromDict(
	dict: Record<string, unknown>,
	index: number
): ProvenanceAnchor | null {
	const quote = String(dict.quote_snippet ?? dict.quoteSnippet ?? '').trim();
	const messageId = (dict.message_id ?? dict.messageId ?? undefined) as string | undefined;
	if (!quote && !messageId) return null;
	return {
		id: String(dict.id ?? `prov-backend-${index}`),
		messageId,
		turnIndex:
			typeof dict.turn_index === 'number'
				? Number(dict.turn_index)
				: typeof (dict as { turnIndex?: unknown }).turnIndex === 'number'
					? Number((dict as { turnIndex?: number }).turnIndex)
					: undefined,
		docId: (dict.doc_id ?? dict.docId ?? undefined) as string | undefined,
		chunkId: (dict.chunk_id ?? dict.chunkId ?? undefined) as string | undefined,
		docTitle: (dict.doc_title ?? dict.docTitle ?? undefined) as string | undefined,
		pageStart:
			typeof dict.page_start === 'number'
				? Number(dict.page_start)
				: typeof (dict as { pageStart?: unknown }).pageStart === 'number'
					? Number((dict as { pageStart?: number }).pageStart)
					: undefined,
		pageEnd:
			typeof dict.page_end === 'number'
				? Number(dict.page_end)
				: typeof (dict as { pageEnd?: unknown }).pageEnd === 'number'
					? Number((dict as { pageEnd?: number }).pageEnd)
					: undefined,
		quoteSnippet: quote,
		createdAt: Date.now()
	};
}

/** Map persisted backend provenance (primary anchor + extra.all_anchors) to anchors. */
export function mapBackendProvenanceToAnchors(provenance: unknown): ProvenanceAnchor[] {
	if (!provenance || typeof provenance !== 'object') return [];
	const dict = provenance as Record<string, unknown>;
	const anchors: ProvenanceAnchor[] = [];
	const primary = anchorFromDict(dict, 0);
	if (primary) anchors.push(primary);

	const extra = dict.extra;
	if (extra && typeof extra === 'object') {
		const allAnchors = (extra as Record<string, unknown>).all_anchors;
		if (Array.isArray(allAnchors)) {
			allAnchors.forEach((item, index) => {
				if (item && typeof item === 'object') {
					const anchor = anchorFromDict(item as Record<string, unknown>, index + 1);
					if (anchor) anchors.push(anchor);
				}
			});
		}
	}
	return anchors;
}

/** Build a canvas knowledge card from a backend knowledge point. */
export function backendKpToCanvasNode(
	kp: BackendKnowledgePoint,
	index: number,
	fallbackCourseId = 'default'
): CustomCanvasNode {
	const provenance = mapBackendProvenanceToAnchors(kp.provenance);
	const revisions = mapBackendHistoryToRevisions(kp.history, kp.name);
	const col = index % 3;
	const row = Math.floor(index / 3);
	return {
		id: `kn-${kp.id}`,
		kind: 'knowledge',
		knowledgePointId: kp.id,
		courseId: kp.course_id || fallbackCourseId,
		title: kp.name || '知识点',
		content: kp.content || kp.summary || '',
		rawQuote: kp.raw_quote || provenance[0]?.quoteSnippet || '',
		color: 'emerald',
		width: 380,
		x: 640 + col * 430,
		y: 80 + row * 420,
		difficulty: typeof kp.difficulty === 'number' ? kp.difficulty : 0.5,
		mastery: 0.3,
		currentVersion: Number(kp.version) || revisions[revisions.length - 1]?.version || 1,
		provenance,
		revisions,
		historyExpanded: false,
		activeProvenanceIndex: 0,
		syncPending: false
	};
}

export interface MergeResult {
	nodes: CustomCanvasNode[];
	added: number;
	updated: number;
}

/**
 * Merge backend knowledge points into canvas nodes without touching layout of
 * existing cards. Missing cards are added; existing cards adopt newer backend
 * content/version/history (backend is the single source of truth for versions).
 */
export function mergeBackendKnowledgePoints(
	currentNodes: CustomCanvasNode[],
	backendKps: BackendKnowledgePoint[],
	courseId = 'default'
): MergeResult {
	const next = currentNodes.map((node) => ({ ...node }));
	const byKpId = new Map<string, CustomCanvasNode>();
	for (const node of next) {
		if (node.knowledgePointId) byKpId.set(node.knowledgePointId, node);
	}

	let added = 0;
	let updated = 0;

	for (const kp of backendKps) {
		const existing = byKpId.get(kp.id);
		if (!existing) {
			next.push(backendKpToCanvasNode(kp, next.filter((n) => n.kind === 'knowledge').length, courseId));
			added += 1;
			continue;
		}
		const backendVersion = Number(kp.version) || 1;
		const localVersion = existing.currentVersion || 1;
		if (backendVersion > localVersion) {
			existing.title = kp.name || existing.title;
			existing.content = kp.content || kp.summary || existing.content;
			existing.rawQuote = kp.raw_quote || existing.rawQuote;
			existing.currentVersion = backendVersion;
			existing.courseId = kp.course_id || existing.courseId;
			existing.revisions = mapBackendHistoryToRevisions(kp.history, kp.name);
			const anchors = mapBackendProvenanceToAnchors(kp.provenance);
			if (anchors.length > 0) existing.provenance = anchors;
			existing.syncPending = false;
			updated += 1;
		}
	}

	return { nodes: next, added, updated };
}

/** Map backend relations to canvas edges when both endpoints have cards. */
export function mapBackendRelationsToEdges(
	relations: BackendRelation[],
	nodes: CustomCanvasNode[]
): CanvasEdge[] {
	const nodeIdByKpId = new Map<string, string>();
	for (const node of nodes) {
		if (node.knowledgePointId) nodeIdByKpId.set(node.knowledgePointId, node.id);
	}

	const edges: CanvasEdge[] = [];
	for (const rel of relations) {
		const from = nodeIdByKpId.get(rel.src_id);
		const to = nodeIdByKpId.get(rel.dst_id);
		if (!from || !to) continue;
		edges.push({
			id: `edge-backend-${rel.id ?? `${rel.src_id}-${rel.rel_type}-${rel.dst_id}`}`,
			from,
			to,
			relType: rel.rel_type,
			isAuto: true,
			label:
				rel.rel_type === 'prerequisite'
					? '前置知识'
					: rel.rel_type === 'causal'
						? '因果关系'
						: rel.rel_type === 'similar'
							? '相似概念'
							: '分支追问'
		});
	}
	return edges;
}
