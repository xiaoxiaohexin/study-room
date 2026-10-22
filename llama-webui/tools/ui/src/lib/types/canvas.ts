/**
 * Canvas Types & Data Contracts
 * Milestone 3: Frontend Text Selection & Dual-Zone Provenance Card
 */

export type CanvasViewMode = 'canvas' | 'split' | 'list';

export type NodeCardColor = 'amber' | 'sky' | 'emerald' | 'violet' | 'rose' | 'slate';

export type CustomNodeKind = 'note' | 'rag' | 'flashcard' | 'hub' | 'image' | 'knowledge';

export interface TurnImageItem {
	id: string;
	name: string;
	url: string;
	source: 'message' | 'custom' | 'linked' | 'inherited';
}

export interface TurnDocItem {
	id: string;
	name: string;
	content: string;
	kind: string;
}

export interface TurnMemoryItem {
	id: string;
	front: string;
	back: string;
	interval: number;
	easiness: number;
	flipped?: boolean;
}

export interface TurnAssociationData {
	customImages?: Array<{ id: string; name: string; url: string }>;
	activeImageIndex?: number;
	inheritedImages?: boolean;
	imagePlacement?: 'top' | 'side' | 'hidden';
	ragItems?: Array<{ content: string; source: string; score: number }>;
	ragLoading?: boolean;
	ragExpanded?: boolean;
	memoryItems?: TurnMemoryItem[];
	memoryExpanded?: boolean;
	expandedDocId?: string | null;
}

/**
 * Provenance Anchor for bi-directional linking:
 * Chat Message ↔ Knowledge Node ↔ Textbook Page
 */
export interface ProvenanceAnchor {
	/** Unique anchor id */
	id: string;
	/** Associated assistant message id */
	messageId?: string;
	/** Conversation turn index (1-based) */
	turnIndex?: number;
	/** Course or topic namespace */
	courseId?: string;
	/** Backend RAG document id (structural anchor for real page preview) */
	docId?: string;
	/** Backend RAG chunk id */
	chunkId?: string;
	/** Reference textbook or document title */
	docTitle?: string;
	/** Physical starting page number in document */
	pageStart?: number;
	/** Physical ending page number in document */
	pageEnd?: number;
	/** Verbatim excerpt text from chat or textbook */
	quoteSnippet: string;
	/** Timestamp when anchor was extracted */
	createdAt: number;
	/** Optional metadata */
	extra?: Record<string, unknown>;
}

export type RevisionChangeType =
	| 'initial'
	| 'edit_content'
	| 'append_quote'
	| 'revert'
	| 'create'
	| 'edit';

/**
 * Knowledge revision snapshot for non-destructive version history & rollback
 */
export interface KnowledgeRevision {
	/** Incrementing version number (1, 2, 3...) */
	version: number;
	/** Revision creation timestamp (ms) */
	timestamp: number;
	/** Editor role */
	editor: 'user' | 'ai' | 'extract';
	/** Type of change */
	changeType: RevisionChangeType;
	/** Human-readable change summary */
	summary: string;
	/** Markdown content snapshot */
	content: string;
	/** Concept title snapshot */
	title?: string;
	/** Quote snippet snapshot */
	quoteSnippet?: string;
	/** Backend raw_quote alias */
	rawQuote?: string;
	/** Difficulty snapshot */
	difficulty?: number;
}

/**
 * Canvas node with dual-zone knowledge extensions
 */
export interface CustomCanvasNode {
	id: string;
	kind: CustomNodeKind;
	title: string;
	content: string;
	color: NodeCardColor;
	x: number;
	y: number;
	width: number;

	// Non-knowledge card fields
	backContent?: string;
	flipped?: boolean;
	ragQuery?: string;
	ragResults?: Array<{ content: string; source: string; score: number }>;
	ragLoading?: boolean;
	cardId?: string;
	sm2Interval?: number;
	sm2Easiness?: number;
	images?: Array<{ id: string; name: string; url: string }>;
	activeImageIndex?: number;
	linkedTurnId?: string;

	// Knowledge card extensions (Milestone 3)
	/** Backend SQLite knowledge_point ID */
	knowledgePointId?: string;
	/** Course that owns this knowledge point (backend `course_id`) */
	courseId?: string;
	/** True when a backend write failed and needs a retry */
	syncPending?: boolean;
	/** Immutable raw quote text */
	rawQuote?: string;
	/** Cognitive difficulty (0.1 ~ 1.0, default 0.5) */
	difficulty?: number;
	/** Mastery level (0.0 ~ 1.0, default 0.3) */
	mastery?: number;
	/** Provenance anchor chain */
	provenance?: ProvenanceAnchor[];
	/** Revision history snapshots */
	revisions?: KnowledgeRevision[];
	/** Active version number */
	currentVersion?: number;

	// UI state
	historyExpanded?: boolean;
	activeRevisionPreview?: number | null;
	activeProvenanceIndex?: number;
}

export type SemanticRelationType =
	| 'prerequisite'
	| 'causal'
	| 'similar'
	| 'branch_followup'
	| 'default';

export interface CanvasEdge {
	id: string;
	from: string;
	to: string;
	label?: string;
	isAuto?: boolean;
	relType?: 'prerequisite' | 'causal' | 'similar' | 'branch_followup';
}

export interface PersistedCanvasData {
	panX: number;
	panY: number;
	zoom: number;
	nodePositions: Record<string, { x: number; y: number }>;
	customNodes: CustomCanvasNode[];
	customEdges: CanvasEdge[];
	turnAssociations?: Record<string, TurnAssociationData>;
}
