import type { CustomCanvasNode, ProvenanceAnchor } from '$lib/types/canvas';

export interface TextbookNavigationTarget {
	/** Backend RAG document id; required for real page preview */
	docId?: string;
	/** Knowledge base namespace the document belongs to */
	namespace?: string;
	docTitle?: string;
	pageStart: number;
	pageEnd?: number;
	quoteSnippet?: string;
}

/**
 * Extracts a concise title candidate from selected text.
 * Prioritizes bold markdown, definition statements, and short clauses.
 */
export function extractTitleCandidate(text: string): string {
	if (!text || !text.trim()) return '知识节点提炼';

	let raw = text.trim();

	// If starts with or contains code block, strip code blocks first
	if (raw.includes('```')) {
		const withoutCode = raw.replace(/```[\s\S]*?```/g, '').trim();
		if (withoutCode) {
			raw = withoutCode;
		} else {
			raw = raw
				.replace(/```[a-zA-Z0-9_-]*\n?/g, '')
				.replace(/```/g, '')
				.trim();
		}
	}

	// Rule 1: Markdown bold/italic wraps, e.g. ***过拟合***, **Transformer 架构**, *Dropout*, **\lim_{x \to 0}**
	const bold3Match = raw.match(/^\*\*\*([^*\n]+)\*\*\*/);
	if (bold3Match && bold3Match[1].trim()) {
		return bold3Match[1].trim();
	}
	const bold2Match = raw.match(/^\*\*([^*\n]+)\*\*/);
	if (bold2Match && bold2Match[1].trim()) {
		return bold2Match[1].trim();
	}
	const italicMatch = raw.match(/^\*([^*\n]+)\*/);
	if (italicMatch && italicMatch[1].trim()) {
		return italicMatch[1].trim();
	}

	// Clean leading symbols/bullets/numbers (e.g. - , * , > , 1. , 2、 , --- , ===)
	let cleaned = raw
		.replace(/^[\s#*>\-–—=~`]+/, '')
		.replace(/^[0-9]+[.、]\s*/, '')
		.replace(/^[\s#*>\-–—=~`]+/, '')
		.replace(/[\r\n]+/g, ' ')
		.replace(/\s+/g, ' ')
		.trim();

	// Check if only non-word / punctuation / symbols remain
	const hasMeaningfulContent = /[\p{L}\p{N}]/u.test(cleaned);
	if (!hasMeaningfulContent) {
		return '知识节点提炼';
	}

	// Rule 2: Definition statement, e.g. "深度残差学习 是为了...", "LayerNorm 是在...", "反向传播 是基于...", "残差连接是指..."
	const defMatch = cleaned.match(
		/^([^\s，。！？,.!?;:：]{2,25})\s*(?:是指|是由于|是为|是为了|是一项|是一种|是在|是基于|是)/
	);
	if (defMatch && defMatch[1].trim()) {
		return defMatch[1].trim();
	}

	// Rule 3: Short clause / punctuation boundary
	cleaned = cleaned.replace(/[，。！？,.!?;:：]+$/, '').trim();

	const boundaryMatch = cleaned.match(/^([^，。！？,.!?;:：\n]+)/);
	if (boundaryMatch && boundaryMatch[1]) {
		cleaned = boundaryMatch[1].trim();
	}

	if (cleaned.length > 40) {
		cleaned = cleaned.slice(0, 40);
	}
	return cleaned || '知识节点提炼';
}

/**
 * Extracts provenance info (chat turn index, document title, page numbers)
 * from message metadata, DOM attributes, or citations.
 */
export function extractProvenanceInfo(
	selectionText: string,
	messageContainer?: HTMLElement | null
): {
	turnIndex?: number;
	docTitle?: string;
	pageStart?: number;
	pageEnd?: number;
} {
	let turnIndex: number | undefined;
	let docTitle: string | undefined;
	let pageStart: number | undefined;
	let pageEnd: number | undefined;

	if (messageContainer) {
		const turnAttr = messageContainer.getAttribute('data-turn-index');
		if (turnAttr) {
			const parsed = parseInt(turnAttr, 10);
			if (!isNaN(parsed)) turnIndex = parsed;
		}

		// Look for citation badges / data attributes in the message element
		const docAttr = messageContainer.getAttribute('data-doc-title');
		if (docAttr) docTitle = docAttr;

		const pageStartAttr = messageContainer.getAttribute('data-page-start');
		if (pageStartAttr) {
			const parsed = parseInt(pageStartAttr, 10);
			if (!isNaN(parsed)) pageStart = parsed;
		}

		const pageEndAttr = messageContainer.getAttribute('data-page-end');
		if (pageEndAttr) {
			const parsed = parseInt(pageEndAttr, 10);
			if (!isNaN(parsed)) pageEnd = parsed;
		}
	}

	// 1. Doc title regex: 《...》 or filename with extension like LinearAlgebra.pdf
	if (!docTitle) {
		const bookMatch = selectionText.match(/《([^》]+)》/);
		if (bookMatch && bookMatch[1]) {
			docTitle = bookMatch[1];
		} else {
			const pdfMatch = selectionText.match(/([a-zA-Z0-9_\u4e00-\u9fa5\-.]+\.(?:pdf|docx?|pptx?|epub|txt))/i);
			if (pdfMatch && pdfMatch[1]) {
				docTitle = pdfMatch[1];
			}
		}
	}

	// 2. Page numbers regex
	// E.g. "第 P.42-43 页", "第 P.108 页", "第 77 到 79 页", "P.42-43", "pages 10-12"
	const pageRangeMatch = selectionText.match(/(?:P\.|第|\bpages?\s*)\s*([0-9]+)\s*(?:[-–—~到至]\s*([0-9]+))?\s*(?:页)?/i);
	if (pageRangeMatch) {
		if (pageStart === undefined && pageRangeMatch[1]) {
			pageStart = parseInt(pageRangeMatch[1], 10);
		}
		if (pageEnd === undefined) {
			if (pageRangeMatch[2]) {
				pageEnd = parseInt(pageRangeMatch[2], 10);
			} else if (pageStart !== undefined) {
				pageEnd = pageStart;
			}
		}
	}

	// Fallback when no match found
	if (!docTitle) {
		docTitle = '知识源文档';
	}
	if (pageStart === undefined) {
		pageStart = 1;
	}

	return { turnIndex, docTitle, pageStart, pageEnd };
}

/**
 * Sanitizes markdown content for preview while preserving safe LaTeX math blocks.
 */
export function sanitizeHtml(raw: string): string {
	if (!raw) return '';
	return raw
		.replace(/&/g, '&amp;')
		.replace(/</g, '&lt;')
		.replace(/>/g, '&gt;')
		.replace(/"/g, '&quot;')
		.replace(/'/g, '&#39;');
}

class ProvenanceNavigationService {
	// Svelte 5 runes for reactive state
	highlightedMessageId = $state<string | null>(null);
	spotlightNodeId = $state<string | null>(null);
	textbookViewerTarget = $state<TextbookNavigationTarget | null>(null);
	textbookViewerOpen = $state<boolean>(false);
	knowledgeNodes = $state<CustomCanvasNode[]>([]);

	setKnowledgeNodes(nodes: CustomCanvasNode[]) {
		this.knowledgeNodes = nodes.filter((n) => n.kind === 'knowledge');
	}

	getKnowledgeNodesForMessage(messageId: string): CustomCanvasNode[] {
		if (!messageId) return [];
		return this.knowledgeNodes.filter((node) =>
			node.provenance?.some((p: ProvenanceAnchor) => p.messageId === messageId)
		);
	}

	/**
	 * Synchronously sets the highlighted message ID and dispatches navigation.
	 */
	navigateToChatMessage(messageId: string): void {
		if (!messageId) return;
		this.highlightedMessageId = messageId;
		void this.scrollToChatMessage(messageId);
	}

	/**
	 * Navigates to a specific chat message, ensuring split view if needed,
	 * scrolling the message into view, and playing a pulse highlight animation.
	 */
	async scrollToChatMessage(messageId: string): Promise<boolean> {
		if (!messageId) return false;
		this.highlightedMessageId = messageId;

		if (typeof document !== 'undefined') {
			document.dispatchEvent(
				new CustomEvent('provenance-navigate-to-chat', {
					detail: { messageId }
				})
			);
		}

		const success = await this.scrollAndHighlightMessage(messageId);

		setTimeout(() => {
			if (this.highlightedMessageId === messageId) {
				this.highlightedMessageId = null;
			}
		}, 2500);

		return success;
	}

	private async scrollAndHighlightMessage(messageId: string): Promise<boolean> {
		if (typeof document === 'undefined') return false;

		// First try immediate lookup
		let el = document.querySelector(`[data-message-id="${messageId}"]`) as HTMLElement | null;

		// If virtualized, wait a tick for split view / layout hydration
		if (!el) {
			await new Promise((resolve) => setTimeout(resolve, 80));
			el = document.querySelector(`[data-message-id="${messageId}"]`) as HTMLElement | null;
		}

		if (!el) {
			return false;
		}

		el.scrollIntoView({ behavior: 'smooth', block: 'center' });
		el.classList.add('pulse-provenance-highlight');
		setTimeout(() => {
			el?.classList.remove('pulse-provenance-highlight');
		}, 2400);

		return true;
	}

	/**
	 * Focuses and spotlights a canvas node, transitioning to split/canvas mode if needed.
	 */
	focusCanvasNode(nodeId: string): void {
		if (!nodeId) return;
		this.spotlightNodeId = nodeId;

		if (typeof document !== 'undefined') {
			document.dispatchEvent(
				new CustomEvent('provenance-focus-canvas-node', {
					detail: { nodeId }
				})
			);
		}

		setTimeout(() => {
			if (this.spotlightNodeId === nodeId) {
				this.spotlightNodeId = null;
			}
		}, 3000);
	}

	/**
	 * Opens the textbook physical page viewer modal.
	 */
	openTextbookPreview(target: TextbookNavigationTarget): void {
		this.textbookViewerTarget = target;
		this.textbookViewerOpen = true;
	}

	/**
	 * Closes the textbook viewer modal.
	 */
	closeTextbookPreview(): void {
		this.textbookViewerOpen = false;
		this.textbookViewerTarget = null;
	}
}

export const provenanceNavigation = new ProvenanceNavigationService();
