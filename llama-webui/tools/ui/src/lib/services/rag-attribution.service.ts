/**
 * Quote attribution: ask the backend which knowledge-base chunk (and physical
 * page) a selected quote actually came from. The backend only reports a match
 * when character-trigram overlap reaches its threshold, so the UI never has to
 * fabricate a "reference textbook P.1" anchor.
 */

import { learningHeaders } from './learning-context';

export interface AttributionChunk {
	doc_id: string;
	chunk_id: string;
	chunk_index: number;
	filename: string;
	page_start: number;
	page_end: number;
	page_label: string;
	headers: string[];
	content: string;
}

export interface AttributionResult {
	matched: boolean;
	overlap?: number;
	threshold?: number;
	chunk?: AttributionChunk;
}

/** Returns null when the backend is offline or attribution failed. */
export async function attributeQuote(
	quote: string,
	namespace: string,
	options?: { timeoutMs?: number; minOverlap?: number }
): Promise<AttributionResult | null> {
	const text = (quote || '').trim();
	if (text.length < 2) return null;

	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), options?.timeoutMs ?? 6000);
	try {
		const payload: Record<string, unknown> = { quote: text, namespace };
		if (options?.minOverlap !== undefined) payload.min_overlap = options.minOverlap;

		const response = await fetch('/api/rag/attribute', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				...learningHeaders(),
				// Explicit header wins over the session binding on the backend; keep the
				// attribution scoped to the namespace the caller asked for.
				'X-RAG-Namespace': namespace
			},
			body: JSON.stringify(payload),
			signal: controller.signal
		});
		if (!response.ok) return null;
		return (await response.json()) as AttributionResult;
	} catch {
		return null;
	} finally {
		clearTimeout(timer);
	}
}
