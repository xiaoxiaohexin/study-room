/**
 * Retrieval diagnostics reported by the backend for the latest chat turn
 * (`X-RAG-Hits` / `X-Web-Sources` / `X-Learning-Mode`). A plain llama-server
 * sends none of these headers, so `current` stays null there.
 */

export interface RetrievalStatus {
	conversationId: string;
	ragHits: number;
	webSources: number;
	mode: string;
	retrievalMode: string;
	updatedAt: number;
}

class RetrievalStatusStore {
	current = $state<RetrievalStatus | null>(null);

	set(status: Omit<RetrievalStatus, 'updatedAt'>): void {
		this.current = { ...status, updatedAt: Date.now() };
	}
}

export const retrievalStatusStore = new RetrievalStatusStore();
