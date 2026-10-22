import { afterEach, describe, expect, it, vi } from 'vitest';
import { attributeQuote } from '$lib/services/rag-attribution.service';

describe('attributeQuote', () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('returns the matched chunk from the backend', async () => {
		const fetchMock = vi.fn().mockResolvedValue({
			ok: true,
			json: async () => ({
				matched: true,
				overlap: 0.81,
				chunk: {
					doc_id: 'doc_1',
					chunk_id: 'doc_1_2',
					chunk_index: 2,
					filename: '物理.pdf',
					page_start: 12,
					page_end: 12,
					page_label: 'P.12',
					headers: ['第一章'],
					content: '原文内容'
				}
			})
		});
		vi.stubGlobal('fetch', fetchMock);

		const result = await attributeQuote('牛顿第二定律的内容', 'physics');
		expect(result?.matched).toBe(true);
		expect(result?.chunk?.doc_id).toBe('doc_1');
		expect(result?.chunk?.page_start).toBe(12);

		const [url, init] = fetchMock.mock.calls[0];
		expect(url).toBe('/api/rag/attribute');
		const body = JSON.parse(init.body);
		expect(body.namespace).toBe('physics');
		expect(body.quote).toContain('牛顿');
		expect(init.headers['X-RAG-Namespace']).toBe('physics');
	});

	it('returns null when the backend reports no confident match', async () => {
		vi.stubGlobal(
			'fetch',
			vi.fn().mockResolvedValue({
				ok: true,
				json: async () => ({ matched: false, overlap: 0.1, reason: 'no_confident_match' })
			})
		);
		const result = await attributeQuote('完全无关的句子', 'default');
		expect(result?.matched).toBe(false);
	});

	it('returns null for network errors or non-ok responses', async () => {
		vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
		expect(await attributeQuote('一段选中的文本', 'default')).toBeNull();

		vi.unstubAllGlobals();
		vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 500 }));
		expect(await attributeQuote('一段选中的文本', 'default')).toBeNull();
	});

	it('skips attribution for very short quotes', async () => {
		const fetchMock = vi.fn();
		vi.stubGlobal('fetch', fetchMock);
		expect(await attributeQuote('a', 'default')).toBeNull();
		expect(fetchMock).not.toHaveBeenCalled();
	});
});
