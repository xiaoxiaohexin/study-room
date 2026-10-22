import { afterEach, describe, expect, it, vi } from 'vitest';
import {
	deleteStagingDocument,
	distillStagingDocument,
	getWebStatus,
	listStagingCandidates,
	listStagingDocuments,
	promoteStaging,
	revokePromotion,
	searchWeb,
	stageUrl
} from '$lib/services/web-staging.service';

function mockJson(payload: unknown, ok = true, status = 200) {
	const fetchMock = vi.fn().mockResolvedValue({
		ok,
		status,
		json: async () => payload
	});
	vi.stubGlobal('fetch', fetchMock);
	return fetchMock;
}

describe('web-staging service', () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it('loads web status with learning headers', async () => {
		const fetchMock = mockJson({ enabled: true, search_available: false });
		const status = await getWebStatus();
		expect(status.enabled).toBe(true);
		const [url, init] = fetchMock.mock.calls[0];
		expect(url).toBe('/api/web/status');
		expect(init?.headers?.['X-User-Id']).toBeTruthy();
	});

	it('stages a URL with session and course scope', async () => {
		const fetchMock = mockJson({ status: 'ok', document: { doc_id: 'doc_1' } });
		const result = await stageUrl('https://example.com/a', {
			sessionId: 's1',
			targetCourseId: 'course_1'
		});
		expect(result.document.doc_id).toBe('doc_1');
		const [url, init] = fetchMock.mock.calls[0];
		expect(url).toBe('/api/web/stage');
		const body = JSON.parse(init.body);
		expect(body.session_id).toBe('s1');
		expect(body.target_course_id).toBe('course_1');
	});

	it('encodes filters when listing staging documents', async () => {
		const fetchMock = mockJson({ status: 'ok', count: 0, documents: [] });
		await listStagingDocuments({ sessionId: 's 1', status: 'staged', limit: 20 });
		const [url] = fetchMock.mock.calls[0];
		expect(url).toContain('session_id=s+1');
		expect(url).toContain('status=staged');
		expect(url).toContain('limit=20');
	});

	it('distills, lists candidates, deletes and revokes with encoded ids', async () => {
		let fetchMock = mockJson({ status: 'ok', candidates: [] });
		await distillStagingDocument('doc/1');
		expect(fetchMock.mock.calls[0][0]).toBe('/api/staging/documents/doc%2F1/distill');

		fetchMock = mockJson({ status: 'ok', candidates: [] });
		await listStagingCandidates({ status: 'pending' });
		expect(fetchMock.mock.calls[0][0]).toContain('/api/staging/candidates?');
		expect(fetchMock.mock.calls[0][0]).toContain('status=pending');

		fetchMock = mockJson({ status: 'ok', deleted: true });
		await deleteStagingDocument('doc_1');
		expect(fetchMock.mock.calls[0][0]).toBe('/api/staging/documents/doc_1');
		expect(fetchMock.mock.calls[0][1].method).toBe('DELETE');

		fetchMock = mockJson({ ok: true, removed_doc_id: '', revoked_knowledge_points: [] });
		await revokePromotion('doc_1');
		expect(fetchMock.mock.calls[0][0]).toBe('/api/staging/revoke');
		expect(JSON.parse(fetchMock.mock.calls[0][1].body).staging_doc_id).toBe('doc_1');
	});

	it('promotes candidates into the target course', async () => {
		const fetchMock = mockJson({
			ok: true,
			target_course_id: 'course_1',
			target_namespace: 'course_1',
			promoted_documents: [],
			promoted_knowledge_points: [],
			errors: []
		});
		const result = await promoteStaging({
			stagingDocIds: ['doc_1'],
			candidateIds: ['cand_1'],
			targetCourseId: 'course_1'
		});
		expect(result.target_namespace).toBe('course_1');
		const body = JSON.parse(fetchMock.mock.calls[0][1].body);
		expect(body.staging_doc_ids).toEqual(['doc_1']);
		expect(body.candidate_ids).toEqual(['cand_1']);
		expect(body.target_course_id).toBe('course_1');
	});

	it('search passes stage flag and surfaces backend errors', async () => {
		const fetchMock = mockJson({ status: 'ok', results: [], staged: [] });
		await searchWeb('光合作用', { topK: 3, stage: true, sessionId: 's1' });
		const body = JSON.parse(fetchMock.mock.calls[0][1].body);
		expect(body.query).toBe('光合作用');
		expect(body.stage).toBe(true);

		mockJson({ detail: '联网功能未开启' }, false, 403);
		await expect(getWebStatus()).rejects.toThrow('联网功能未开启');
	});
});
