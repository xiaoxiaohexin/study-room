import { afterEach, describe, expect, it } from 'vitest';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import {
	DEFAULT_COURSE_ID,
	DEFAULT_RAG_NAMESPACE,
	getActiveCourseId,
	getActiveRagNamespace,
	getActiveUserId,
	learningHeaders,
	learningJsonHeaders
} from '$lib/services/learning-context';

describe('learning-context active scope', () => {
	afterEach(() => {
		conversationsStore.clearActiveConversation();
	});

	it('falls back to default course / namespace when no conversation is active', () => {
		conversationsStore.clearActiveConversation();
		expect(getActiveCourseId()).toBe(DEFAULT_COURSE_ID);
		expect(getActiveRagNamespace()).toBe(DEFAULT_RAG_NAMESPACE);
		expect(getActiveUserId()).toBe('default_user');
	});

	it('reads the course and knowledge base bound to the active conversation', () => {
		conversationsStore.activeConversation = {
			currNode: null,
			id: 'conv-scope-test',
			lastModified: Date.now(),
			name: 'scope test',
			ragNamespace: 'calculus',
			courseId: 'math_calc'
		};

		expect(getActiveRagNamespace()).toBe('calculus');
		expect(getActiveCourseId()).toBe('math_calc');

		const headers = learningHeaders();
		expect(headers['X-RAG-Namespace']).toBe('calculus');
		expect(headers['X-Course-Id']).toBe('math_calc');
		expect(headers['X-User-Id']).toBe('default_user');
	});

	it('learningJsonHeaders merges JSON content type with scope headers', () => {
		conversationsStore.activeConversation = {
			currNode: null,
			id: 'conv-scope-json',
			lastModified: Date.now(),
			name: 'scope json',
			ragNamespace: 'physics'
		};

		const headers = learningJsonHeaders();
		expect(headers['Content-Type']).toBe('application/json');
		expect(headers['X-RAG-Namespace']).toBe('physics');
		expect(headers['X-Course-Id']).toBe(DEFAULT_COURSE_ID);
	});
});
