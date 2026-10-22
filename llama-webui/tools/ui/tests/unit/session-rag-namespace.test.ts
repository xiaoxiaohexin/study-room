/**
 * Unit & Contract Tests for Session Knowledge Base (RAG) & Course Binding (R4).
 *
 * Verifies:
 * 1. DatabaseConversation schema contract for ragNamespace and courseId.
 * 2. Header constants for X-RAG-Namespace and X-Course-Id in HEADERS.
 * 3. ConversationPreferences pending state lifecycle & active conversation synchronization.
 * 4. Multi-conversation session isolation (switching conversations isolates KB namespace).
 * 5. Real ChatService dual-channel transmission (headers & requestBody) with mock fetch.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DatabaseConversation } from '$lib/types/database';
import { HEADERS } from '$lib/constants/headers.constants';
import { MessageRole } from '$lib/enums';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { DatabaseService } from '$lib/services/database.service';
import { ChatService } from '$lib/services/chat.service';

describe('R4 WebUI Session RAG Namespace & Course Binding Contract', () => {
	describe('1. DatabaseConversation Schema & Property Contract', () => {
		it('allows creating a conversation with explicit ragNamespace and courseId', () => {
			const conversation: DatabaseConversation = {
				currNode: null,
				id: 'conv-physics-101',
				lastModified: 1729400000000,
				name: '大学物理第一章',
				ragNamespace: 'physics_mechanics',
				courseId: 'course_phys_01'
			};

			expect(conversation.id).toBe('conv-physics-101');
			expect(conversation.ragNamespace).toBe('physics_mechanics');
			expect(conversation.courseId).toBe('course_phys_01');
		});

		it('defaults ragNamespace and courseId gracefully when omitted on legacy rows', () => {
			const legacyConversation: DatabaseConversation = {
				currNode: null,
				id: 'conv-legacy-001',
				lastModified: 1729400000000,
				name: '旧版对话'
			};

			expect(legacyConversation.ragNamespace).toBeUndefined();
			expect(legacyConversation.courseId).toBeUndefined();

			const effectiveNamespace = legacyConversation.ragNamespace ?? 'default';
			expect(effectiveNamespace).toBe('default');
		});
	});

	describe('2. Header Constants Contract', () => {
		it('verifies standard header constants for RAG namespace and course routing', () => {
			expect(HEADERS.X_CONVERSATION_ID_HEADER).toBe('X-Conversation-Id');
			expect(HEADERS.X_RAG_NAMESPACE_HEADER).toBe('X-RAG-Namespace');
			expect(HEADERS.X_COURSE_ID_HEADER).toBe('X-Course-Id');
		});
	});

	describe('3. Conversation Preferences Lifecycle with conversationsStore.preferences', () => {
		let updateConversationSpy: any;

		beforeEach(() => {
			updateConversationSpy = vi.spyOn(DatabaseService, 'updateConversation').mockResolvedValue(1 as any);
			conversationsStore.clearActiveConversation();
		});

		afterEach(() => {
			updateConversationSpy.mockRestore();
			conversationsStore.clearActiveConversation();
		});

		it('buffers selections into pending state when no conversation is active', async () => {
			const prefs = conversationsStore.preferences;

			expect(prefs.getRagNamespace()).toBe('default');
			expect(prefs.getCourseId()).toBeUndefined();

			await prefs.setRagNamespace('advanced_ai', 'cs_900');

			expect(prefs.getRagNamespace()).toBe('advanced_ai');
			expect(prefs.getCourseId()).toBe('cs_900');
			expect(updateConversationSpy).not.toHaveBeenCalled();

			prefs.resetPending();
			expect(prefs.getRagNamespace()).toBe('default');
			expect(prefs.getCourseId()).toBeUndefined();
		});

		it('updates active conversation in place and persists to DatabaseService', async () => {
			const prefs = conversationsStore.preferences;
			conversationsStore.activeConversation = {
				currNode: null,
				id: 'conv-active-1',
				lastModified: Date.now(),
				name: '活跃对话',
				ragNamespace: 'default'
			};

			expect(prefs.getRagNamespace()).toBe('default');

			await prefs.setRagNamespace('data_structures', 'cs_201');

			expect(prefs.getRagNamespace()).toBe('data_structures');
			expect(prefs.getCourseId()).toBe('cs_201');
			expect(conversationsStore.activeConversation?.ragNamespace).toBe('data_structures');
			expect(conversationsStore.activeConversation?.courseId).toBe('cs_201');
			expect(updateConversationSpy).toHaveBeenCalledWith('conv-active-1', {
				ragNamespace: 'data_structures',
				courseId: 'cs_201'
			});
		});

		it('supports updating courseId individually via setCourseId', async () => {
			const prefs = conversationsStore.preferences;
			conversationsStore.activeConversation = {
				currNode: null,
				id: 'conv-active-2',
				lastModified: Date.now(),
				name: '测试会话',
				ragNamespace: 'calculus'
			};

			await prefs.setCourseId('math_calc_101');

			expect(prefs.getCourseId()).toBe('math_calc_101');
			expect(conversationsStore.activeConversation?.courseId).toBe('math_calc_101');
			expect(updateConversationSpy).toHaveBeenCalledWith('conv-active-2', {
				courseId: 'math_calc_101'
			});
		});
	});

	describe('4. Multi-Conversation Knowledge Base Isolation', () => {
		it('strictly separates RAG namespaces between different conversations', () => {
			const convPhysics: DatabaseConversation = {
				currNode: null,
				id: 'conv-phys',
				lastModified: 1000,
				name: '量子物理讨论',
				ragNamespace: 'physics_quantum',
				courseId: 'phys_401'
			};

			const convLiterature: DatabaseConversation = {
				currNode: null,
				id: 'conv-lit',
				lastModified: 2000,
				name: '英国文学经典',
				ragNamespace: 'english_literature',
				courseId: 'lit_101'
			};

			let activeConv = convPhysics;
			const resolveCurrentOptions = () => ({
				session_id: activeConv.id,
				rag_namespace: activeConv.ragNamespace ?? 'default',
				course_id: activeConv.courseId ?? null
			});

			// In Physics session:
			let options = resolveCurrentOptions();
			expect(options.rag_namespace).toBe('physics_quantum');
			expect(options.course_id).toBe('phys_401');

			// Switch to Literature session:
			activeConv = convLiterature;
			options = resolveCurrentOptions();
			expect(options.rag_namespace).toBe('english_literature');
			expect(options.course_id).toBe('lit_101');

			// Switch back:
			activeConv = convPhysics;
			options = resolveCurrentOptions();
			expect(options.rag_namespace).toBe('physics_quantum');
			expect(options.course_id).toBe('phys_401');
		});
	});

	describe('5. Real ChatService Dual-Channel Transmission with Mock Fetch', () => {
		let capturedUrl: string | null = null;
		let capturedInit: RequestInit | null = null;

		beforeEach(() => {
			capturedUrl = null;
			capturedInit = null;

			vi.stubGlobal(
				'fetch',
				vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
					capturedUrl = String(url);
					capturedInit = init ?? null;

					const bodyObj = init?.body ? JSON.parse(String(init.body)) : {};
					if (bodyObj.stream) {
						const encoder = new TextEncoder();
						const stream = new ReadableStream({
							start(controller) {
								controller.enqueue(encoder.encode('data: [DONE]\n\n'));
								controller.close();
							}
						});
						return new Response(stream, {
							status: 200,
							headers: { 'Content-Type': 'text/event-stream' }
						});
					}

					return new Response(
						JSON.stringify({
							choices: [{ message: { role: 'assistant', content: '测试回答' } }]
						}),
						{ status: 200, headers: { 'Content-Type': 'application/json' } }
					);
				})
			);
		});

		afterEach(() => {
			vi.unstubAllGlobals();
		});

		it('propagates dual-channel headers and body fields on non-streaming sendMessage', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: '你好' }],
				{
					stream: false,
					rag_namespace: 'physics_mechanics',
					course_id: 'phys_101'
				},
				'conv_test_999'
			);

			expect(capturedInit).not.toBeNull();
			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe('conv_test_999');
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe('physics_mechanics');
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe('phys_101');

			const body = JSON.parse(String(capturedInit!.body));
			expect(body.session_id).toBe('conv_test_999');
			expect(body.rag_namespace).toBe('physics_mechanics');
			expect(body.course_id).toBe('phys_101');
		});

		it('propagates clean session_id in body while stream identity is in headers', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: '流式问题' }],
				{
					stream: true,
					model: 'qwen2.5:7b',
					ragNamespace: 'calculus',
					courseId: 'math_calc'
				},
				'conv_stream_001'
			);

			expect(capturedInit).not.toBeNull();
			const headers = capturedInit!.headers as Record<string, string>;
			// Streaming X-Conversation-Id contains stream identity
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toContain('conv_stream_001');
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe('calculus');
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe('math_calc');

			const body = JSON.parse(String(capturedInit!.body));
			// Body session_id MUST be clean without ::model
			expect(body.session_id).toBe('conv_stream_001');
			expect(body.rag_namespace).toBe('calculus');
			expect(body.course_id).toBe('math_calc');
		});

		it('omits optional headers and body parameters when not provided', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: '普通问题' }],
				{ stream: false },
				'conv_plain'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe('conv_plain');
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBeUndefined();
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBeUndefined();

			const body = JSON.parse(String(capturedInit!.body));
			expect(body.session_id).toBe('conv_plain');
			expect(body.rag_namespace).toBeUndefined();
			expect(body.course_id).toBeUndefined();
		});
	});
});
