/**
 * Empirical Adversarial Test Suite for ChatService API Transmission & Backend Contract Alignment
 *
 * Verifies Milestone M4 interface contracts:
 * 1. Dual-channel passthrough: headers (X-RAG-Namespace, X-Course-Id, X-Conversation-Id)
 *    and body (rag_namespace, course_id, session_id).
 * 2. Streaming vs non-streaming header handling (streamIdentity vs raw session ID).
 * 3. Omission & cleanliness of undefined / null / whitespace fields (zero pollution).
 * 4. Precedence rules: snake_case vs camelCase, conversationId vs session_id vs sessionId.
 * 5. Whitespace trimming and boundary preservation for complex namespaces.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { HEADERS } from '$lib/constants/headers.constants';
import { MessageRole } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';

describe('Adversarial API Contract Passthrough: ChatService.sendMessage', () => {
	let capturedUrl: string | null = null;
	let capturedInit: RequestInit | null = null;
	let capturedBodyText: string | null = null;
	let capturedBody: any = null;

	beforeEach(() => {
		capturedUrl = null;
		capturedInit = null;
		capturedBodyText = null;
		capturedBody = null;

		vi.stubGlobal(
			'fetch',
			vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
				capturedUrl = String(url);
				capturedInit = init ?? null;
				capturedBodyText = init?.body ? String(init.body) : null;
				capturedBody = capturedBodyText ? JSON.parse(capturedBodyText) : null;

				if (capturedBody?.stream) {
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
						choices: [{ message: { role: 'assistant', content: 'adversarial response' } }]
					}),
					{ status: 200, headers: { 'Content-Type': 'application/json' } }
				);
			})
		);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	describe('1. Dual-Channel Passthrough (Headers & Body Parity)', () => {
		it('transmits complete headers and body in non-streaming mode', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Explain thermodynamics' }],
				{
					stream: false,
					rag_namespace: 'physics_thermo',
					course_id: 'phys_202'
				},
				'conv_session_101'
			);

			expect(capturedInit).not.toBeNull();
			const headers = capturedInit!.headers as Record<string, string>;

			// Headers
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe('conv_session_101');
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe('physics_thermo');
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe('phys_202');

			// Body
			expect(capturedBody.session_id).toBe('conv_session_101');
			expect(capturedBody.rag_namespace).toBe('physics_thermo');
			expect(capturedBody.course_id).toBe('phys_202');
		});

		it('transmits complete headers and body in streaming mode with model', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Solve eigenvalue equation' }],
				{
					stream: true,
					model: 'qwen2.5:14b',
					rag_namespace: 'linear_algebra',
					course_id: 'math_la_1'
				},
				'conv_session_stream_01'
			);

			expect(capturedInit).not.toBeNull();
			const headers = capturedInit!.headers as Record<string, string>;

			// Header X-Conversation-Id must contain streamIdentity(convId, model)
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe('conv_session_stream_01::qwen2.5:14b');
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe('linear_algebra');
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe('math_la_1');

			// Body session_id MUST be clean without ::model
			expect(capturedBody.session_id).toBe('conv_session_stream_01');
			expect(capturedBody.rag_namespace).toBe('linear_algebra');
			expect(capturedBody.course_id).toBe('math_la_1');
		});
	});

	describe('2. Streaming vs Non-Streaming StreamIdentity Separation', () => {
		it('appends model suffix to header in streaming mode, but leaves body session_id clean', async () => {
			const convId = 'session_alpha_99';
			const model = 'deepseek-r1:7b';

			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Stream test' }],
				{
					stream: true,
					model,
					rag_namespace: 'default'
				},
				convId
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe(`${convId}::${model}`);
			expect(capturedBody.session_id).toBe(convId);
			expect(capturedBody.session_id).not.toContain('::');
			expect(capturedBody.session_id).not.toContain(model);
		});

		it('does NOT append model suffix in non-streaming mode even if model is specified', async () => {
			const convId = 'session_alpha_99';
			const model = 'deepseek-r1:7b';

			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Non-stream test' }],
				{
					stream: false,
					model,
					rag_namespace: 'default'
				},
				convId
			);

			const headers = capturedInit!.headers as Record<string, string>;
			// Non-streaming must NOT have ::model in X-Conversation-Id
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe(convId);
			expect(capturedBody.session_id).toBe(convId);
		});

		it('streaming without model uses raw conversationId for both header and body', async () => {
			const convId = 'session_no_model';

			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Stream without explicit model' }],
				{
					stream: true
				},
				convId
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe(convId);
			expect(capturedBody.session_id).toBe(convId);
		});
	});

	describe('3. Clean Omission & Zero Pollution (Undefined, Null, Whitespace)', () => {
		it('cleanly omits headers and body fields when not provided in options', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Hello' }],
				{ stream: false },
				'conv_empty_meta'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBeUndefined();
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBeUndefined();

			expect(capturedBody).not.toHaveProperty('rag_namespace');
			expect(capturedBody).not.toHaveProperty('course_id');
			expect(capturedBodyText).not.toContain('rag_namespace');
			expect(capturedBodyText).not.toContain('course_id');
		});

		it('never stringifies literal "undefined" or "null" when explicitly passed as undefined/null', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Null check' }],
				{
					stream: false,
					rag_namespace: undefined,
					course_id: null as any,
					ragNamespace: undefined,
					courseId: null as any
				},
				'conv_null_check'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBeUndefined();
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBeUndefined();

			expect(capturedBody).not.toHaveProperty('rag_namespace');
			expect(capturedBody).not.toHaveProperty('course_id');

			// Raw payload must not contain dirty stringified values
			expect(capturedBodyText).not.toContain('"undefined"');
			expect(capturedBodyText).not.toContain('"null"');
			expect(capturedBodyText).not.toContain('rag_namespace');
			expect(capturedBodyText).not.toContain('course_id');
		});

		it('trims whitespace and treats whitespace-only strings as empty/omitted', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Whitespace test' }],
				{
					stream: false,
					rag_namespace: '   \t  \n  ',
					course_id: '    '
				},
				'conv_ws_check'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			// Falsy after trim -> omitted
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBeUndefined();
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBeUndefined();

			expect(capturedBody).not.toHaveProperty('rag_namespace');
			expect(capturedBody).not.toHaveProperty('course_id');
		});

		it('trims leading and trailing spaces for valid namespace and courseId', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Trim test' }],
				{
					stream: false,
					rag_namespace: '  cs_operating_systems   ',
					course_id: '   CS_301  '
				},
				'conv_trim_check'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe('cs_operating_systems');
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe('CS_301');

			expect(capturedBody.rag_namespace).toBe('cs_operating_systems');
			expect(capturedBody.course_id).toBe('CS_301');
		});
	});

	describe('4. Precedence Rules & Alias Resolution', () => {
		it('prefers conversationId argument over options.session_id and options.sessionId', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Precedence test' }],
				{
					stream: false,
					session_id: 'option_session_snake',
					sessionId: 'option_session_camel'
				},
				'arg_conversation_id'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe('arg_conversation_id');
			expect(capturedBody.session_id).toBe('arg_conversation_id');
		});

		it('falls back to options.session_id when conversationId is undefined', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Fallback test 1' }],
				{
					stream: false,
					session_id: 'option_session_snake',
					sessionId: 'option_session_camel'
				},
				undefined
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe('option_session_snake');
			expect(capturedBody.session_id).toBe('option_session_snake');
		});

		it('falls back to options.sessionId when conversationId and session_id are undefined', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Fallback test 2' }],
				{
					stream: false,
					sessionId: 'option_session_camel'
				},
				undefined
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe('option_session_camel');
			expect(capturedBody.session_id).toBe('option_session_camel');
		});

		it('prefers snake_case rag_namespace over camelCase ragNamespace when both are present', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Alias test' }],
				{
					stream: false,
					rag_namespace: 'primary_namespace',
					ragNamespace: 'secondary_namespace'
				},
				'conv_alias'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe('primary_namespace');
			expect(capturedBody.rag_namespace).toBe('primary_namespace');
		});

		it('accepts camelCase ragNamespace when snake_case is omitted', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'CamelCase test' }],
				{
					stream: false,
					ragNamespace: 'pure_camel_ns'
				},
				'conv_camel'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe('pure_camel_ns');
			expect(capturedBody.rag_namespace).toBe('pure_camel_ns');
		});

		it('prefers snake_case course_id over camelCase courseId when both are present', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Course alias test' }],
				{
					stream: false,
					course_id: 'primary_course',
					courseId: 'secondary_course'
				},
				'conv_course_alias'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe('primary_course');
			expect(capturedBody.course_id).toBe('primary_course');
		});

		it('accepts camelCase courseId when snake_case is omitted', async () => {
			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'CamelCourse test' }],
				{
					stream: false,
					courseId: 'pure_camel_course'
				},
				'conv_camel_course'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe('pure_camel_course');
			expect(capturedBody.course_id).toBe('pure_camel_course');
		});
	});

	describe('5. Complex Namespace Characters & Unicode', () => {
		it('handles complex and unicode namespace identifiers without corruption', async () => {
			const complexNs = 'course/cs-401.v2:高等数学_微积分';
			const complexCourse = 'COURSE::MATH-99#ALPHA';

			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Complex identifier' }],
				{
					stream: false,
					rag_namespace: complexNs,
					course_id: complexCourse
				},
				'conv_complex'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe(complexNs);
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe(complexCourse);

			expect(capturedBody.rag_namespace).toBe(complexNs);
			expect(capturedBody.course_id).toBe(complexCourse);
		});
	});

	describe('6. Integration with chatStore.getApiOptions', () => {
		it('integrates cleanly when chatStore.getApiOptions yields bindings', async () => {
			// Mock conversation preferences
			conversationsStore.clearActiveConversation();
			await conversationsStore.preferences.setRagNamespace('advanced_algorithms', 'cs_algo_400');

			const options = chatStore.getApiOptions();
			expect(options.rag_namespace).toBe('advanced_algorithms');
			expect(options.ragNamespace).toBe('advanced_algorithms');
			expect(options.course_id).toBe('cs_algo_400');
			expect(options.courseId).toBe('cs_algo_400');

			await ChatService.sendMessage(
				[{ role: MessageRole.USER, content: 'Graph shortest path' }],
				options,
				'conv_store_integration'
			);

			const headers = capturedInit!.headers as Record<string, string>;
			expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe('advanced_algorithms');
			expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe('cs_algo_400');

			expect(capturedBody.rag_namespace).toBe('advanced_algorithms');
			expect(capturedBody.course_id).toBe('cs_algo_400');

			conversationsStore.preferences.resetPending();
		});
	});
});
