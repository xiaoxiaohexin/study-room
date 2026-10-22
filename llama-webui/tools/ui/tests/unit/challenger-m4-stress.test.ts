/**
 * Challenger Adversarial Stress Suite for Milestone M4:
 * Frontend WebUI Session Knowledge Base (RAG) & Course State Isolation.
 *
 * Covers:
 * 1. High frequency session switching across diverse namespaces (120+ switches + live mutations).
 * 2. Empty/whitespace and adversarial custom namespace input sanitization.
 * 3. Deep fork child independence and chained inheritance (Parent -> Child -> Grandchild).
 * 4. resetPending() state sanitization across new-chat, active, and loadConversation lifecycles.
 * 5. Concurrent dual-channel ChatService transmission (header/body integrity & stream identity).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DatabaseConversation } from '$lib/types/database';
import { HEADERS } from '$lib/constants/headers.constants';
import { MessageRole } from '$lib/enums';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { DatabaseService } from '$lib/services/database.service';
import { ChatService } from '$lib/services/chat.service';
import { chatStore } from '$lib/stores/chat/index.svelte';
import {
	getKnowledgeBaseBadgeLabel,
	PRESET_KNOWLEDGE_BASES
} from '$lib/constants/knowledge-base.constants';

describe('Milestone M4 Challenger Adversarial Stress Suite', () => {
	let updateConversationSpy: any;
	let getConversationSpy: any;
	let getMessagesSpy: any;

	beforeEach(() => {
		updateConversationSpy = vi
			.spyOn(DatabaseService, 'updateConversation')
			.mockResolvedValue(1 as any);
		getConversationSpy = vi.spyOn(DatabaseService, 'getConversation');
		getMessagesSpy = vi
			.spyOn(DatabaseService, 'getConversationMessages')
			.mockResolvedValue([]);
		conversationsStore.clearActiveConversation();
		conversationsStore.conversations = [];
	});

	afterEach(() => {
		updateConversationSpy.mockRestore();
		getConversationSpy.mockRestore();
		getMessagesSpy.mockRestore();
		conversationsStore.clearActiveConversation();
		conversationsStore.conversations = [];
	});

	// =========================================================================
	// 1. Adversarial Input Sanitization & Edge Cases
	// =========================================================================
	describe('1. Adversarial Input Sanitization & Boundary Handling', () => {
		it('falls back to "default" for empty, whitespace, null and undefined inputs on new chat', async () => {
			const prefs = conversationsStore.preferences;

			// Empty string
			await prefs.setRagNamespace('');
			expect(prefs.getRagNamespace()).toBe('default');

			// Multiple spaces
			await prefs.setRagNamespace('     ');
			expect(prefs.getRagNamespace()).toBe('default');

			// Tab, newline, carriage return
			await prefs.setRagNamespace('\t\r\n   ');
			expect(prefs.getRagNamespace()).toBe('default');

			// Null
			await prefs.setRagNamespace(null);
			expect(prefs.getRagNamespace()).toBe('default');

			// Undefined as any
			await prefs.setRagNamespace(undefined as any);
			expect(prefs.getRagNamespace()).toBe('default');
		});

		it('properly trims leading/trailing whitespace without corrupting namespace', async () => {
			const prefs = conversationsStore.preferences;

			await prefs.setRagNamespace('   calculus   ');
			expect(prefs.getRagNamespace()).toBe('calculus');

			await prefs.setRagNamespace('\t\n  physics_mechanics  \r\n');
			expect(prefs.getRagNamespace()).toBe('physics_mechanics');
		});

		it('handles complex, unicode, Chinese, and special character namespaces robustly', async () => {
			const prefs = conversationsStore.preferences;

			const unicodeTests = [
				'高等数学:微积分/极限',
				'rag_ns-123.456_beta',
				'AI & DeepLearning 🤖',
				'namespace.with.many.dots.and-dashes',
				'a'.repeat(500) // Extremely long namespace
			];

			for (const ns of unicodeTests) {
				await prefs.setRagNamespace(ns);
				expect(prefs.getRagNamespace()).toBe(ns.trim());
			}
		});

		it('sanitizes courseId parameter alongside namespace in setRagNamespace', async () => {
			const prefs = conversationsStore.preferences;

			// Empty string courseId becomes undefined
			await prefs.setRagNamespace('calculus', '');
			expect(prefs.getRagNamespace()).toBe('calculus');
			expect(prefs.getCourseId()).toBeUndefined();

			// Whitespace courseId becomes undefined
			await prefs.setRagNamespace('calculus', '   \t  ');
			expect(prefs.getCourseId()).toBeUndefined();

			// Null courseId clears course
			await prefs.setRagNamespace('calculus', null);
			expect(prefs.getCourseId()).toBeUndefined();

			// Padded courseId is trimmed cleanly
			await prefs.setRagNamespace('calculus', '   math_101   ');
			expect(prefs.getCourseId()).toBe('math_101');
		});

		it('sanitizes courseId in setCourseId directly', async () => {
			const prefs = conversationsStore.preferences;

			await prefs.setCourseId('   cs_net_202   ');
			expect(prefs.getCourseId()).toBe('cs_net_202');

			await prefs.setCourseId('');
			expect(prefs.getCourseId()).toBeUndefined();

			await prefs.setCourseId('   ');
			expect(prefs.getCourseId()).toBeUndefined();

			await prefs.setCourseId(null);
			expect(prefs.getCourseId()).toBeUndefined();
		});

		it('applies input sanitization when an active conversation is present', async () => {
			const prefs = conversationsStore.preferences;
			const conv: DatabaseConversation = {
				currNode: null,
				id: 'active-edge-conv',
				lastModified: 1000,
				name: 'Active Conv',
				ragNamespace: 'physics',
				courseId: 'phys_gen'
			};
			conversationsStore.conversations = [conv];
			conversationsStore.activeConversation = conv;

			// Set whitespace -> should reset to default and update DB
			await prefs.setRagNamespace('   \n\t  ', '   ');
			expect(conv.ragNamespace).toBe('default');
			expect(conv.courseId).toBeUndefined();
			expect(updateConversationSpy).toHaveBeenCalledWith('active-edge-conv', {
				ragNamespace: 'default',
				courseId: undefined
			});

			// Set trimmed custom with trimmed course
			await prefs.setRagNamespace('  linear_algebra  ', '  math_linalg  ');
			expect(conv.ragNamespace).toBe('linear_algebra');
			expect(conv.courseId).toBe('math_linalg');
			expect(updateConversationSpy).toHaveBeenCalledWith('active-edge-conv', {
				ragNamespace: 'linear_algebra',
				courseId: 'math_linalg'
			});
		});

		it('verifies getKnowledgeBaseBadgeLabel handles boundary edge cases', () => {
			expect(getKnowledgeBaseBadgeLabel(undefined)).toBeNull();
			expect(getKnowledgeBaseBadgeLabel(null as any)).toBeNull();
			expect(getKnowledgeBaseBadgeLabel('')).toBeNull();
			expect(getKnowledgeBaseBadgeLabel('default')).toBeNull();

			// Course fallback when namespace is default or empty
			expect(getKnowledgeBaseBadgeLabel('default', 'MATH_101')).toBe('MATH_101');
			expect(getKnowledgeBaseBadgeLabel('', 'PHYS_201')).toBe('PHYS_201');
			expect(getKnowledgeBaseBadgeLabel(undefined, 'CS_301')).toBe('CS_301');

			// Presets return short badges
			expect(getKnowledgeBaseBadgeLabel('calculus')).toBe('高数');
			expect(getKnowledgeBaseBadgeLabel('physics')).toBe('物理');
			expect(getKnowledgeBaseBadgeLabel('linear_algebra')).toBe('线代');
			expect(getKnowledgeBaseBadgeLabel('cs_network')).toBe('计网');

			// Custom namespaces return raw label
			expect(getKnowledgeBaseBadgeLabel('astrophysics_v2')).toBe('astrophysics_v2');
			expect(getKnowledgeBaseBadgeLabel('ai_agent_lab', 'CS_500')).toBe('ai_agent_lab');
		});
	});

	// =========================================================================
	// 2. High Frequency Session Switching Stress Test
	// =========================================================================
	describe('2. High Frequency Multi-Session Switching Stress Test', () => {
		it('executes 120 rapid session switches across 6 diverse conversations without state leakage', () => {
			const convCalc: DatabaseConversation = {
				currNode: null,
				id: 'conv-calc',
				lastModified: 1000,
				name: '微积分',
				ragNamespace: 'calculus',
				courseId: 'math_calc'
			};
			const convPhys: DatabaseConversation = {
				currNode: null,
				id: 'conv-phys',
				lastModified: 2000,
				name: '大学物理',
				ragNamespace: 'physics',
				courseId: 'phys_gen'
			};
			const convLinAlg: DatabaseConversation = {
				currNode: null,
				id: 'conv-linalg',
				lastModified: 3000,
				name: '线性代数',
				ragNamespace: 'linear_algebra',
				courseId: 'math_linalg'
			};
			const convCS: DatabaseConversation = {
				currNode: null,
				id: 'conv-cs',
				lastModified: 4000,
				name: '计算机网络',
				ragNamespace: 'cs_network',
				courseId: 'cs_net'
			};
			const convCustom: DatabaseConversation = {
				currNode: null,
				id: 'conv-custom',
				lastModified: 5000,
				name: '机器学习实验室',
				ragNamespace: 'custom_ai_lab',
				courseId: 'cs_999'
			};
			const convLegacy: DatabaseConversation = {
				currNode: null,
				id: 'conv-legacy',
				lastModified: 6000,
				name: '旧版未绑定对话'
				// ragNamespace and courseId are undefined
			};

			const convList = [convCalc, convPhys, convLinAlg, convCS, convCustom, convLegacy];
			conversationsStore.conversations = [...convList];

			const prefs = conversationsStore.preferences;

			// Run 120 rapid switches in varying sequence
			for (let i = 0; i < 120; i++) {
				const targetConv = convList[i % convList.length];
				conversationsStore.activeConversation = targetConv;

				// Verify preferences resolve correctly
				const expectedNs = targetConv.ragNamespace || 'default';
				const expectedCourse = targetConv.courseId ?? undefined;

				expect(prefs.getRagNamespace()).toBe(expectedNs);
				expect(prefs.getCourseId()).toBe(expectedCourse);

				// Verify chatStore.getApiOptions() produces exact isolation
				const apiOptions = chatStore.getApiOptions();
				expect(apiOptions.rag_namespace).toBe(expectedNs);
				expect(apiOptions.ragNamespace).toBe(expectedNs);

				if (expectedCourse) {
					expect(apiOptions.course_id).toBe(expectedCourse);
					expect(apiOptions.courseId).toBe(expectedCourse);
				} else {
					expect(apiOptions.course_id).toBeUndefined();
					expect(apiOptions.courseId).toBeUndefined();
				}
			}
		});

		it('maintains strict isolation when in-place mutations occur during rapid switching', async () => {
			const convCalc: DatabaseConversation = {
				currNode: null,
				id: 'conv-calc-m',
				lastModified: 1000,
				name: '微积分',
				ragNamespace: 'calculus',
				courseId: 'math_calc'
			};
			const convPhys: DatabaseConversation = {
				currNode: null,
				id: 'conv-phys-m',
				lastModified: 2000,
				name: '大学物理',
				ragNamespace: 'physics',
				courseId: 'phys_gen'
			};
			const convLegacy: DatabaseConversation = {
				currNode: null,
				id: 'conv-legacy-m',
				lastModified: 3000,
				name: '未绑定会话'
			};

			const pool = [convCalc, convPhys, convLegacy];
			conversationsStore.conversations = [...pool];
			const prefs = conversationsStore.preferences;

			let physMutated = false;
			let legacyMutated = false;

			for (let i = 0; i < 60; i++) {
				const target = pool[i % pool.length];
				conversationsStore.activeConversation = target;

				// Mid-run mutation when Physics is active
				if (target === convPhys && !physMutated && i >= 15) {
					await prefs.setRagNamespace('physics_astronomy', 'phys_astro_400');
					expect(convPhys.ragNamespace).toBe('physics_astronomy');
					expect(convPhys.courseId).toBe('phys_astro_400');
					physMutated = true;
				}

				// Mid-run mutation when Legacy is active
				if (target === convLegacy && !legacyMutated && i >= 30) {
					await prefs.setRagNamespace('biology_genetics', 'bio_101');
					expect(convLegacy.ragNamespace).toBe('biology_genetics');
					expect(convLegacy.courseId).toBe('bio_101');
					legacyMutated = true;
				}

				// Check active conversation resolution
				const expectedNs = target.ragNamespace || 'default';
				const expectedCourse = target.courseId ?? undefined;
				expect(prefs.getRagNamespace()).toBe(expectedNs);
				expect(prefs.getCourseId()).toBe(expectedCourse);

				// Calculus must have NEVER been mutated
				expect(convCalc.ragNamespace).toBe('calculus');
				expect(convCalc.courseId).toBe('math_calc');
			}
		});
	});

	// =========================================================================
	// 3. Deep Fork Child Independence & Chained Inheritance Stress
	// =========================================================================
	describe('3. Deep Fork Child Independence & Chained Inheritance', () => {
		it('modifying child KB does NOT mutate parent KB (Direct Fork Independence)', async () => {
			const parentConv: DatabaseConversation = {
				currNode: 'msg-parent-leaf',
				id: 'conv-parent-01',
				lastModified: 1000,
				name: '父会话: 经典力学',
				ragNamespace: 'physics',
				courseId: 'phys_gen'
			};

			// Simulate DatabaseService.forkConversation creating child
			const childConv: DatabaseConversation = {
				currNode: 'msg-child-leaf',
				id: 'conv-child-01',
				forkedFromConversationId: parentConv.id,
				lastModified: 2000,
				name: '子会话: 分支力学',
				ragNamespace: parentConv.ragNamespace,
				courseId: parentConv.courseId
			};

			conversationsStore.conversations = [parentConv, childConv];
			conversationsStore.activeConversation = childConv;

			const prefs = conversationsStore.preferences;
			expect(prefs.getRagNamespace()).toBe('physics');
			expect(prefs.getCourseId()).toBe('phys_gen');

			// Mutate child knowledge base to calculus
			await prefs.setRagNamespace('calculus', 'math_calc');

			// Assert child updated
			expect(childConv.ragNamespace).toBe('calculus');
			expect(childConv.courseId).toBe('math_calc');
			expect(prefs.getRagNamespace()).toBe('calculus');
			expect(prefs.getCourseId()).toBe('math_calc');

			// CRITICAL ASSERTION: Parent conversation MUST remain 100% unchanged!
			expect(parentConv.ragNamespace).toBe('physics');
			expect(parentConv.courseId).toBe('phys_gen');

			// Verify DatabaseService.updateConversation was called with child ID, NEVER parent ID
			expect(updateConversationSpy).toHaveBeenCalledWith('conv-child-01', {
				ragNamespace: 'calculus',
				courseId: 'math_calc'
			});
			expect(updateConversationSpy).not.toHaveBeenCalledWith(
				'conv-parent-01',
				expect.anything()
			);
		});

		it('ensures 3-generation chained forks remain completely isolated (Parent -> Child -> Grandchild)', async () => {
			const parent: DatabaseConversation = {
				currNode: 'n1',
				id: 'conv-root',
				lastModified: 1000,
				name: 'Generation 1: Root',
				ragNamespace: 'physics',
				courseId: 'phys_gen'
			};
			const child: DatabaseConversation = {
				currNode: 'n2',
				id: 'conv-gen2',
				forkedFromConversationId: parent.id,
				lastModified: 2000,
				name: 'Generation 2: Child',
				ragNamespace: 'calculus',
				courseId: 'math_calc'
			};
			const grandchild: DatabaseConversation = {
				currNode: 'n3',
				id: 'conv-gen3',
				forkedFromConversationId: child.id,
				lastModified: 3000,
				name: 'Generation 3: Grandchild',
				ragNamespace: 'calculus',
				courseId: 'math_calc'
			};

			conversationsStore.conversations = [parent, child, grandchild];
			const prefs = conversationsStore.preferences;

			// 1. Activate Grandchild and mutate to linear_algebra
			conversationsStore.activeConversation = grandchild;
			await prefs.setRagNamespace('linear_algebra', 'math_linalg');

			expect(grandchild.ragNamespace).toBe('linear_algebra');
			expect(grandchild.courseId).toBe('math_linalg');
			expect(child.ragNamespace).toBe('calculus');
			expect(child.courseId).toBe('math_calc');
			expect(parent.ragNamespace).toBe('physics');
			expect(parent.courseId).toBe('phys_gen');

			// 2. Activate Root and mutate to cs_network
			conversationsStore.activeConversation = parent;
			await prefs.setRagNamespace('cs_network', 'cs_net');

			expect(parent.ragNamespace).toBe('cs_network');
			expect(parent.courseId).toBe('cs_net');
			expect(child.ragNamespace).toBe('calculus');
			expect(child.courseId).toBe('math_calc');
			expect(grandchild.ragNamespace).toBe('linear_algebra');
			expect(grandchild.courseId).toBe('math_linalg');

			// 3. Activate Child and mutate courseId only
			conversationsStore.activeConversation = child;
			await prefs.setCourseId('math_calc_honors');

			expect(child.ragNamespace).toBe('calculus');
			expect(child.courseId).toBe('math_calc_honors');
			expect(parent.courseId).toBe('cs_net');
			expect(grandchild.courseId).toBe('math_linalg');
		});

		it('inherits and isolates gracefully when forking a legacy conversation with undefined KB fields', async () => {
			const legacyParent: DatabaseConversation = {
				currNode: 'root-msg',
				id: 'conv-legacy-parent',
				lastModified: 500,
				name: 'Legacy Parent'
			};

			const legacyChild: DatabaseConversation = {
				currNode: 'child-msg',
				id: 'conv-legacy-child',
				forkedFromConversationId: legacyParent.id,
				lastModified: 600,
				name: 'Legacy Child',
				ragNamespace: legacyParent.ragNamespace,
				courseId: legacyParent.courseId
			};

			conversationsStore.conversations = [legacyParent, legacyChild];
			conversationsStore.activeConversation = legacyChild;

			const prefs = conversationsStore.preferences;
			expect(prefs.getRagNamespace()).toBe('default');
			expect(prefs.getCourseId()).toBeUndefined();

			// Bind child to biology
			await prefs.setRagNamespace('biology', 'bio_101');

			expect(legacyChild.ragNamespace).toBe('biology');
			expect(legacyChild.courseId).toBe('bio_101');

			// Parent MUST remain undefined
			expect(legacyParent.ragNamespace).toBeUndefined();
			expect(legacyParent.courseId).toBeUndefined();
		});
	});

	// =========================================================================
	// 4. resetPending() State Sanitization & Lifecycle Boundaries
	// =========================================================================
	describe('4. resetPending() State Sanitization & Lifecycle Boundaries', () => {
		it('resets pending state on new-chat screen without residuals', async () => {
			const prefs = conversationsStore.preferences;

			// Buffer pending picks
			await prefs.setRagNamespace('advanced_machine_learning', 'cs_888');
			expect(prefs.pendingRagNamespace).toBe('advanced_machine_learning');
			expect(prefs.pendingCourseId).toBe('cs_888');
			expect(prefs.getRagNamespace()).toBe('advanced_machine_learning');
			expect(prefs.getCourseId()).toBe('cs_888');

			// Trigger resetPending()
			prefs.resetPending();

			expect(prefs.pendingRagNamespace).toBe('default');
			expect(prefs.pendingCourseId).toBeNull();
			expect(prefs.getRagNamespace()).toBe('default');
			expect(prefs.getCourseId()).toBeUndefined();
		});

		it('ensures resetPending() does NOT wipe out active conversation options', () => {
			const activeConv: DatabaseConversation = {
				currNode: null,
				id: 'conv-active-safe',
				lastModified: 1000,
				name: 'Active Safe',
				ragNamespace: 'physics_quantum',
				courseId: 'phys_401'
			};
			conversationsStore.conversations = [activeConv];
			conversationsStore.activeConversation = activeConv;

			const prefs = conversationsStore.preferences;
			expect(prefs.getRagNamespace()).toBe('physics_quantum');
			expect(prefs.getCourseId()).toBe('phys_401');

			// Calling resetPending() while active conversation is loaded
			prefs.resetPending();

			// Active conversation MUST still resolve to physics_quantum
			expect(prefs.getRagNamespace()).toBe('physics_quantum');
			expect(prefs.getCourseId()).toBe('phys_401');
			expect(activeConv.ragNamespace).toBe('physics_quantum');
			expect(activeConv.courseId).toBe('phys_401');
		});

		it('clearActiveConversation() thoroughly sanitizes active and pending state', async () => {
			const activeConv: DatabaseConversation = {
				currNode: null,
				id: 'conv-to-clear',
				lastModified: 1000,
				name: 'To Clear',
				ragNamespace: 'linear_algebra',
				courseId: 'math_linalg'
			};
			conversationsStore.conversations = [activeConv];
			conversationsStore.activeConversation = activeConv;

			const prefs = conversationsStore.preferences;
			expect(prefs.getRagNamespace()).toBe('linear_algebra');

			conversationsStore.clearActiveConversation();

			expect(conversationsStore.activeConversation).toBeNull();
			expect(conversationsStore.activeMessages).toEqual([]);
			expect(prefs.pendingRagNamespace).toBe('default');
			expect(prefs.pendingCourseId).toBeNull();
			expect(prefs.getRagNamespace()).toBe('default');
			expect(prefs.getCourseId()).toBeUndefined();
		});

		it('loadConversation() drops uncommitted draft pending picks', async () => {
			const prefs = conversationsStore.preferences;

			// Draft pick on new-chat screen
			await prefs.setRagNamespace('draft_uncommitted_kb', 'draft_course');
			expect(prefs.pendingRagNamespace).toBe('draft_uncommitted_kb');
			expect(prefs.pendingCourseId).toBe('draft_course');

			// User opens existing conversation
			const existingConv: DatabaseConversation = {
				currNode: null,
				id: 'conv-existing-db',
				lastModified: 2000,
				name: 'Existing DB Conv',
				ragNamespace: 'calculus',
				courseId: 'math_calc'
			};
			getConversationSpy.mockResolvedValue(existingConv);

			const success = await conversationsStore.loadConversation('conv-existing-db');
			expect(success).toBe(true);

			// Pending state MUST be reset to default / null
			expect(prefs.pendingRagNamespace).toBe('default');
			expect(prefs.pendingCourseId).toBeNull();

			// Active conversation resolves to calculus
			expect(prefs.getRagNamespace()).toBe('calculus');
			expect(prefs.getCourseId()).toBe('math_calc');

			// When active conversation is cleared, it does not restore draft_uncommitted_kb
			conversationsStore.clearActiveConversation();
			expect(prefs.getRagNamespace()).toBe('default');
		});
	});

	// =========================================================================
	// 5. Dual-Channel ChatService Passthrough Under High Concurrency
	// =========================================================================
	describe('5. Dual-Channel ChatService Passthrough Under High Concurrency', () => {
		let capturedRequests: Array<{ url: string; init: RequestInit; body: any }> = [];

		beforeEach(() => {
			capturedRequests = [];
			vi.stubGlobal(
				'fetch',
				vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
					const body = init?.body ? JSON.parse(String(init.body)) : {};
					capturedRequests.push({ url: String(url), init: init!, body });

					if (body.stream) {
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
							choices: [{ message: { role: 'assistant', content: 'OK' } }]
						}),
						{ status: 200, headers: { 'Content-Type': 'application/json' } }
					);
				})
			);
		});

		afterEach(() => {
			vi.unstubAllGlobals();
		});

		it('fires 25 concurrent requests with varying namespaces and validates header & body alignment', async () => {
			const configs = [
				{ id: 'c1', ns: 'calculus', course: 'math_calc', stream: true },
				{ id: 'c2', ns: 'physics', course: 'phys_gen', stream: false },
				{ id: 'c3', ns: 'linear_algebra', course: 'math_linalg', stream: true },
				{ id: 'c4', ns: 'default', course: undefined, stream: false },
				{ id: 'c5', ns: 'cs_network', course: 'cs_net', stream: true }
			];

			const promises = [];
			for (let i = 0; i < 25; i++) {
				const conf = configs[i % configs.length];
				const convId = `${conf.id}_req_${i}`;
				promises.push(
					ChatService.sendMessage(
						[{ role: MessageRole.USER, content: `Message ${i}` }],
						{
							stream: conf.stream,
							rag_namespace: conf.ns,
							course_id: conf.course,
							model: 'qwen2.5:7b'
						},
						convId
					)
				);
			}

			await Promise.all(promises);

			expect(capturedRequests.length).toBe(25);

			// Inspect every single captured request
			for (let i = 0; i < 25; i++) {
				const conf = configs[i % configs.length];
				const convId = `${conf.id}_req_${i}`;
				const req = capturedRequests[i];
				const headers = req.init.headers as Record<string, string>;

				// Header checks
				if (conf.stream) {
					expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toContain(convId);
				} else {
					expect(headers[HEADERS.X_CONVERSATION_ID_HEADER]).toBe(convId);
				}
				expect(headers[HEADERS.X_RAG_NAMESPACE_HEADER]).toBe(conf.ns);

				if (conf.course) {
					expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBe(conf.course);
				} else {
					expect(headers[HEADERS.X_COURSE_ID_HEADER]).toBeUndefined();
				}

				// Body checks: session_id must NEVER contain stream identity suffix
				expect(req.body.session_id).toBe(convId);
				expect(req.body.rag_namespace).toBe(conf.ns);

				if (conf.course) {
					expect(req.body.course_id).toBe(conf.course);
				} else {
					expect(req.body.course_id).toBeUndefined();
				}
			}
		});
	});
});
