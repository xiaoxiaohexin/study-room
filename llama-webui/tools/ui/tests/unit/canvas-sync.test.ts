import { describe, expect, it } from 'vitest';
import type { CustomCanvasNode } from '$lib/types/canvas';
import {
	backendKpToCanvasNode,
	mapBackendHistoryToRevisions,
	mapBackendProvenanceToAnchors,
	mapBackendRelationsToEdges,
	mergeBackendKnowledgePoints
} from '$lib/services/canvas-sync';

const baseKp = {
	id: 'kp_1',
	course_id: 'math_calc',
	name: '洛必达法则',
	summary: '未定式求极限',
	content: '个人笔记',
	raw_quote: '在满足 0/0 未定式条件下，分子分母分别求导',
	difficulty: 0.6,
	version: 2,
	provenance: {
		message_id: 'msg_1',
		turn_index: 3,
		doc_id: 'doc_abc',
		chunk_id: 'doc_abc_4',
		doc_title: '高等数学.pdf',
		page_start: 42,
		page_end: 43,
		quote_snippet: '在满足 0/0 未定式条件下',
		extra: {
			all_anchors: [
				{
					message_id: 'msg_2',
					turn_index: 5,
					doc_title: '高等数学.pdf',
					page_start: 88,
					quote_snippet: '第二个引用'
				}
			]
		}
	},
	history: [
		{ version: 1, timestamp: 1729400000, content: 'v1', raw_quote: 'q1', change_type: 'create' },
		{
			version: 2,
			timestamp: 1729500000,
			content: 'v2',
			raw_quote: 'q1',
			change_type: 'edit',
			summary: '更新笔记'
		}
	]
};

function knowledgeNode(overrides: Partial<CustomCanvasNode> = {}): CustomCanvasNode {
	return {
		id: 'kn-local',
		kind: 'knowledge',
		knowledgePointId: 'kp_1',
		title: '本地标题',
		content: '本地内容',
		color: 'emerald',
		x: 100,
		y: 200,
		width: 380,
		currentVersion: 2,
		courseId: 'math_calc',
		...overrides
	};
}

describe('canvas-sync history mapping', () => {
	it('maps backend history (seconds timestamps) to local revisions', () => {
		const revisions = mapBackendHistoryToRevisions(baseKp.history, '洛必达法则');
		expect(revisions).toHaveLength(2);
		expect(revisions[0].version).toBe(1);
		expect(revisions[1].changeType).toBe('edit');
		expect(revisions[1].summary).toBe('更新笔记');
		// 1729400000s -> ms
		expect(revisions[0].timestamp).toBe(1729400000000);
	});

	it('maps rollback change types to revert', () => {
		const revisions = mapBackendHistoryToRevisions(
			[{ version: 3, timestamp: 1729600000, change_type: 'rollback_to_v1', content: 'v1' }],
			'x'
		);
		expect(revisions[0].changeType).toBe('revert');
	});

	it('returns empty list for malformed history', () => {
		expect(mapBackendHistoryToRevisions(null)).toEqual([]);
		expect(mapBackendHistoryToRevisions('nope')).toEqual([]);
	});
});

describe('canvas-sync provenance mapping', () => {
	it('maps primary anchor and extra.all_anchors', () => {
		const anchors = mapBackendProvenanceToAnchors(baseKp.provenance);
		expect(anchors).toHaveLength(2);
		expect(anchors[0].messageId).toBe('msg_1');
		expect(anchors[0].docId).toBe('doc_abc');
		expect(anchors[0].chunkId).toBe('doc_abc_4');
		expect(anchors[0].pageStart).toBe(42);
		expect(anchors[1].pageStart).toBe(88);
	});
});

describe('canvas-sync knowledge node construction & merge', () => {
	it('builds a canvas node from a backend knowledge point', () => {
		const node = backendKpToCanvasNode(baseKp, 0, 'math_calc');
		expect(node.id).toBe('kn-kp_1');
		expect(node.knowledgePointId).toBe('kp_1');
		expect(node.courseId).toBe('math_calc');
		expect(node.currentVersion).toBe(2);
		expect(node.provenance).toHaveLength(2);
		expect(node.revisions).toHaveLength(2);
	});

	it('adds missing backend knowledge points without touching existing layout', () => {
		const local = knowledgeNode();
		const result = mergeBackendKnowledgePoints([local], [baseKp], 'math_calc');
		expect(result.nodes).toHaveLength(1);
		expect(result.added).toBe(0);
		expect(result.nodes[0].x).toBe(100);
	});

	it('adds a new card for a knowledge point missing locally', () => {
		const otherKp = { ...baseKp, id: 'kp_2', name: '拉格朗日中值定理', version: 1 };
		const result = mergeBackendKnowledgePoints([], [otherKp], 'math_calc');
		expect(result.added).toBe(1);
		expect(result.nodes[0].knowledgePointId).toBe('kp_2');
	});

	it('adopts newer backend version/content while keeping position', () => {
		const local = knowledgeNode({ currentVersion: 1, content: '旧内容', x: 777, y: 888 });
		const newerKp = { ...baseKp, version: 5, content: '后端新内容' };
		const result = mergeBackendKnowledgePoints([local], [newerKp], 'math_calc');
		expect(result.updated).toBe(1);
		expect(result.nodes[0].currentVersion).toBe(5);
		expect(result.nodes[0].content).toBe('后端新内容');
		expect(result.nodes[0].x).toBe(777);
	});
});

describe('canvas-sync relation mapping', () => {
	it('maps relations only when both endpoints have cards', () => {
		const nodes = [
			knowledgeNode({ id: 'node-a', knowledgePointId: 'kp_a' }),
			knowledgeNode({ id: 'node-b', knowledgePointId: 'kp_b' })
		];
		const edges = mapBackendRelationsToEdges(
			[
				{ id: 'rel_1', src_id: 'kp_a', rel_type: 'prerequisite', dst_id: 'kp_b' },
				{ id: 'rel_2', src_id: 'kp_a', rel_type: 'causal', dst_id: 'kp_missing' }
			],
			nodes
		);
		expect(edges).toHaveLength(1);
		expect(edges[0].from).toBe('node-a');
		expect(edges[0].to).toBe('node-b');
		expect(edges[0].relType).toBe('prerequisite');
	});
});
