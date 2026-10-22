import { describe, expect, it } from 'vitest';
import type {
	CustomCanvasNode,
	KnowledgeRevision,
	PersistedCanvasData,
	ProvenanceAnchor
} from '$lib/types/canvas';
import {
	extractProvenanceInfo,
	extractTitleCandidate,
	provenanceNavigation,
	sanitizeHtml
} from '$lib/services/provenance-navigation.service';

/**
 * Normalization function matching InfiniteCanvasWorkspace.svelte lines 851-895
 */
function normalizeCustomNode(rawNode: CustomCanvasNode): CustomCanvasNode {
	if (rawNode.kind === 'knowledge') {
		const provList: ProvenanceAnchor[] = Array.isArray(rawNode.provenance)
			? rawNode.provenance
			: rawNode.provenance
				? [rawNode.provenance]
				: [];

		const revList: KnowledgeRevision[] =
			Array.isArray(rawNode.revisions) && rawNode.revisions.length > 0
				? rawNode.revisions
				: [
						{
							version: 1,
							timestamp: Date.now(),
							editor: 'user',
							changeType: 'initial',
							summary: '初始版本',
							content: rawNode.content || '',
							title: rawNode.title || '',
							quoteSnippet: provList[0]?.quoteSnippet || rawNode.rawQuote || '',
							rawQuote: rawNode.rawQuote || provList[0]?.quoteSnippet || '',
							difficulty: rawNode.difficulty ?? 0.5
						}
					];

		return {
			...rawNode,
			kind: 'knowledge',
			color: rawNode.color || 'emerald',
			width: typeof rawNode.width === 'number' ? rawNode.width : 380,
			difficulty: typeof rawNode.difficulty === 'number' ? rawNode.difficulty : 0.5,
			mastery: typeof rawNode.mastery === 'number' ? rawNode.mastery : 0.3,
			currentVersion:
				typeof rawNode.currentVersion === 'number'
					? rawNode.currentVersion
					: revList[revList.length - 1]?.version ?? 1,
			provenance: provList,
			revisions: revList,
			historyExpanded: false,
			activeProvenanceIndex: 0
		};
	}
	return rawNode;
}

/**
 * Simulates user blur action on Zone 2 (InfiniteCanvasWorkspace.svelte lines 1226-1252)
 */
function handleKnowledgeNodeBlur(
	node: CustomCanvasNode,
	newTitle?: string,
	newContent?: string
): { revisionCreated: boolean; node: CustomCanvasNode } {
	if (newTitle !== undefined) node.title = newTitle;
	if (newContent !== undefined) node.content = newContent;

	const lastRev = node.revisions?.[node.revisions.length - 1];
	if (lastRev && lastRev.content === node.content && lastRev.title === node.title) {
		return { revisionCreated: false, node };
	}

	const nextVersion = (node.currentVersion ?? 1) + 1;
	const revision: KnowledgeRevision = {
		version: nextVersion,
		timestamp: Date.now(),
		editor: 'user',
		changeType: 'edit_content',
		summary: '更新概念笔记与思考',
		content: node.content,
		title: node.title,
		difficulty: node.difficulty ?? 0.5
	};

	node.revisions = [...(node.revisions ?? []), revision];
	node.currentVersion = nextVersion;

	return { revisionCreated: true, node };
}

/**
 * Simulates user rollback action from history drawer (InfiniteCanvasWorkspace.svelte lines 1254-1291)
 */
function handleKnowledgeRollback(
	node: CustomCanvasNode,
	targetRev: KnowledgeRevision
): { rollbackRev: KnowledgeRevision; node: CustomCanvasNode } {
	node.title = targetRev.title ?? node.title;
	node.content = targetRev.content;
	if (targetRev.difficulty !== undefined) {
		node.difficulty = targetRev.difficulty;
	}

	const nextVersion = (node.currentVersion ?? 1) + 1;
	const rollbackRev: KnowledgeRevision = {
		version: nextVersion,
		timestamp: Date.now(),
		editor: 'user',
		changeType: 'revert',
		summary: `回退至版本 v${targetRev.version}`,
		content: node.content,
		title: node.title,
		difficulty: node.difficulty ?? 0.5
	};

	node.revisions = [...(node.revisions ?? []), rollbackRev];
	node.currentVersion = nextVersion;
	node.historyExpanded = false;

	return { rollbackRev, node };
}

/**
 * Simulates append quote action (InfiniteCanvasWorkspace.svelte lines 1175-1224)
 */
function handleAppendKnowledgeNode(
	node: CustomCanvasNode,
	detail: {
		quoteSnippet: string;
		messageId?: string;
		turnIndex?: number;
		docTitle?: string;
		pageStart?: number;
		pageEnd?: number;
	}
): { anchor: ProvenanceAnchor; revision: KnowledgeRevision; node: CustomCanvasNode } {
	const anchor: ProvenanceAnchor = {
		id: `prov-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
		messageId: detail.messageId,
		turnIndex: detail.turnIndex,
		docTitle: detail.docTitle,
		pageStart: detail.pageStart,
		pageEnd: detail.pageEnd,
		quoteSnippet: detail.quoteSnippet,
		createdAt: Date.now()
	};

	const nextVersion = (node.currentVersion ?? 1) + 1;
	const revision: KnowledgeRevision = {
		version: nextVersion,
		timestamp: Date.now(),
		editor: 'extract',
		changeType: 'append_quote',
		summary: detail.turnIndex ? `追加来自对话 #${detail.turnIndex} 的引用` : '追加引用片段',
		content: node.content,
		title: node.title,
		quoteSnippet: detail.quoteSnippet,
		rawQuote: detail.quoteSnippet,
		difficulty: node.difficulty ?? 0.5
	};

	node.rawQuote = node.rawQuote
		? `${node.rawQuote}\n\n---\n\n${detail.quoteSnippet}`
		: detail.quoteSnippet;
	node.provenance = [...(node.provenance ?? []), anchor];
	node.revisions = [...(node.revisions ?? []), revision];
	node.currentVersion = nextVersion;
	node.activeProvenanceIndex = (node.provenance.length || 1) - 1;

	return { anchor, revision, node };
}

/**
 * Navigation flight math implementation matching InfiniteCanvasWorkspace.svelte lines 1052-1064
 */
function computeFlightCoordinates(
	targetX: number,
	targetY: number,
	targetWidth: number,
	targetHeight: number,
	zoom: number,
	vpWidth: number,
	vpHeight: number,
	autoReadabilityClamp = true
) {
	let effectiveZoom = zoom;
	if (autoReadabilityClamp && effectiveZoom < 0.8) {
		effectiveZoom = 0.95;
	}

	const centerX = targetX + targetWidth / 2;
	const centerY = targetY + targetHeight / 2;

	const panX = Math.round(vpWidth / 2 - centerX * effectiveZoom);
	const panY = Math.round(vpHeight / 2 - centerY * effectiveZoom);

	return { panX, panY, effectiveZoom, centerX, centerY };
}

describe('M3 Adversarial Challenge 1: Rapid Sequential Multi-Version Edits (v1 -> v2 -> v3 -> v4 -> v5)', () => {
	it('increments version monotonically and preserves each snapshot during rapid sequential edits', () => {
		const initialAnchor: ProvenanceAnchor = {
			id: 'prov_v1',
			messageId: 'msg_init',
			turnIndex: 1,
			quoteSnippet: '基底变换矩阵具有满秩性质',
			createdAt: 1000
		};

		const initialRevision: KnowledgeRevision = {
			version: 1,
			timestamp: 1000,
			editor: 'extract',
			changeType: 'initial',
			summary: '初始提炼',
			content: '基底变换笔记 v1',
			title: '基底变换 v1',
			rawQuote: initialAnchor.quoteSnippet,
			quoteSnippet: initialAnchor.quoteSnippet,
			difficulty: 0.5
		};

		const card: CustomCanvasNode = {
			id: 'kn_seq_test',
			kind: 'knowledge',
			title: '基底变换 v1',
			content: '基底变换笔记 v1',
			rawQuote: initialAnchor.quoteSnippet,
			color: 'emerald',
			width: 380,
			x: 100,
			y: 100,
			currentVersion: 1,
			provenance: [initialAnchor],
			revisions: [initialRevision]
		};

		// Edit 1: v1 -> v2
		const e1 = handleKnowledgeNodeBlur(card, '基底变换 v2', '引入渡矩阵 P: [α1...αn] = [β1...βn]P');
		expect(e1.revisionCreated).toBe(true);
		expect(card.currentVersion).toBe(2);
		expect(card.revisions).toHaveLength(2);
		expect(card.revisions![1].version).toBe(2);
		expect(card.revisions![1].title).toBe('基底变换 v2');

		// Edit 2: v2 -> v3
		const e2 = handleKnowledgeNodeBlur(card, '基底变换与过渡矩阵 v3', '坐标变换公式: X = P Y');
		expect(e2.revisionCreated).toBe(true);
		expect(card.currentVersion).toBe(3);
		expect(card.revisions).toHaveLength(3);
		expect(card.revisions![2].version).toBe(3);
		expect(card.revisions![2].content).toBe('坐标变换公式: X = P Y');

		// Edit 3: v3 -> v4
		const e3 = handleKnowledgeNodeBlur(card, '基底变换与过渡矩阵 v4', '补充几何直观：旋转与剪切对应的过渡矩阵行列表达');
		expect(e3.revisionCreated).toBe(true);
		expect(card.currentVersion).toBe(4);
		expect(card.revisions).toHaveLength(4);
		expect(card.revisions![3].version).toBe(4);

		// Edit 4: v4 -> v5
		const e4 = handleKnowledgeNodeBlur(card, '线性空间基底变换与坐标定理 v5', '最终定理总结：dim(V)=n，任何非退化方阵均可视为某两组基底的过渡矩阵');
		expect(e4.revisionCreated).toBe(true);
		expect(card.currentVersion).toBe(5);
		expect(card.revisions).toHaveLength(5);
		expect(card.revisions![4].version).toBe(5);

		// Verify version list integrity
		const versions = card.revisions!.map((r) => r.version);
		expect(versions).toEqual([1, 2, 3, 4, 5]);

		// Verify historical snapshots are immutable (no overwrite of v1..v4)
		expect(card.revisions![0].content).toBe('基底变换笔记 v1');
		expect(card.revisions![1].content).toBe('引入渡矩阵 P: [α1...αn] = [β1...βn]P');
		expect(card.revisions![2].content).toBe('坐标变换公式: X = P Y');
		expect(card.revisions![3].content).toContain('旋转与剪切');
		expect(card.revisions![4].content).toContain('最终定理总结');

		// Verify no-op blur: blurring without modifications does not create duplicate revision
		const noop = handleKnowledgeNodeBlur(card);
		expect(noop.revisionCreated).toBe(false);
		expect(card.currentVersion).toBe(5);
		expect(card.revisions).toHaveLength(5);
	});

	it('stress-tests 50 sequential rapid blur edits without state loss or version collisions', () => {
		const card: CustomCanvasNode = normalizeCustomNode({
			id: 'kn_fuzz_50',
			kind: 'knowledge',
			title: 'Init Title',
			content: 'Init Content',
			color: 'emerald',
			width: 380,
			x: 0,
			y: 0
		});

		expect(card.currentVersion).toBe(1);
		expect(card.revisions).toHaveLength(1);

		for (let i = 2; i <= 50; i++) {
			const res = handleKnowledgeNodeBlur(
				card,
				`Title Iteration ${i} - $\\lambda_${i}$`,
				`Content at step ${i}: $\\sum_{j=1}^{${i}} j^2 = \\frac{${i}(${i}+1)(2\\cdot${i}+1)}{6}$`
			);
			expect(res.revisionCreated).toBe(true);
			expect(card.currentVersion).toBe(i);
			expect(card.revisions).toHaveLength(i);
			expect(card.revisions![i - 1].version).toBe(i);
			expect(card.revisions![i - 1].changeType).toBe('edit_content');
			expect(card.revisions![i - 1].title).toContain(`Iteration ${i}`);
		}

		// Ensure all 50 versions are strictly sequential [1, 2, ..., 50]
		const allVersions = card.revisions!.map((r) => r.version);
		expect(allVersions).toEqual(Array.from({ length: 50 }, (_, idx) => idx + 1));
	});
});

describe('M3 Adversarial Challenge 2: Rollback to v1 from v5 (Creating v6 with Prior History Preserved)', () => {
	it('restores v1 content into newly created v6 and preserves all versions 1 through 5 in history', () => {
		const card: CustomCanvasNode = normalizeCustomNode({
			id: 'kn_rollback_test',
			kind: 'knowledge',
			title: 'Version 1 Golden Title',
			content: 'Version 1 Golden Excerpt & Content',
			rawQuote: 'Golden quote excerpt',
			difficulty: 0.3,
			color: 'emerald',
			width: 380,
			x: 0,
			y: 0
		});

		// Build versions 2, 3, 4, 5
		handleKnowledgeNodeBlur(card, 'V2 Modified Title', 'V2 Modified Content');
		handleKnowledgeNodeBlur(card, 'V3 Flawed Title', 'V3 Flawed Content');
		handleKnowledgeNodeBlur(card, 'V4 Experimental Title', 'V4 Experimental Content');
		handleKnowledgeNodeBlur(card, 'V5 Bad Title', 'V5 Bad Content with Syntax Errors');

		expect(card.currentVersion).toBe(5);
		expect(card.revisions).toHaveLength(5);

		// Locate v1 snapshot
		const v1Snapshot = card.revisions!.find((r) => r.version === 1);
		expect(v1Snapshot).toBeDefined();
		expect(v1Snapshot!.title).toBe('Version 1 Golden Title');
		expect(v1Snapshot!.content).toBe('Version 1 Golden Excerpt & Content');

		// Execute rollback to v1
		card.historyExpanded = true;
		const { rollbackRev } = handleKnowledgeRollback(card, v1Snapshot!);

		// Invariant 1: New version v6 is created
		expect(card.currentVersion).toBe(6);
		expect(rollbackRev.version).toBe(6);
		expect(rollbackRev.changeType).toBe('revert');
		expect(rollbackRev.summary).toBe('回退至版本 v1');

		// Invariant 2: Active card title and content match v1 exactly
		expect(card.title).toBe('Version 1 Golden Title');
		expect(card.content).toBe('Version 1 Golden Excerpt & Content');
		expect(card.difficulty).toBe(0.3);

		// Invariant 3: History drawer collapses
		expect(card.historyExpanded).toBe(false);

		// Invariant 4: All prior versions 1..5 remain in history alongside v6
		expect(card.revisions).toHaveLength(6);
		const versions = card.revisions!.map((r) => r.version);
		expect(versions).toEqual([1, 2, 3, 4, 5, 6]);

		// Verify contents of every version in sequence
		expect(card.revisions![0].title).toBe('Version 1 Golden Title');
		expect(card.revisions![1].title).toBe('V2 Modified Title');
		expect(card.revisions![2].title).toBe('V3 Flawed Title');
		expect(card.revisions![3].title).toBe('V4 Experimental Title');
		expect(card.revisions![4].title).toBe('V5 Bad Title');
		expect(card.revisions![5].title).toBe('Version 1 Golden Title');
		expect(card.revisions![5].changeType).toBe('revert');
	});
});

describe('M3 Adversarial Challenge 3: Rollback of a Rolled-Back Card (v6 -> Rollback to v3 Creating v7)', () => {
	it('handles chained rollbacks gracefully creating v7 from v3 without history corruption', () => {
		const card: CustomCanvasNode = normalizeCustomNode({
			id: 'kn_chained_rollback',
			kind: 'knowledge',
			title: 'Original v1',
			content: 'Original content v1',
			color: 'emerald',
			width: 380,
			x: 0,
			y: 0
		});

		// Build to v5
		handleKnowledgeNodeBlur(card, 'Title v2', 'Content v2');
		handleKnowledgeNodeBlur(card, 'Canonical Title v3', 'Canonical Good Content v3');
		handleKnowledgeNodeBlur(card, 'Title v4', 'Content v4');
		handleKnowledgeNodeBlur(card, 'Title v5', 'Content v5');

		// First rollback: v5 -> rollback to v1 (creates v6)
		const v1Snapshot = card.revisions![0];
		handleKnowledgeRollback(card, v1Snapshot);
		expect(card.currentVersion).toBe(6);
		expect(card.content).toBe('Original content v1');

		// Second rollback: v6 (which is a revert to v1) -> rollback to v3 (creates v7)
		const v3Snapshot = card.revisions!.find((r) => r.version === 3);
		expect(v3Snapshot).toBeDefined();
		expect(v3Snapshot!.title).toBe('Canonical Title v3');
		expect(v3Snapshot!.content).toBe('Canonical Good Content v3');

		const { rollbackRev: rev7 } = handleKnowledgeRollback(card, v3Snapshot!);

		expect(card.currentVersion).toBe(7);
		expect(rev7.version).toBe(7);
		expect(rev7.changeType).toBe('revert');
		expect(rev7.summary).toBe('回退至版本 v3');
		expect(card.title).toBe('Canonical Title v3');
		expect(card.content).toBe('Canonical Good Content v3');

		// Verify 7 full revisions present in order
		expect(card.revisions).toHaveLength(7);
		expect(card.revisions!.map((r) => r.version)).toEqual([1, 2, 3, 4, 5, 6, 7]);
		expect(card.revisions![5].summary).toBe('回退至版本 v1');
		expect(card.revisions![6].summary).toBe('回退至版本 v3');

		// Third rollback: rollback to v6 (the revert of v1) -> creates v8
		const v6Snapshot = card.revisions!.find((r) => r.version === 6)!;
		const { rollbackRev: rev8 } = handleKnowledgeRollback(card, v6Snapshot);
		expect(card.currentVersion).toBe(8);
		expect(rev8.version).toBe(8);
		expect(card.content).toBe('Original content v1');
		expect(card.revisions).toHaveLength(8);
		expect(card.revisions!.map((r) => r.version)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
	});

	it('stress-tests 15 recursive ping-pong rollbacks across past revisions', () => {
		const card: CustomCanvasNode = normalizeCustomNode({
			id: 'kn_pingpong_stress',
			kind: 'knowledge',
			title: 'Init',
			content: 'Init Content',
			color: 'emerald',
			width: 380,
			x: 0,
			y: 0
		});

		// Build baseline v1..v5
		for (let i = 2; i <= 5; i++) {
			handleKnowledgeNodeBlur(card, `Title v${i}`, `Content v${i}`);
		}
		expect(card.currentVersion).toBe(5);

		// Ping-pong between v2, v3, v4, v1 across 15 additional versions (v6 through v20)
		const targetVersions = [2, 4, 1, 3, 5, 2, 1, 4, 3, 2, 5, 1, 3, 2, 4];
		for (let i = 0; i < targetVersions.length; i++) {
			const targetNum = targetVersions[i];
			const targetRev = card.revisions!.find((r) => r.version === targetNum)!;
			const expectedNextVersion = 6 + i;

			handleKnowledgeRollback(card, targetRev);

			expect(card.currentVersion).toBe(expectedNextVersion);
			expect(card.revisions).toHaveLength(expectedNextVersion);
			expect(card.title).toBe(targetRev.title);
			expect(card.content).toBe(targetRev.content);
			expect(card.revisions![card.revisions!.length - 1].changeType).toBe('revert');
		}

		expect(card.currentVersion).toBe(20);
		expect(card.revisions).toHaveLength(20);
		expect(card.revisions!.map((r) => r.version)).toEqual(
			Array.from({ length: 20 }, (_, idx) => idx + 1)
		);
	});
});

describe('M3 Adversarial Challenge 4: Multi-Quote Appending Stress (10+ Distinct Provenance Anchors)', () => {
	it('appends 10 distinct quotes to a knowledge card and verifies all 10 ProvenanceAnchors persist without memory leak or state loss', () => {
		const card: CustomCanvasNode = normalizeCustomNode({
			id: 'kn_multi_quote_card',
			kind: 'knowledge',
			title: '深度学习核心定理汇编',
			content: '汇总多轮对话与教科书定理证明',
			rawQuote: '初始定理 0：万能逼近定理',
			color: 'emerald',
			width: 380,
			x: 200,
			y: 200,
			provenance: [
				{
					id: 'anchor_init_0',
					messageId: 'msg_0',
					turnIndex: 1,
					docTitle: 'DeepLearning.pdf',
					pageStart: 10,
					pageEnd: 11,
					quoteSnippet: '初始定理 0：万能逼近定理',
					createdAt: 1000
				}
			]
		});

		expect(card.provenance).toHaveLength(1);
		expect(card.currentVersion).toBe(1);

		// Append 10 quotes consecutively from different turns and documents
		const appendedQuotes: Array<{
			turnIndex: number;
			messageId: string;
			docTitle: string;
			pageStart: number;
			pageEnd: number;
			quote: string;
		}> = [
			{
				turnIndex: 2,
				messageId: 'msg_2',
				docTitle: 'LinearAlgebra.pdf',
				pageStart: 42,
				pageEnd: 43,
				quote: '定理 1：奇异值分解 (SVD) 存在性证明与几何意义'
			},
			{
				turnIndex: 3,
				messageId: 'msg_3',
				docTitle: 'ConvexOptimization.pdf',
				pageStart: 77,
				pageEnd: 79,
				quote: '定理 2：KKT 条件在凸优化中的充要性'
			},
			{
				turnIndex: 4,
				messageId: 'msg_4',
				docTitle: 'InformationTheory.pdf',
				pageStart: 104,
				pageEnd: 105,
				quote: '定理 3：数据处理不等式与互信息单调性'
			},
			{
				turnIndex: 5,
				messageId: 'msg_5',
				docTitle: 'ProbabilisticGraphicalModels.pdf',
				pageStart: 210,
				pageEnd: 212,
				quote: '定理 4：Hammersley-Clifford 定理与吉布斯分布'
			},
			{
				turnIndex: 6,
				messageId: 'msg_6',
				docTitle: 'StatisticalLearning.pdf',
				pageStart: 315,
				pageEnd: 318,
				quote: '定理 5：VC 维与泛化误差界限的非渐进分析'
			},
			{
				turnIndex: 7,
				messageId: 'msg_7',
				docTitle: 'NeuralTangentKernel.pdf',
				pageStart: 55,
				pageEnd: 56,
				quote: '定理 6：无限宽神经网络的神经切空间核收敛性'
			},
			{
				turnIndex: 8,
				messageId: 'msg_8',
				docTitle: 'DifferentialPrivacy.pdf',
				pageStart: 88,
				pageEnd: 90,
				quote: '定理 7：高斯机制的 (ε, δ)-差分隐私保证'
			},
			{
				turnIndex: 9,
				messageId: 'msg_9',
				docTitle: 'ReinforcementLearning.pdf',
				pageStart: 160,
				pageEnd: 162,
				quote: '定理 8：Bellman 期望方程的最优性与不动点压缩映射'
			},
			{
				turnIndex: 10,
				messageId: 'msg_10',
				docTitle: 'OptimalTransport.pdf',
				pageStart: 25,
				pageEnd: 28,
				quote: '定理 9：Kantorovich-Rubinstein 对偶定理'
			},
			{
				turnIndex: 11,
				messageId: 'msg_11',
				docTitle: 'StochasticCalculus.pdf',
				pageStart: 92,
				pageEnd: 95,
				quote: '定理 10：伊藤引理与连续鞅扩散表示'
			}
		];

		for (let i = 0; i < appendedQuotes.length; i++) {
			const item = appendedQuotes[i];
			const { anchor, revision } = handleAppendKnowledgeNode(card, {
				quoteSnippet: item.quote,
				messageId: item.messageId,
				turnIndex: item.turnIndex,
				docTitle: item.docTitle,
				pageStart: item.pageStart,
				pageEnd: item.pageEnd
			});

			expect(anchor.messageId).toBe(item.messageId);
			expect(anchor.turnIndex).toBe(item.turnIndex);
			expect(anchor.docTitle).toBe(item.docTitle);
			expect(anchor.quoteSnippet).toBe(item.quote);

			expect(revision.changeType).toBe('append_quote');
			expect(revision.version).toBe(i + 2);
		}

		// Total anchors: 1 initial + 10 appended = 11 anchors
		expect(card.provenance).toHaveLength(11);
		expect(card.currentVersion).toBe(11);
		expect(card.revisions).toHaveLength(11);

		// Active tab index must point to latest appended anchor
		expect(card.activeProvenanceIndex).toBe(10);

		// Verify all 11 anchors are distinct and preserve their metadata
		const anchorIds = new Set(card.provenance!.map((p) => p.id));
		expect(anchorIds.size).toBe(11);

		// Verify aggregated rawQuote contains all 10 appended quotes separated by '---'
		for (const item of appendedQuotes) {
			expect(card.rawQuote).toContain(item.quote);
		}
		expect(card.rawQuote).toContain('---');

		// Test bi-directional integration with provenanceNavigation
		provenanceNavigation.setKnowledgeNodes([card]);

		// Verify that searching by messageId finds this node for ALL 10 appended messages
		for (const item of appendedQuotes) {
			const found = provenanceNavigation.getKnowledgeNodesForMessage(item.messageId);
			expect(found).toHaveLength(1);
			expect(found[0].id).toBe('kn_multi_quote_card');
			expect(found[0].title).toBe('深度学习核心定理汇编');
		}

		// Also verify initial message
		const initFound = provenanceNavigation.getKnowledgeNodesForMessage('msg_0');
		expect(initFound).toHaveLength(1);
		expect(initFound[0].id).toBe('kn_multi_quote_card');
	});

	it('stress-tests appending 50 quotes and ensures clean memory and lookup performance', () => {
		const card: CustomCanvasNode = normalizeCustomNode({
			id: 'kn_stress_50_quotes',
			kind: 'knowledge',
			title: 'Big Knowledge Depot',
			content: 'Master Notes',
			color: 'emerald',
			width: 380,
			x: 0,
			y: 0
		});

		for (let i = 1; i <= 50; i++) {
			handleAppendKnowledgeNode(card, {
				quoteSnippet: `High dimensional data quote #${i}: \\mathcal{N}(\\mu, \\Sigma_${i})`,
				messageId: `msg_stress_${i}`,
				turnIndex: i,
				docTitle: `Volume_${(i % 5) + 1}.pdf`,
				pageStart: i * 2,
				pageEnd: i * 2 + 1
			});
		}

		expect(card.provenance!.length).toBeGreaterThanOrEqual(50);
		expect(card.currentVersion).toBe(51);

		provenanceNavigation.setKnowledgeNodes([card]);

		// Lookup all 50 messages
		for (let i = 1; i <= 50; i++) {
			const res = provenanceNavigation.getKnowledgeNodesForMessage(`msg_stress_${i}`);
			expect(res).toHaveLength(1);
			expect(res[0].id).toBe('kn_stress_50_quotes');
		}
	});
});

describe('M3 Adversarial Challenge 5: LocalStorage Serialization Roundtrip & Quota Resilience', () => {
	it('roundtrips large knowledge cards with revision snapshots through JSON serialization without data loss', () => {
		// Construct an adversarial knowledge card with complex unicode, LaTeX, multiline text, and quotes
		const complexRevisions: KnowledgeRevision[] = [];
		for (let i = 1; i <= 25; i++) {
			complexRevisions.push({
				version: i,
				timestamp: 1700000000000 + i * 60000,
				editor: i % 2 === 0 ? 'user' : 'extract',
				changeType: i === 1 ? 'initial' : i % 3 === 0 ? 'append_quote' : 'edit_content',
				summary: `版本 ${i} 快照：包含公式与特殊字符 \`code\` & "quotes"`,
				title: `概念：高斯积分与傅里叶变换 [第 ${i} 次修订]`,
				content: `$$ \\int_{-\\infty}^{+\\infty} e^{-a x^2} dx = \\sqrt{\\frac{\\pi}{a}} $$\n\n> 思考笔记 #${i}：\n- 变量替换 $u = \\sqrt{a}x$\n- 二重极坐标变换 $r dr d\\theta$\n- 特殊字符：<script>alert("safe")</script> & 符号 ⚡🧠`,
				quoteSnippet: `原文引文片段 #${i}：$\\mathcal{F}\\{e^{-t^2}\\} = \\sqrt{\\pi}e^{-\\omega^2/4}$`,
				difficulty: 0.8
			});
		}

		const complexAnchors: ProvenanceAnchor[] = [];
		for (let i = 1; i <= 15; i++) {
			complexAnchors.push({
				id: `anchor_complex_${i}`,
				messageId: `msg_fourier_${i}`,
				turnIndex: i,
				courseId: 'math_advanced',
				docTitle: `数学物理方程-第${i}卷.pdf`,
				pageStart: 100 + i * 5,
				pageEnd: 102 + i * 5,
				quoteSnippet: `“傅里叶变换的酉算子性质保证了能量守恒 (Parseval 定理)”——第 ${i} 节`,
				createdAt: 1700000000000 + i * 1000,
				extra: { confidence: 0.98, tags: ['fourier', 'calculus'] }
			});
		}

		const largeCard: CustomCanvasNode = {
			id: 'kn_large_persisted_card',
			kind: 'knowledge',
			title: '概念：高斯积分与傅里叶变换 [第 25 次修订]',
			content: complexRevisions[24].content,
			rawQuote: complexAnchors.map((a) => a.quoteSnippet).join('\n\n---\n\n'),
			color: 'emerald',
			width: 420,
			x: 650,
			y: 420,
			difficulty: 0.8,
			mastery: 0.75,
			currentVersion: 25,
			provenance: complexAnchors,
			revisions: complexRevisions,
			historyExpanded: false,
			activeProvenanceIndex: 14
		};

		const canvasPayload: PersistedCanvasData = {
			panX: -340,
			panY: 180,
			zoom: 1.15,
			nodePositions: {
				kn_large_persisted_card: { x: 650, y: 420 }
			},
			customNodes: [largeCard],
			customEdges: [
				{
					id: 'edge_rel_1',
					from: 'turn-1',
					to: 'kn_large_persisted_card',
					label: '知识提炼',
					relType: 'prerequisite'
				}
			],
			turnAssociations: {}
		};

		// 1. Serialization
		const serialized = JSON.stringify(canvasPayload);
		expect(serialized.length).toBeGreaterThan(5000);

		// 2. Deserialization
		const deserialized = JSON.parse(serialized) as PersistedCanvasData;

		// 3. Normalization through normalizeCustomNode
		expect(deserialized.customNodes).toHaveLength(1);
		const restoredCard = normalizeCustomNode(deserialized.customNodes[0]);

		// Verify Card Top-Level Fields
		expect(restoredCard.id).toBe('kn_large_persisted_card');
		expect(restoredCard.kind).toBe('knowledge');
		expect(restoredCard.currentVersion).toBe(25);
		expect(restoredCard.difficulty).toBe(0.8);
		expect(restoredCard.mastery).toBe(0.75);
		expect(restoredCard.width).toBe(420);
		expect(restoredCard.x).toBe(650);
		expect(restoredCard.y).toBe(420);

		// Verify Content & Title with Unicode, Math and Quotes
		expect(restoredCard.title).toBe(largeCard.title);
		expect(restoredCard.content).toBe(largeCard.content);
		expect(restoredCard.content).toContain('\\int_{-\\infty}^{+\\infty}');
		expect(restoredCard.content).toContain('⚡🧠');
		expect(restoredCard.content).toContain('<script>alert("safe")</script>');

		// Verify 25 Revisions Intact
		expect(restoredCard.revisions).toHaveLength(25);
		for (let i = 0; i < 25; i++) {
			expect(restoredCard.revisions![i].version).toBe(i + 1);
			expect(restoredCard.revisions![i].content).toBe(complexRevisions[i].content);
			expect(restoredCard.revisions![i].summary).toBe(complexRevisions[i].summary);
			expect(restoredCard.revisions![i].timestamp).toBe(complexRevisions[i].timestamp);
		}

		// Verify 15 Provenance Anchors Intact
		expect(restoredCard.provenance).toHaveLength(15);
		for (let i = 0; i < 15; i++) {
			expect(restoredCard.provenance![i].id).toBe(`anchor_complex_${i + 1}`);
			expect(restoredCard.provenance![i].messageId).toBe(`msg_fourier_${i + 1}`);
			expect(restoredCard.provenance![i].pageStart).toBe(100 + (i + 1) * 5);
			expect(restoredCard.provenance![i].pageEnd).toBe(102 + (i + 1) * 5);
			expect(restoredCard.provenance![i].quoteSnippet).toContain(`第 ${i + 1} 节`);
		}

		// Verify Viewport Positions & Zoom
		expect(deserialized.panX).toBe(-340);
		expect(deserialized.panY).toBe(180);
		expect(deserialized.zoom).toBe(1.15);
		expect(deserialized.customEdges[0].relType).toBe('prerequisite');
	});

	it('normalizes legacy or incomplete serialized data without crashing', () => {
		// Legacy node missing revisions, provenance as bare object, missing color and currentVersion
		const legacyCard: any = {
			id: 'kn_legacy_001',
			kind: 'knowledge',
			title: 'Old Card',
			content: 'Old Content',
			provenance: {
				id: 'prov_single',
				messageId: 'msg_old',
				quoteSnippet: 'Old Quote',
				createdAt: 5000
			}
		};

		const normalized = normalizeCustomNode(legacyCard as CustomCanvasNode);

		expect(normalized.kind).toBe('knowledge');
		expect(normalized.color).toBe('emerald');
		expect(normalized.width).toBe(380);
		expect(normalized.difficulty).toBe(0.5);
		expect(normalized.mastery).toBe(0.3);
		expect(normalized.provenance).toHaveLength(1);
		expect(normalized.provenance![0].id).toBe('prov_single');
		expect(normalized.revisions).toHaveLength(1);
		expect(normalized.revisions![0].version).toBe(1);
		expect(normalized.revisions![0].changeType).toBe('initial');
		expect(normalized.revisions![0].content).toBe('Old Content');
		expect(normalized.currentVersion).toBe(1);
	});

	it('tests localStorage quota fallback by stripping non-essential images while preserving knowledge metadata and revisions', () => {
		const heavyCard: CustomCanvasNode = {
			id: 'kn_heavy',
			kind: 'knowledge',
			title: 'Heavy Card with Big Base64 Images',
			content: 'Critical Revision Notes',
			color: 'emerald',
			width: 380,
			x: 100,
			y: 100,
			images: [
				{ id: 'img_1', name: 'large.png', url: 'data:image/png;base64,' + 'A'.repeat(50000) }
			],
			currentVersion: 3,
			revisions: [
				{
					version: 1,
					timestamp: 1000,
					editor: 'extract',
					changeType: 'initial',
					summary: 'Init',
					content: 'v1'
				},
				{
					version: 2,
					timestamp: 2000,
					editor: 'user',
					changeType: 'edit_content',
					summary: 'v2',
					content: 'v2'
				},
				{
					version: 3,
					timestamp: 3000,
					editor: 'user',
					changeType: 'edit_content',
					summary: 'v3',
					content: 'Critical Revision Notes'
				}
			],
			provenance: [
				{
					id: 'p_1',
					messageId: 'm_1',
					quoteSnippet: 'Important quote',
					createdAt: 1000
				}
			]
		};

		// Simulate quota error handler in InfiniteCanvasWorkspace.svelte lines 985-1011
		const slimPayload: PersistedCanvasData = {
			customEdges: [],
			customNodes: [heavyCard].map((n) => ({ ...n, images: undefined })),
			nodePositions: {},
			panX: 50,
			panY: 50,
			turnAssociations: {},
			zoom: 1
		};

		const serializedSlim = JSON.stringify(slimPayload);
		expect(serializedSlim).not.toContain('AAAAA');
		expect(serializedSlim).toContain('Critical Revision Notes');
		expect(serializedSlim).toContain('Important quote');

		const parsedSlim = JSON.parse(serializedSlim) as PersistedCanvasData;
		const recovered = normalizeCustomNode(parsedSlim.customNodes[0]);

		expect(recovered.images).toBeUndefined();
		expect(recovered.revisions).toHaveLength(3);
		expect(recovered.currentVersion).toBe(3);
		expect(recovered.provenance).toHaveLength(1);
	});
});

describe('M3 Adversarial Challenge 6: Navigation Flight Coordinates Math (0.2x to 3.0x Zoom Centering)', () => {
	it('mathematically centers the target card center exactly at viewport center across all zoom levels', () => {
		// World coordinates of card
		const testCases = [
			{ targetX: 0, targetY: 0, width: 380, height: 260 },
			{ targetX: 1200, targetY: 850, width: 380, height: 260 },
			{ targetX: -800, targetY: -500, width: 420, height: 320 },
			{ targetX: 3500, targetY: 2800, width: 380, height: 260 }
		];

		const viewports = [
			{ vpWidth: 1920, vpHeight: 1080 }, // 1080p Desktop
			{ vpWidth: 1366, vpHeight: 768 },  // Laptop
			{ vpWidth: 2560, vpHeight: 1440 }, // 2K
			{ vpWidth: 3840, vpHeight: 2160 }, // 4K
			{ vpWidth: 800, vpHeight: 600 },   // Small window
			{ vpWidth: 390, vpHeight: 844 }    // Mobile
		];

		// Raw zoom levels from 0.2 to 3.0 (without auto-clamp to test pure mathematical centering)
		const zoomLevels = [0.2, 0.35, 0.5, 0.75, 0.8, 0.95, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0];

		for (const card of testCases) {
			for (const vp of viewports) {
				for (const z of zoomLevels) {
					const { panX, panY, centerX, centerY } = computeFlightCoordinates(
						card.targetX,
						card.targetY,
						card.width,
						card.height,
						z,
						vp.vpWidth,
						vp.vpHeight,
						/* autoReadabilityClamp= */ false
					);

					// Apply CSS transform: translate3d(panX, panY, 0) scale(z) with origin (0, 0)
					const screenCenterX = panX + centerX * z;
					const screenCenterY = panY + centerY * z;

					const expectedVpCenterX = vp.vpWidth / 2;
					const expectedVpCenterY = vp.vpHeight / 2;

					// Difference must be strictly within 0.5px due to Math.round()
					const diffX = Math.abs(screenCenterX - expectedVpCenterX);
					const diffY = Math.abs(screenCenterY - expectedVpCenterY);

					expect(diffX).toBeLessThanOrEqual(0.5);
					expect(diffY).toBeLessThanOrEqual(0.5);
				}
			}
		}
	});

	it('verifies the auto-readability zoom threshold (zoom < 0.8 is boosted to 0.95)', () => {
		const targetX = 500;
		const targetY = 300;
		const width = 380;
		const height = 260;
		const vpWidth = 1920;
		const vpHeight = 1080;

		// 1. When zoom is very low (e.g. 0.2x overview), auto-boost to 0.95
		const lowZoomRes = computeFlightCoordinates(
			targetX,
			targetY,
			width,
			height,
			0.2,
			vpWidth,
			vpHeight,
			/* autoReadabilityClamp= */ true
		);
		expect(lowZoomRes.effectiveZoom).toBe(0.95);

		// Verified that with effectiveZoom 0.95, card is perfectly centered
		const screenLowX = lowZoomRes.panX + lowZoomRes.centerX * lowZoomRes.effectiveZoom;
		const screenLowY = lowZoomRes.panY + lowZoomRes.centerY * lowZoomRes.effectiveZoom;
		expect(Math.abs(screenLowX - vpWidth / 2)).toBeLessThanOrEqual(0.5);
		expect(Math.abs(screenLowY - vpHeight / 2)).toBeLessThanOrEqual(0.5);

		// 2. When zoom is >= 0.8 (e.g. 1.25x), preserve current zoom
		const normalZoomRes = computeFlightCoordinates(
			targetX,
			targetY,
			width,
			height,
			1.25,
			vpWidth,
			vpHeight,
			/* autoReadabilityClamp= */ true
		);
		expect(normalZoomRes.effectiveZoom).toBe(1.25);

		const screenNormalX = normalZoomRes.panX + normalZoomRes.centerX * normalZoomRes.effectiveZoom;
		const screenNormalY = normalZoomRes.panY + normalZoomRes.centerY * normalZoomRes.effectiveZoom;
		expect(Math.abs(screenNormalX - vpWidth / 2)).toBeLessThanOrEqual(0.5);
		expect(Math.abs(screenNormalY - vpHeight / 2)).toBeLessThanOrEqual(0.5);

		// 3. When zoom is high (3.0x), preserve current zoom and center accurately
		const highZoomRes = computeFlightCoordinates(
			targetX,
			targetY,
			width,
			height,
			3.0,
			vpWidth,
			vpHeight,
			/* autoReadabilityClamp= */ true
		);
		expect(highZoomRes.effectiveZoom).toBe(3.0);

		const screenHighX = highZoomRes.panX + highZoomRes.centerX * highZoomRes.effectiveZoom;
		const screenHighY = highZoomRes.panY + highZoomRes.centerY * highZoomRes.effectiveZoom;
		expect(Math.abs(screenHighX - vpWidth / 2)).toBeLessThanOrEqual(0.5);
		expect(Math.abs(screenHighY - vpHeight / 2)).toBeLessThanOrEqual(0.5);
	});

	it('verifies bidirectional screen-to-world and world-to-screen inverse transformation', () => {
		const panX = 145;
		const panY = -80;
		const zoom = 1.4;

		const worldPoints = [
			{ x: 0, y: 0 },
			{ x: 450, y: 320 },
			{ x: -1200, y: 800 },
			{ x: 2800, y: -450 }
		];

		for (const pt of worldPoints) {
			// World to Screen
			const screenX = panX + pt.x * zoom;
			const screenY = panY + pt.y * zoom;

			// Inverse Screen to World (matching InfiniteCanvasWorkspace line 1402)
			const reconstructedWorldX = (screenX - panX) / zoom;
			const reconstructedWorldY = (screenY - panY) / zoom;

			expect(reconstructedWorldX).toBeCloseTo(pt.x, 8);
			expect(reconstructedWorldY).toBeCloseTo(pt.y, 8);
		}
	});
});
